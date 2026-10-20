"""FastAPI backend: quality gate -> ordinal grade -> calibrated confidence -> MC-dropout uncertainty -> Grad-CAM + lesion candidates.
Run: uvicorn server:app --port 8000  (serves frontend/ at /)"""
import base64, hashlib, json, os, sqlite3, time, subprocess, threading, sys, secrets, ipaddress
import cv2, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from fastapi.staticfiles import StaticFiles
from model import RetinaNet, class_probs, to_tensor, K
from quality import quality, preprocess, crop_box
from lesions import find_lesions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(os.environ.get('RG_WEIGHTS', os.path.join(BASE_DIR, 'outputs/best.pt')))
MET = os.path.join(BASE_DIR, 'outputs/metrics.json')
DB_PATH = os.path.abspath(os.environ.get('RG_DB_PATH', os.path.join(BASE_DIR, 'outputs/app.db')))
DATA_ROOT = os.path.realpath(os.environ.get('RG_DATA_ROOT', os.path.join(BASE_DIR, 'data')))
API_TOKEN = os.environ.get('RG_API_TOKEN', '').strip()
MAX_UPLOAD_BYTES = int(os.environ.get('RG_MAX_UPLOAD_BYTES', '20000000'))
MAX_IMAGE_PIXELS = int(os.environ.get('RG_MAX_IMAGE_PIXELS', '40000000'))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; model, NAME, SIZE = None, '', 512
MODEL_STATUS = 'missing_weights'; MODEL_SHA256 = None

def load_checkpoint(path):
    # weights_only prevents arbitrary pickle object loading on supported PyTorch versions.
    if not os.path.isfile(path): return None, '', 512, None
    ck = torch.load(path, map_location=dev, weights_only=True)
    if not isinstance(ck, dict) or not isinstance(ck.get('state'), dict):
        raise ValueError('Checkpoint must contain a state dictionary.')
    candidate = RetinaNet(False).to(dev)
    candidate.load_state_dict(ck['state'], strict=True)
    candidate.eval()
    size = int(ck.get('size', 512))
    if size not in (256, 384, 512, 768, 1024): raise ValueError('Checkpoint image size is unsupported.')
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''): digest.update(chunk)
    return candidate, str(ck.get('name', 'Unnamed research model'))[:120], size, digest.hexdigest()

if os.path.isfile(W):
    try:
        model, NAME, SIZE, MODEL_SHA256 = load_checkpoint(W); MODEL_STATUS = 'ready'
    except Exception as exc:
        model, NAME, SIZE, MODEL_SHA256, MODEL_STATUS = None, '', 512, None, 'invalid_checkpoint'
        print(f'WARNING: model checkpoint rejected; inference disabled ({type(exc).__name__}).')
app = FastAPI(title='RetinaGuard Triage API', description='Research prototype; not for clinical diagnosis.')

@app.middleware('http')
async def api_security_boundary(request: Request, call_next):
    # Local-only by default. Remote API access requires a configured bearer token.
    if request.url.path.startswith('/api/') and request.url.path != '/api/health':
        host = request.client.host if request.client else ''
        try: local = ipaddress.ip_address(host).is_loopback
        except ValueError: local = host in ('testclient', 'localhost')
        if API_TOKEN:
            if not secrets.compare_digest(request.headers.get('authorization', ''), 'Bearer ' + API_TOKEN):
                from fastapi.responses import JSONResponse
                return JSONResponse({'detail': 'Authentication required.'}, status_code=401)
        elif not local:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'Remote API disabled. Configure RG_API_TOKEN or use localhost.'}, status_code=403)
    return await call_next(request)
os.makedirs('outputs', exist_ok=True)
TRAIN_LOCK = threading.Lock()
TRAIN_PROC = None
TRAIN_LOG = ''
TRAIN_LOG_PATH = 'outputs/train.log'
TRAIN_RELOADED = False

def db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.execute('pragma journal_mode=WAL'); c.execute('pragma busy_timeout=10000')
    c.execute('create table if not exists state(k text primary key, v text)')
    c.execute('create table if not exists audit(id integer primary key autoincrement, t real not null, ev text not null, detail text not null)')
    return c

def log(ev, d=''):
    # Never pass filenames, image bytes, credentials or user-supplied free text here.
    safe_detail = str(d).replace('\n', ' ')[:300]
    c = db(); c.execute('insert into audit(t,ev,detail) values(?,?,?)', (time.time(), str(ev)[:80], safe_detail)); c.commit(); c.close()

@app.get('/api/state')
def get_state():
    c = db(); r = c.execute("select v from state where k='s'").fetchone(); c.close(); return json.loads(r[0]) if r else None

@app.put('/api/state')
async def put_state(req: Request):
    raw = await req.body()
    if len(raw) > 2_000_000: raise HTTPException(413, 'State payload exceeds 2 MB.')
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict): raise ValueError('State must be a JSON object.')
    except (json.JSONDecodeError, ValueError) as e: raise HTTPException(400, str(e))
    # Persist UI metadata/settings but never persist retinal image bytes or data URLs in SQLite.
    def remove_image_payloads(value):
        if isinstance(value, dict):
            cleaned = {}
            for key, item in value.items():
                if str(key).lower() in {'img', 'image', 'image_data', 'heat', 'heatmap', 'thumbnail'}:
                    continue
                cleaned[key] = remove_image_payloads(item)
            return cleaned
        if isinstance(value, list): return [remove_image_payloads(item) for item in value]
        if isinstance(value, str) and value.startswith('data:image/'): return None
        return value
    safe_state = remove_image_payloads(parsed)
    b = json.dumps(safe_state, ensure_ascii=False); c = db(); c.execute("insert or replace into state values('s',?)", (b,)); c.commit(); c.close(); log('state_saved', 'UI metadata/settings only; image payloads excluded'); return dict(ok=True, images_persisted=False)

@app.get('/api/audit')
def audit():
    c = db(); r = c.execute('select t,ev,detail from audit order by t desc limit 200').fetchall(); c.close(); return [dict(t=a, event=b, detail=d) for a, b, d in r]

@app.get('/api/health')
def health(): return dict(model_loaded=model is not None, model_status=MODEL_STATUS, model_sha256=MODEL_SHA256, device=dev, model=NAME, mode='trained' if model is not None else MODEL_STATUS, training_running=TRAIN_PROC is not None and TRAIN_PROC.poll() is None)

@app.post('/api/model/upload')
async def upload_model(file: UploadFile = File(...)):
    global model, NAME, SIZE, W, MODEL_STATUS, MODEL_SHA256
    if not file.filename or not file.filename.lower().endswith(('.pt', '.pth')): raise HTTPException(415, 'Upload a trusted RetinaGuard .pt or .pth checkpoint.')
    blob = await file.read(500_000_001)
    if len(blob) > 500_000_000: raise HTTPException(413, 'Model checkpoint exceeds 500 MB.')
    temp = W + '.incoming'
    try:
        with open(temp, 'wb') as f: f.write(blob)
        ck = torch.load(temp, map_location=dev, weights_only=True)
        if not isinstance(ck, dict) or not isinstance(ck.get('state'), dict): raise ValueError('Checkpoint must contain a state dictionary.')
        candidate = RetinaNet(False).to(dev); candidate.load_state_dict(ck['state'], strict=True); candidate.eval()
        os.replace(temp, W); model, NAME, SIZE = candidate, str(ck.get('name', 'Uploaded RetinaGuard model'))[:120], int(ck.get('size', 512))
        if SIZE not in (256,384,512,768,1024): SIZE = 512
        MODEL_SHA256 = hashlib.sha256(blob).hexdigest(); MODEL_STATUS = 'ready'
        log('model_uploaded', 'checkpoint validated; provenance must be reviewed')
        return {'ok': True, 'model': NAME, 'size': SIZE, 'message': 'Checkpoint validated and loaded. Confirm provenance and evaluation metrics before use.'}
    except Exception as e:
        try: os.remove(temp)
        except OSError: pass
        raise HTTPException(422, f'Checkpoint validation failed: {e}')

@app.post('/api/model/export')
def export_model():
    if not os.path.isfile(W): raise HTTPException(409, 'No trained checkpoint is available to export.')
    out = os.path.abspath('outputs/model.onnx')
    try:
        p = subprocess.run([sys.executable, 'export.py', '--weights', os.path.abspath(W), '--out', out], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, timeout=300)
        if p.returncode != 0: raise RuntimeError((p.stderr or p.stdout or 'ONNX export failed')[-1500:])
        log('model_exported', out)
        return FileResponse(out, media_type='application/onnx', filename='retinaguard_model.onnx')
    except Exception as e: raise HTTPException(500, f'Model export failed. Ensure ONNX dependencies are installed: {e}')

@app.get('/api/dataset/status')
def dataset_status():
    return {'aptos_available': os.path.isdir('data/aptos') and any(os.scandir('data/aptos')), 'idrid_available': os.path.isdir('data/idrid') and any(os.scandir('data/idrid')), 'weights_available': os.path.isfile(W), 'metrics_available': os.path.isfile(MET), 'note': 'Dataset availability is checked on the server; dataset licences and access terms remain the operator’s responsibility.'}

@app.get('/api/metrics')
def metrics(): return json.load(open(MET)) if os.path.isfile(MET) else None

def mc_dropout(x, n=12):
    model.eval(); model.drop.train(); P = []
    with torch.no_grad():
        for _ in range(n): P.append(class_probs(model(x))[0].cpu().numpy())
    model.drop.eval(); P = np.array(P); mean = P.mean(0); grade_samples = (P * np.arange(K)).sum(1)
    entropy = lambda p: float(-np.sum(np.clip(p, 1e-9, 1) * np.log(np.clip(p, 1e-9, 1))))
    expected_entropy = float(np.mean([entropy(p) for p in P]))
    return mean, float(grade_samples.std()), {'samples': int(n), 'predictive_entropy': entropy(mean), 'expected_entropy': expected_entropy, 'mutual_information': max(0.0, entropy(mean) - expected_entropy)}

def gradcam(x):
    with torch.enable_grad():
        model.eval(); z, f = model(x.requires_grad_(True), feats=True); f.retain_grad(); torch.sigmoid(z).sum().backward()
        cam = torch.relu((f.grad.mean((2, 3), keepdim=True) * f).sum(1))[0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

@app.get('/api/train/status')
def train_status():
    global TRAIN_PROC, TRAIN_LOG, TRAIN_RELOADED, model, NAME, SIZE, MODEL_STATUS, MODEL_SHA256
    running = TRAIN_PROC is not None and TRAIN_PROC.poll() is None
    if os.path.isfile(TRAIN_LOG_PATH):
        try:
            with open(TRAIN_LOG_PATH, 'r', errors='replace') as f: TRAIN_LOG = f.read()[-12000:]
        except Exception: pass
    if TRAIN_PROC is not None and not running and TRAIN_PROC.returncode == 0 and not TRAIN_RELOADED and os.path.isfile(W):
        try:
            new_model, new_name, new_size, new_hash = load_checkpoint(W)
            model, NAME, SIZE, MODEL_SHA256, MODEL_STATUS, TRAIN_RELOADED = new_model, new_name, new_size, new_hash, 'ready', True
            log('training_completed', NAME)
        except Exception as e: TRAIN_LOG += '\nModel reload failed: ' + str(e)
    return dict(running=running, return_code=None if running or TRAIN_PROC is None else TRAIN_PROC.returncode, model_loaded=model is not None, model=NAME, log=TRAIN_LOG)

@app.post('/api/train')
async def start_train(req: Request):
    global TRAIN_PROC, TRAIN_LOG, TRAIN_LOG_PATH, TRAIN_RELOADED
    raw_body = await req.body()
    if len(raw_body) > 1_000_000: raise HTTPException(413, 'Training request exceeds 1 MB.')
    try: body = json.loads(raw_body)
    except Exception: raise HTTPException(400, 'Training request must be valid JSON.')
    if not isinstance(body, dict): raise HTTPException(400, 'Training request must be a JSON object.')
    csv = os.path.realpath(str(body.get('csv', ''))); images = os.path.realpath(str(body.get('images', '')))
    # Prevent arbitrary server filesystem reads through the training endpoint.
    def inside_root(path):
        try: return os.path.commonpath([DATA_ROOT, path]) == DATA_ROOT
        except ValueError: return False
    if not inside_root(csv) or not inside_root(images):
        raise HTTPException(403, 'Training inputs must be inside the configured data root.')
    try: epochs = max(1, min(100, int(body.get('epochs', 15))))
    except Exception: raise HTTPException(400, 'epochs must be an integer from 1 to 100')
    if not os.path.isfile(csv): raise HTTPException(400, 'Dataset CSV was not found on the server.')
    if not os.path.isdir(images): raise HTTPException(400, 'Image directory was not found on the server.')
    if TRAIN_PROC is not None and TRAIN_PROC.poll() is None: raise HTTPException(409, 'Training is already running.')
    cmd = [sys.executable, 'train.py', '--train', f'aptos|{csv}|{images}', '--epochs', str(epochs), '--out', 'outputs']
    try:
        TRAIN_LOG = ''; TRAIN_RELOADED = False
        TRAIN_LOG_PATH = os.path.join('outputs', 'train.log')
        log_handle = open(TRAIN_LOG_PATH, 'w')
        TRAIN_PROC = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    except Exception as e: raise HTTPException(500, f'Could not start training: {e}')
    log('training_started', f'epochs={epochs}')
    return dict(ok=True, message='Training started. It uses the dataset on this server; real results require real labelled retinal images.', pid=TRAIN_PROC.pid)

@app.post('/api/analyze')
async def analyze(file: UploadFile = File(...)):
    blob = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(blob) > MAX_UPLOAD_BYTES: raise HTTPException(413, f'Image exceeds the {MAX_UPLOAD_BYTES // 1000000} MB upload limit.')
    if file.content_type and not file.content_type.lower().startswith('image/'): raise HTTPException(415, 'Upload must be an image.')
    raw = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    if raw is None or raw.ndim != 3: raise HTTPException(400, 'Uploaded content is not a decodable color image.')
    if raw.shape[0] < 32 or raw.shape[1] < 32: raise HTTPException(422, 'Image dimensions are too small.')
    if raw.shape[0] * raw.shape[1] > MAX_IMAGE_PIXELS: raise HTTPException(413, 'Image dimensions exceed the pixel limit.')
    q = quality(raw)
    if min(q.values()) < 50:
        raise HTTPException(422, detail={'message': 'Image quality gate failed. Retake or select a clearer, well-lit fundus image.', 'quality': q})
    if model is None: raise HTTPException(503, 'No trained weights at %s. Run train.py first.' % W)
    H, Wd = raw.shape[:2]; x0, y0, cw, ch = crop_box(raw); img = preprocess(raw, SIZE)
    x = to_tensor(img).to(dev); P, unc, unc_detail = mc_dropout(x); c = np.array([P[k + 1:].sum() for k in range(K - 1)]); g = int((c > .5).sum())
    cam = cv2.resize(gradcam(x.clone()), (cw, ch)); full = np.zeros((H, Wd), np.float32); full[y0:y0 + ch, x0:x0 + cw] = cam
    col = cv2.applyColorMap((full * 255).astype(np.uint8), cv2.COLORMAP_JET); rgba = np.dstack([col, (full * 255).astype(np.uint8)])
    rgba = cv2.resize(rgba, (384, 384)); heat = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', rgba)[1]).decode()
    les = [dict(t=t, x=((x0 + px / SIZE * cw) / Wd - .5) * 2, y=((y0 + py / SIZE * ch) / H - .5) * 2) for t, px, py in find_lesions(preprocess(raw, 512))]
    log('analyze', f'grade={g}; quality_ok={min(q.values()) >= 50}; model={NAME[:80]}')
    return dict(grade=g, generated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), model_sha256=MODEL_SHA256, grade_name=['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'][g], probs=P.tolist(),
                confidence=float(P[g]), uncertainty=unc, uncertainty_detail=unc_detail, quality=q, gate_passed=min(q.values()) >= 50, model=NAME,
                lesions=les, lesion_counts=[sum(l['t'] == t for l in les) for t in range(3)], heatmap=heat,
                disclaimer='Research prototype. Not a medical device. Lesion markers are heuristic candidates.')

@app.post('/api/report')
async def create_report(req: Request):
    """Create a PDF research report from a supplied analysis result; does not infer a diagnosis."""
    try:
        raw = await req.body()
        if len(raw) > 1_000_000: raise ValueError('Report request exceeds 1 MB')
        data = json.loads(raw)
        if not isinstance(data, dict): raise ValueError('JSON object required')
    except Exception as e: raise HTTPException(400, f'Invalid report payload: {e}')
    grade = data.get('grade', 'Not provided')
    try:
        grade = int(grade)
        if grade not in range(5): raise ValueError()
    except Exception:
        raise HTTPException(422, 'grade must be an integer from 0 to 4')
    names = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
    buf = BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet(); story = [Paragraph('RetinaGuard Triage — Research Report', styles['Title']), Spacer(1, 8)]
    story.append(Paragraph('Research prototype only. This output is not a diagnosis or a substitute for review by a qualified eye-care professional.', styles['BodyText'])); story.append(Spacer(1, 12))
    source = str(data.get('source', 'sample_demo'))[:80]
    source_label = 'BACKEND INFERENCE (UNVERIFIED)' if source == 'backend_inference' else 'SAMPLE / CLIENT-SUPPLIED DATA — NOT A VERIFIED MODEL PREDICTION'
    rows = [['Field', 'Value'], ['Result source', source_label], ['Record ID', str(data.get('record_id', 'Not provided'))[:120]], ['Predicted grade', f'{grade} — {names[grade]}'], ['Model', str(data.get('model', 'Not provided'))[:120]], ['Checkpoint SHA-256', str(data.get('model_sha256') or 'Not provided')[:80]], ['Inference timestamp UTC', str(data.get('generated_at_utc') or 'Not provided')[:40]], ['Confidence', str(data.get('confidence', 'Not provided'))], ['MC-dropout uncertainty', str(data.get('uncertainty', 'Not provided'))], ['Quality gate', str(data.get('gate_passed', 'Not provided'))], ['Image quality scores', json.dumps(data.get('quality', {}))[:300]], ['Lesion candidates', json.dumps(data.get('lesion_counts', []))[:200]], ['Generated UTC', time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())]]
    table = Table(rows, colWidths=[48*mm, 112*mm]); table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#123944')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#b6c8cc')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf5f6')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story += [table, Spacer(1, 12), Paragraph('Explainability caveat: Grad-CAM highlights model attention and does not prove pathology. Heuristic lesion candidates are unvalidated unless separate annotation-based evaluation is supplied.', styles['BodyText'])]
    doc.build(story); buf.seek(0); log('report_pdf_created', f'grade={grade}')
    return StreamingResponse(buf, media_type='application/pdf', headers={'Content-Disposition': 'attachment; filename="retinaguard_research_report.pdf"'})

app.mount('/', StaticFiles(directory='frontend', html=True), name='ui')
