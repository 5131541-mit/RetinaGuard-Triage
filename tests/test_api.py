"""FastAPI smoke tests; these do not claim model accuracy or clinical validity."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
import cv2
import numpy as np
import torch
import server

client = TestClient(server.app)

def test_health_and_frontend():
    assert client.get('/api/health').status_code == 200
    assert client.get('/').status_code == 200

def test_state_roundtrip():
    payload = {'test_marker': 'api-smoke', 'recs': [], 'set': {'theme': 'dark'}}
    assert client.put('/api/state', json=payload).status_code == 200
    assert client.get('/api/state').json()['test_marker'] == 'api-smoke'
    assert 'img' not in client.get('/api/state').json().get('recs', [{}])[0] if client.get('/api/state').json().get('recs') else True

def test_invalid_upload_rejected():
    r = client.post('/api/analyze', files={'file': ('not-image.txt', b'not an image', 'text/plain')})
    assert r.status_code == 415

def test_quality_gate_rejects_blank_image():
    ok, enc = cv2.imencode('.png', np.zeros((512, 512, 3), dtype=np.uint8))
    assert ok
    r = client.post('/api/analyze', files={'file': ('blank.png', enc.tobytes(), 'image/png')})
    assert r.status_code == 422

def test_training_rejects_paths_outside_data_root():
    r = client.post('/api/train', json={'csv': '/no/such/train.csv', 'images': '/no/such/images', 'epochs': 1})
    assert r.status_code == 403

def test_training_rejects_missing_dataset_inside_data_root(tmp_path, monkeypatch):
    # The path policy is tested separately from existence checks.
    import server as s
    monkeypatch.setattr(s, 'DATA_ROOT', str(tmp_path))
    r = client.post('/api/train', json={'csv': str(tmp_path/'missing.csv'), 'images': str(tmp_path/'images'), 'epochs': 1})
    assert r.status_code == 400

def test_state_rejects_non_object():
    r = client.put('/api/state', content='[]', headers={'content-type': 'application/json'})
    assert r.status_code == 400

def test_pdf_report_download():
    r = client.post('/api/report', json={'record_id':'test-1','grade':2,'confidence':0.8,'uncertainty':0.1,'quality':{'b':80}})
    assert r.status_code == 200 and r.headers['content-type'] == 'application/pdf'
    assert r.content.startswith(b'%PDF')

def test_pdf_report_rejects_bad_grade():
    assert client.post('/api/report', json={'grade':9}).status_code == 422

def test_dataset_status():
    r = client.get('/api/dataset/status')
    assert r.status_code == 200 and 'weights_available' in r.json()

def test_model_upload_rejects_wrong_extension():
    r = client.post('/api/model/upload', files={'file': ('bad.txt', b'not a checkpoint', 'text/plain')})
    assert r.status_code == 415

def test_export_requires_real_checkpoint():
    if not server.model:
        assert client.post('/api/model/export').status_code == 409


def test_state_does_not_persist_image_payloads():
    payload = {'recs': [{'id': 'case-1', 'img': 'data:image/png;base64,SECRET', 'g': 2}], 'set': {}}
    assert client.put('/api/state', json=payload).status_code == 200
    saved = client.get('/api/state').json()
    assert 'img' not in saved['recs'][0]
    assert 'SECRET' not in str(saved)

def test_remote_api_requires_auth_when_token_configured(monkeypatch):
    monkeypatch.setattr(server, 'API_TOKEN', 'unit-test-secret')
    # TestClient uses a synthetic host but configured tokens must still be checked.
    r = client.get('/api/audit')
    assert r.status_code == 401
    r = client.get('/api/audit', headers={'Authorization': 'Bearer unit-test-secret'})
    assert r.status_code == 200
    monkeypatch.setattr(server, 'API_TOKEN', '')

def test_invalid_checkpoint_is_rejected(tmp_path):
    import torch
    checkpoint = tmp_path / 'invalid.pt'
    torch.save({'not_state': {}}, checkpoint)
    try:
        server.load_checkpoint(str(checkpoint))
        assert False, 'invalid checkpoint should not load'
    except ValueError as exc:
        assert 'state dictionary' in str(exc)


def test_valid_checkpoint_loads_and_is_hashed(tmp_path):
    """Regression: a VALID checkpoint must load (previously failed with NameError: hashlib was not imported)."""
    from model import RetinaNet
    p = tmp_path / 'valid.pt'
    torch.save({'state': RetinaNet(False).state_dict(), 'name': 'test-model', 'size': 256}, p)
    m, name, size, digest = server.load_checkpoint(str(p))
    assert m is not None and name == 'test-model' and size == 256 and len(digest) == 64
