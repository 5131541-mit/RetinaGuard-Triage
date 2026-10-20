"""Export trained weights to ONNX for edge deployment: python export.py [--weights outputs/best.pt] [--out outputs/model.onnx]
Output logits are the K-1 cumulative logits (apply sigmoid + monotone fix as in model.class_probs)."""
import argparse, torch
from model import RetinaNet
ap = argparse.ArgumentParser(); ap.add_argument('--weights', default='outputs/best.pt'); ap.add_argument('--out', default='outputs/model.onnx'); a = ap.parse_args()
ck = torch.load(a.weights, map_location='cpu'); m = RetinaNet(False); m.load_state_dict(ck['state']); m.eval(); s = ck.get('size', 512)
torch.onnx.export(m, torch.randn(1, 3, s, s), a.out, input_names=['image'], output_names=['cum_logits'], dynamic_axes={'image': {0: 'b'}}, opset_version=17)
print('saved', a.out)
