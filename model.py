"""EfficientNet-B3 transfer learning with an ORDINAL head: K-1 cumulative logits, P(grade > k)."""
import numpy as np, torch, torch.nn as nn, cv2
from torchvision import models
K = 5
GRADES = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class RetinaNet(nn.Module):
    def __init__(s, pretrained=True):
        super().__init__()
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        s.features, s.pool, s.drop = m.features, nn.AdaptiveAvgPool2d(1), nn.Dropout(.3)
        s.fc = nn.Linear(1536, K - 1)
        s.T = nn.Parameter(torch.ones(1), requires_grad=False)  # temperature (calibration)
    def forward(s, x, feats=False):
        f = s.features(x); z = s.fc(s.drop(s.pool(f).flatten(1))) / s.T
        return (z, f) if feats else z

def class_probs(z):
    c = torch.cummin(torch.sigmoid(z), 1).values          # monotone P(y>k)
    one, zero = torch.ones(len(c), 1, device=c.device), torch.zeros(len(c), 1, device=c.device)
    cc = torch.cat([one, c, zero], 1); return (cc[:, :-1] - cc[:, 1:]).clamp_min(0)

def to_tensor(bgr):
    x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    x = (x - MEAN) / STD; return torch.from_numpy(x.transpose(2, 0, 1)).float()[None]
