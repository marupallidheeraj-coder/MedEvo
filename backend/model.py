"""Image model (ResNet18) + Grad-CAM helpers shared by train.py and app.py.

The Grad-CAM grid pooling lives in heatmap.py (NumPy only, unit-tested without PyTorch).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

TF = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(pretrained: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(512, 2)          # class 0 = NORMAL, class 1 = PNEUMONIA
    return m


def load_model(path: str, device: str = "cpu"):
    """Returns (model, trained_flag). Falls back to untrained weights if file is missing."""
    m = build_model(pretrained=False)
    trained = os.path.exists(path)
    if trained:
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))   # no pickle code execution
    m.to(device).eval()
    return m, trained


def gradcam(model: nn.Module, x: torch.Tensor, cls: int = 1) -> np.ndarray:
    """x: (3,224,224) tensor. Returns a 224x224 float array in [0,1]."""
    acts, grads = [], []

    def fwd(_m, _i, o):
        acts.append(o)
        o.register_hook(lambda g: grads.append(g))

    h = model.layer4.register_forward_hook(fwd)
    model.eval()
    model.zero_grad()
    dev = next(model.parameters()).device
    out = model(x.unsqueeze(0).to(dev))
    out[0, cls].backward()
    h.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def predict(model: nn.Module, pil_img):
    """Returns (pneumonia_probability, 224x224 Grad-CAM array)."""
    x = TF(pil_img.convert("RGB"))
    dev = next(model.parameters()).device
    with torch.no_grad():
        p = F.softmax(model(x.unsqueeze(0).to(dev)), 1)[0, 1].item()
    cam = gradcam(model, x, cls=1)
    return p, cam
