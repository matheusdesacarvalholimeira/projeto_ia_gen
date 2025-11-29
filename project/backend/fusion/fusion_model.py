# backend/fusion/fusion_model.py
import torch
import torch.nn as nn
import numpy as np


class SimpleMLPFusion(nn.Module):
def __init__(self, input_dim: int, hidden_dim: int = 128, n_classes: int = 3):
super().__init__()
self.net = nn.Sequential(
nn.Linear(input_dim, hidden_dim),
nn.ReLU(),
nn.Dropout(0.2),
nn.Linear(hidden_dim, hidden_dim//2),
nn.ReLU(),
nn.Linear(hidden_dim//2, n_classes)
)


def forward(self, x):
return self.net(x)




# Helpers


def load_mlp(path: str, input_dim: int = 512+768, device: str = None):
device = device or ("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLPFusion(input_dim=input_dim)
try:
model.load_state_dict(torch.load(path, map_location=device))
except Exception as e:
print(f"[fusion] Não foi possível carregar o modelo em {path}: {e}")
# continua com pesos aleatórios — útil para protótipo
model.to(device)
model.eval()
return model




def predict_engagement(model, vision_emb: np.ndarray, text_emb: np.ndarray):
# concatena e normaliza (simples)
v = vision_emb if vision_emb is not None else np.zeros(512, dtype=np.float32)
t = text_emb if text_emb is not None else np.zeros(768, dtype=np.float32)
x = np.concatenate([v, t]).astype(np.float32)
x_t = torch.from_numpy(x).unsqueeze(0)
with torch.no_grad():
logits = model(x_t)
probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
idx = int(probs.argmax())
labels = ['baixo', 'médio', 'alto']
return {
'label': labels[idx],
'probs': probs.tolist()
}