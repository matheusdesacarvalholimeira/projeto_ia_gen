# backend/nlp/text_model.py
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


# Sugestão: usar "neuralmind/bert-base-portuguese-cased" (BERTimbau) ou outra variante
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"


class TextModel:
def __init__(self, model_name=MODEL_NAME, device=None):
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
self.model = AutoModel.from_pretrained(model_name)
self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
self.model.to(self.device)
self.model.eval()


def embed(self, text: str) -> np.ndarray:
with torch.no_grad():
inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
inputs = {k: v.to(self.device) for k, v in inputs.items()}
outputs = self.model(**inputs)
# média dos tokens (pooling simples)
last_hidden = outputs.last_hidden_state # (batch, seq_len, hidden)
attn_mask = inputs['attention_mask'].unsqueeze(-1)
summed = (last_hidden * attn_mask).sum(1)
counts = attn_mask.sum(1).clamp(min=1e-9)
pooled = (summed / counts).squeeze(0).cpu().numpy()
return pooled


def predict_sentiment_label(self, text: str) -> str:
# Como protótipo: usar heurística simples baseada em polaridade de palavras
# (para produção, treine um classificador supervisionado)
lower = text.lower()
positive = ['bom', 'ótimo', 'legal', 'gostei', 'feliz', 'interessante']
negative = ['ruim', 'triste', 'chato', 'não gostei', 'difícil', 'frustrado']
score = 0
for w in positive:
if w in lower:
score += 1
for w in negative:
if w in lower:
score -= 1
if score > 0:
return 'positivo'
elif score < 0:
return 'negativo'
else:
return 'neutro'