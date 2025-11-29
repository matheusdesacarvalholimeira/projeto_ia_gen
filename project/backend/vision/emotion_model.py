# backend/vision/emotion_model.py
from deepface import DeepFace
from PIL import Image
import numpy as np
import io


# Funções simples para inferência de emoção / embedding facial


def image_bytes_to_pil(image_bytes: bytes) -> Image.Image:
return Image.open(io.BytesIO(image_bytes)).convert("RGB")




def get_face_embedding(image_bytes: bytes, model_name: str = "Facenet") -> np.ndarray:

img = image_bytes_to_pil(image_bytes)
img_arr = np.asarray(img)
try:
reps = DeepFace.represent(img_arr, model_name=model_name, enforce_detection=True)

if isinstance(reps, list) and len(reps) > 0:
emb = np.array(reps[0]["embedding"]) if isinstance(reps[0], dict) and "embedding" in reps[0] else np.array(reps[0])
elif isinstance(reps, dict) and "embedding" in reps:
emb = np.array(reps["embedding"])
else:
emb = np.array(reps)
return emb
except Exception as e:
print(f"[vision] DeepFace erro: {e}")
return None




def get_face_emotion_label(image_bytes: bytes) -> str:
img = image_bytes_to_pil(image_bytes)
img_arr = np.asarray(img)
try:
analysis = DeepFace.analyze(img_arr, actions=['emotion'], enforce_detection=True)


if isinstance(analysis, list):
analysis = analysis[0]
emotion = analysis.get('dominant_emotion', 'neutral')
return emotion
except Exception as e:
print(f"[vision] DeepFace.analyze erro: {e}")
return "unknown"