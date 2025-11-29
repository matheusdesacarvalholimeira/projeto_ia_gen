# backend/api.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.vision.emotion_model import get_face_embedding, get_face_emotion_label
from backend.nlp.text_model import TextModel
from backend.fusion.fusion_model import load_mlp, predict_engagement
from backend.llm.recommend import generate_recommendation
import io
import numpy as np


app = FastAPI()
app.add_middleware(
CORSMiddleware,
allow_origins=['*'],
allow_methods=['*'],
allow_headers=['*']
)


# Inicializações (singletons simples)
text_model = TextModel()
# ajustar input_dim conforme embeddings reais
mlp_model = load_mlp('models/mlp_engagement.pt', input_dim=512+768)


@app.post('/vision/emotion')
async def vision_emotion(file: UploadFile = File(...)):
bytes_img = await file.read()
emb = get_face_embedding(bytes_img)
label = get_face_emotion_label(bytes_img)
emb_list = emb.tolist() if emb is not None else None
return {'embedding': emb_list, 'emotion': label}


@app.post('/nlp/analyze')
async def nlp_analyze(text: str = Form(...)):
emb = text_model.embed(text)
label = text_model.predict_sentiment_label(text)
return {'embedding': emb.tolist(), 'sentiment': label}


@app.post('/fusion/engagement')
async def fusion_engagement(
face_embedding: dict = None,
text_embedding: dict = None
):
# face_embedding and text_embedding expected as lists
v = np.array(face_embedding) if face_embedding is not None