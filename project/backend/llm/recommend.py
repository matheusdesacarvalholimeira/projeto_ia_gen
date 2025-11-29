# backend/llm/recommend.py
import requests


# Implementação simples: usa template para gerar recomendações.
# Em produção, você poderia encaminhar para uma LLM (OpenAI, local Llama, etc.).


def generate_recommendation(engagement_label: str, face_emotion: str, text_sentiment: str) -> str:
# Regras simples de negócio
if engagement_label == 'baixo':
recs = [
'reduzir o ritmo e incluir uma pausa rápida',
'usar um exemplo prático relacionado ao cotidiano do estudante',
'aplicar uma atividade de 5 min para recapitular conceitos-chave'
]
elif engagement_label == 'médio':
recs = [
'incluir perguntas abertas para aumentar a participação',
'fornecer feedback positivo nas próximas interações',
'alternar entre exercícios e explicações curtas'
]
else: # alto
recs = [
'manter o ritmo, oferecer desafios adicionais',
'sugestão de aprofundamento ou atividade prática extra',
'incentivar trabalho em pares para consolidar conhecimento'
]


# Ajuste por emoção/sentimento
if face_emotion in ['sad', 'triste', 'sadness'] or text_sentiment == 'negativo':
recs.insert(0, 'verificar se o estudante está com dificuldades específicas e oferecer reforço individual')


# Formata em texto
return 'Sugestões:\n- ' + '\n- '.join(recs)