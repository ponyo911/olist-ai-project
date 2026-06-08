# part7_sentiment_app.py

import gradio as gr
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 모델 불러오기
model = load_model("model/part7_sentiment_model.h5")

# tokenizer 불러오기
with open("model/part7_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 감성 분석 함수
def predict_sentiment(text):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding="post")
    pred   = model.predict(padded)[0][0]

    if pred >= 0.5:
        return f"긍정 리뷰 ({pred:.2f})"
    return f"부정 리뷰 ({pred:.2f})"

# Gradio 화면
demo = gr.Interface(
    fn=predict_sentiment,
    inputs="text",
    outputs="text",
    title="PART7 감성 분석",
    description="""
포르투갈어 리뷰를 입력하면 긍정 / 부정을 예측합니다.

테스트용 리뷰 예시

긍정 리뷰:
- O produto chegou rápido e em ótimo estado.
- Excelente qualidade e ótimo custo-benefício.
- Gostei muito do produto e recomendo.

부정 리뷰:
- O produto veio danificado.
- A entrega demorou muito.
- Qualidade abaixo do esperado.
"""
)

# 실행
demo.launch()