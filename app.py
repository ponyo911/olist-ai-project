# app.py - 4개 앱을 하나로 통합

import gradio as gr
import pickle
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── PART 6. 추천 모델 불러오기 ──────────────────────
with open("model/part6_recommend_model.pkl", "rb") as f:
    model_data = pickle.load(f)

recommend_df      = model_data["recommend_df"]
similarity_matrix = model_data["similarity_matrix"]
product_to_index  = model_data["product_to_index"]

product_options = []
for _, row in recommend_df.head(100).iterrows():
    label = f"{row['product_category_name_english']} | {row['product_id'][:8]}"
    product_options.append((label, row["product_id"]))

def recommend_product(product_id):
    idx = product_to_index[product_id]
    scores = sorted(list(enumerate(similarity_matrix[idx])), key=lambda x: x[1], reverse=True)[1:6]
    result = ""
    for rank, (i, score) in enumerate(scores, start=1):
        category = recommend_df.iloc[i]["product_category_name_english"]
        pid      = recommend_df.iloc[i]["product_id"]
        result  += f"추천 상품 {rank}\n카테고리 : {category}\n상품ID : {pid[:8]}\n유사도 : {score:.2f}\n{'-'*30}\n"
    return result

# ── PART 7. 감성분석 모델 불러오기 ──────────────────────
sentiment_model = load_model("model/part7_sentiment_model.h5")
with open("model/part7_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_sentiment(text):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding="post")
    pred   = sentiment_model.predict(padded)[0][0]
    return f"긍정 리뷰 ({pred:.2f})" if pred >= 0.5 else f"부정 리뷰 ({pred:.2f})"

# ── PART 8. 매출 예측 모델 불러오기 ──────────────────────
sales_model = joblib.load("model/part8_sales_model.pkl")

def predict_sales(price, freight, review_score):
    data = np.array([[price, freight, review_score]])
    pred = sales_model.predict(data)
    return f"예상 결제 금액 : {round(float(pred[0]), 2)} 헤알"

# ── PART 9. 이탈 예측 모델 불러오기 ──────────────────────
churn_model   = joblib.load("model/part9_churn_model.pkl")
feature_names = churn_model.feature_names_in_

def predict_churn(order_count, days):
    input_data = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)
    if "order_count"           in feature_names: input_data["order_count"]           = order_count
    if "days_since_last_order" in feature_names: input_data["days_since_last_order"] = days
    pred = churn_model.predict(input_data)[0]
    return "이탈 가능성 높음" if pred == 1 else "유지 고객"

# ── Gradio 탭 화면 구성 ──────────────────────────────
with gr.Blocks(title="Olist AI 분석 서비스") as demo:
    gr.Markdown("# Olist AI 분석 서비스")

    with gr.Tab("추천 시스템"):
        gr.Interface(
            fn=recommend_product,
            inputs=gr.Dropdown(choices=product_options, label="상품 선택"),
            outputs=gr.Textbox(label="추천 결과", lines=15),
            title="PART6 추천 시스템"
        )

    with gr.Tab("감성 분석"):
        gr.Interface(
            fn=predict_sentiment,
            inputs="text",
            outputs="text",
            title="PART7 감성 분석"
        )

    with gr.Tab("매출 예측"):
        gr.Interface(
            fn=predict_sales,
            inputs=[
                gr.Number(label="상품 가격"),
                gr.Number(label="배송비"),
                gr.Slider(minimum=1, maximum=5, value=5, step=1, label="리뷰 점수")
            ],
            outputs=gr.Textbox(label="예측 결과"),
            title="PART8 매출 예측"
        )

    with gr.Tab("이탈 예측"):
        gr.Interface(
            fn=predict_churn,
            inputs=[
                gr.Number(label="주문 횟수"),
                gr.Number(label="최근 미구매 기간(일)")
            ],
            outputs="text",
            title="PART9 이탈 예측"
        )

demo.launch()