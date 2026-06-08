# part6_recommend_app.py

import gradio as gr
import pickle

# 저장된 추천 모델 불러오기
with open(
    "model/part6_recommend_model.pkl",
    "rb"
) as f:
    model_data = pickle.load(f)

# 저장된 데이터 가져오기
recommend_df      = model_data["recommend_df"]
similarity_matrix = model_data["similarity_matrix"]
product_to_index  = model_data["product_to_index"]

# 드롭다운 상품 목록 생성
product_options = []

for _, row in recommend_df.head(100).iterrows():
    english_category = row["product_category_name_english"]
    product_id       = row["product_id"]
    label = f"{english_category} | {product_id[:8]}"
    value = product_id
    product_options.append((label, value))

# 추천 함수
def recommend_product(product_id):
    idx = product_to_index[product_id]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]

    result = ""
    for rank, (i, score) in enumerate(similarity_scores, start=1):
        category             = recommend_df.iloc[i]["product_category_name_english"]
        recommended_product  = recommend_df.iloc[i]["product_id"]
        result += (
            f"추천 상품 {rank}\n"
            f"카테고리 : {category}\n"
            f"상품ID : {recommended_product[:8]}\n"
            f"유사도 : {score:.2f}\n"
            f"{'-'*30}\n"
        )
    return result

# Gradio 화면 생성
demo = gr.Interface(
    fn=recommend_product,
    inputs=gr.Dropdown(choices=product_options, label="상품 선택"),
    outputs=gr.Textbox(label="추천 결과", lines=15),
    title="PART6 추천 시스템",
    description="상품을 선택하면 비슷한 상품을 추천합니다."
)

# 실행
demo.launch()