# Olist 이커머스 데이터 기반 AI 프로젝트

> 브라질 실거래 데이터 약 10만 건으로 4개의 비즈니스 문제를 AI로 풀고, 실제 웹 서비스로 배포한 포트폴리오 프로젝트입니다.

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-yellow)](https://huggingface.co/spaces/ponyo911/olist-ai-service)

**실제 작동 데모:** https://huggingface.co/spaces/ponyo911/olist-ai-service
<img width="1260" height="459" alt="image" src="https://github.com/user-attachments/assets/13d96407-1a99-45d1-9b9e-1f26fd407b50" />

---

## 프로젝트 개요

[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)는 브라질의 실제 이커머스 플랫폼으로, 중소상인과 고객을 연결하는 마켓플레이스입니다. <br>2016~2018년 실거래 데이터(약 10만 건)를 활용해 이커머스가 공통으로 겪는 네 가지 문제를 AI로 해결하고,<br> 분석에서 끝나지 않고 누구나 사용할 수 있는 웹 서비스로 배포했습니다.

| 단계 | 내용 |
|------|------|
| 데이터 | 7개 테이블 병합 → 119,143행 × 40열 통합 데이터셋 |
| AI 모델 | 추천 · 감성분석 · 매출예측 · 이탈예측 (4종) |
| 배포 | Gradio 통합 앱 → Hugging Face Spaces |

---

## 사용 기술

- **언어/분석:** Python, pandas, NumPy
- **머신러닝:** scikit-learn (RandomForest, 코사인 유사도)
- **딥러닝/NLP:** TensorFlow/Keras, BERT
- **시각화:** Matplotlib
- **배포:** Gradio, Hugging Face Spaces

---

## 4가지 비즈니스 문제와 해결

### 1. 추천 시스템 — 객단가를 어떻게 올릴까
상품 카테고리를 원핫 인코딩한 뒤 코사인 유사도로 비슷한 상품을 추천합니다.<br>
상세페이지에 유사 상품을 노출해 추가 구매(객단가 상승)를 유도하는 것이 목표입니다.

### 2. 감성 분석 — 리뷰를 자동으로 분류
리뷰 텍스트를 긍정/부정으로 분류하는 딥러닝 모델을 직접 구현했습니다.
- 정확도 **90.31%** (Test Accuracy)
- 불균형 처리(부정 리뷰 업샘플링), 데이터 누수 방지(Tokenizer를 train으로만 학습)
- 사전학습 모델 BERT도 함께 적용
- *한계: 포르투갈어 학습 데이터 → 한국어 입력은 정확도 낮음*

### 3. 매출 예측 — 수요·예산을 미리 계획
RandomForest 회귀로 결제 금액을 예측합니다.
- R² **0.8135**, 평균 오차(MAE) 31.13 BRL
- 이상치(상위 1%) 제거 → 변수 중요도 확인 → 핵심 3개 변수(price, freight_value, review_score)로 모델 단순화

### 4. 이탈 예측 — 떠날 고객을 미리 찾기
고객별 행동을 요약해 이탈 여부를 분류하는 과정에서 **데이터 누수(Data Leakage)를 직접 발견**했습니다.
- 정확도 100%가 나왔으나, 이는 이탈 라벨 생성에 쓴 변수(`order_count`, `days_since_last_order`)를 입력 변수에도 그대로 넣었기 때문
- 분류 문제에서 100%는 거의 항상 함정이라는 점을 인지하고 원인을 규명
- **개선 방향:** 라벨 생성 변수를 입력에서 제외, 라벨을 미래 기간으로 분리해 검증

---

## 📑 포트폴리오 PPT

<!--
  슬라이드 PNG 넣는 방법:
  1) 저장소에 assets 폴더를 만들고 (이미 없다면) slide_01.png ~ slide_21.png 형식으로 저장
  2) 아래 각 줄의 경로가 파일명과 일치하면 GitHub에서 자동으로 보입니다
-->

### 01. 프로젝트 소개
프로젝트 개요와 전체 구조
<br>
<img width="1280" height="720" alt="슬라이드1" src="https://github.com/user-attachments/assets/f54c7e2b-84b5-47bf-9455-662eebe22138" />
<img width="1280" height="720" alt="슬라이드2" src="https://github.com/user-attachments/assets/fd9793ea-79cd-41a0-a942-089958d77aab" />
<img width="1280" height="720" alt="슬라이드3" src="https://github.com/user-attachments/assets/8aad9a41-3e3c-4bf7-b141-17937bd61e6c" />

<br>

### 02. 데이터 준비 — 병합과 탐색(EDA)
7개 테이블을 하나로 합치고, 데이터에서 인사이트 도출
<br>
<img width="1280" height="720" alt="슬라이드4" src="https://github.com/user-attachments/assets/8a521796-6471-4a50-afe8-9ad285dfc545" />
<img width="1280" height="720" alt="슬라이드5" src="https://github.com/user-attachments/assets/4c906084-136b-418f-85ea-252ae61fe851" />
<br>

### 03. 추천 시스템 — 객단가 향상
코사인 유사도 기반 유사 상품 추천
<br>
<img width="1280" height="720" alt="슬라이드6" src="https://github.com/user-attachments/assets/a06220b9-54fe-42d1-adce-f64e61d8b3a6" />
<img width="1280" height="720" alt="슬라이드7" src="https://github.com/user-attachments/assets/fd54f646-2861-4673-a03c-71706adbb9d8" />
<img width="1280" height="720" alt="슬라이드8" src="https://github.com/user-attachments/assets/3aff3081-3c46-42f4-9f46-63b63960e545" />

<br>

### 04. 감성 분석 — 리뷰 자동 분류
딥러닝으로 긍정/부정 분류 (정확도 90.31%)
<br>
<img width="1280" height="720" alt="슬라이드9" src="https://github.com/user-attachments/assets/7c7a7ba4-2da2-4fa5-9fa5-d8f4429a42ad" />
<img width="1280" height="720" alt="슬라이드10" src="https://github.com/user-attachments/assets/8a51974b-60ae-49a1-8132-63f4145e221b" />
<img width="1280" height="720" alt="슬라이드11" src="https://github.com/user-attachments/assets/964f7f93-3217-4b88-a125-640fa2bf4d73" />

<br>

### 05. 매출 예측 — 수요·예산 계획
RandomForest 회귀 (R² 0.81)
<br>
<img width="1280" height="720" alt="슬라이드12" src="https://github.com/user-attachments/assets/a097c7dd-53f7-4e38-a87f-ed8c3e5dc7c8" />
<img width="1280" height="720" alt="슬라이드13" src="https://github.com/user-attachments/assets/8d600579-99f1-4b70-9a07-22501331f6ba" />
<img width="1280" height="720" alt="슬라이드14" src="https://github.com/user-attachments/assets/68f968bd-95c9-4227-b6ca-b202405541d4" />

<br>

### 06. 이탈 예측 — 고객 유지 & 데이터 누수 발견
정확도 100%의 함정을 직접 규명
<br>
<img width="1280" height="720" alt="슬라이드15" src="https://github.com/user-attachments/assets/ab053840-2775-49dc-9a57-4a68e99fafaf" />
<img width="1280" height="720" alt="슬라이드16" src="https://github.com/user-attachments/assets/4579b1b4-9c03-4347-b7cd-19af54c3f7e0" />
<img width="1280" height="720" alt="슬라이드17" src="https://github.com/user-attachments/assets/b0b81dff-e062-4a2b-a91f-78f882d85bb9" />

<br>

### 07. 서비스 배포 — 분석에서 실제 서비스로
Gradio 통합 앱 → Hugging Face 배포
<br>
<img width="1280" height="720" alt="슬라이드18" src="https://github.com/user-attachments/assets/22ae41ee-7690-490a-943d-59a2071f1978" />
<img width="1280" height="720" alt="슬라이드19" src="https://github.com/user-attachments/assets/cfee2b05-2413-4246-b247-4ef5f6e92a52" />
<br>

### 08. 한계와 마무리
개선 방향과 프로젝트로 얻은 것
<br>
<img width="1280" height="720" alt="슬라이드20" src="https://github.com/user-attachments/assets/a998aa6a-69f4-44d7-a423-c42e12fd1487" />
<img width="1280" height="720" alt="슬라이드21" src="https://github.com/user-attachments/assets/2a24e1d2-a9f0-4658-a723-2f87907d49d2" />

---

## 실행 방법

> 데이터(CSV)와 학습된 모델 파일(.h5, .pkl)은 용량이 커 저장소에 포함하지 않았습니다.
> 데이터는 [Kaggle Olist 데이터셋](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)에서 받을 수 있고, 모델은 노트북 실행 시 생성됩니다.

```bash
# 1) 저장소 클론
git clone https://github.com/ponyo911/olist-ai-project.git
cd olist-ai-project

# 2) 필요한 라이브러리 설치
pip install -r requirements.txt

# 3) 로컬에서 통합 앱 실행
python app.py
# 실행 후 브라우저에서 http://127.0.0.1:7860 접속
```

분석 노트북(EDA, 모델 학습)은 `notebooks/` 폴더에 있으며, Google Colab 환경 기준으로 작성되었습니다.

---

## 프로젝트 구조

```
olist-ai-project/
├── notebooks/              # 분석·모델 학습 노트북 (Colab 기준)
│   ├── 04_data_merge.ipynb
│   ├── 05_eda.ipynb
│   ├── 06_recommendation.ipynb
│   ├── 07_sentiment1_Tokenizer_Embedding.ipynb
│   ├── 07_sentiment2_bert.ipynb
│   ├── 08_sales.ipynb
│   └── 09_churn.ipynb
├── app.py                  # 4개 모델 통합 Gradio 앱
├── part6_recommend_app.py  # 추천 시스템 앱
├── part7_sentiment_app.py  # 감성분석 앱
├── part8_sales_app.py      # 매출예측 앱
├── part9_churn_app.py      # 이탈예측 앱
├── requirements.txt
```

---

## 이 프로젝트로 얻은 것

- **전 과정 경험:** 데이터 수집·병합부터 모델링, 웹 서비스 배포까지 직접 수행
- **비즈니스 연결:** 기술 결과를 매출·CS·고객유지 같은 실제 의사결정 언어로 해석
- **비판적 시각:** 정확도 100%를 그대로 받아들이지 않고 데이터 누수를 스스로 발견·검증

---

**개발:** [ponyo911](https://github.com/ponyo911)
