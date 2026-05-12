# KBO Attendance Prediction

KBO 경기 관중 수를 예측하고, Streamlit 화면에서 예측 결과와 최근 흐름을 확인할 수 있는 프로젝트입니다.

## 배포 주소

- Streamlit 앱: https://kbo-attendance-prediction.streamlit.app/

## 실행 화면

![Streamlit 실행 화면](assets/frontend-preview.svg)

## 주요 기능

- 경기별 예상 관중 수 예측
- 경기 조건 기반 Dense 예측
- 최근 홈경기 흐름 기반 LSTM 분석
- 최근 홈경기 흐름 기반 GRU 분석
- 최근 5경기 관중 추이 시각화
- 팀별 홈경기 흐름 확인

## 프로젝트 구성

- `app.py`: Streamlit 기반 메인 대시보드
- `train_all_models.py`: Dense, LSTM, GRU 모델 학습 스크립트
- `data/kbo_attendance.csv`: 2024-2026 시즌 경기 관중 데이터
- `models/`: 학습된 모델 파일
- `artifacts/`: 인코더, 스케일러, 비교 지표 등 예측 보조 파일
- `baseball_attendance_analysis.ipynb`: 전처리, 시각화, Dense 모델 실험 노트북
- `assets/frontend-preview.svg`: README용 프론트 화면 미리보기 이미지

## 프로젝트 구조

```text
kbo_attendance_prediction/
|-- app.py
|-- train_all_models.py
|-- baseball_attendance_analysis.ipynb
|-- requirements.txt
|-- README.md
|-- assets/
|   `-- frontend-preview.svg
|-- data/
|   `-- kbo_attendance.csv
|-- models/
|   |-- attendance_dense_model.keras
|   |-- attendance_lstm_model.keras
|   `-- attendance_gru_model.keras
`-- artifacts/
    |-- encoders.pkl
    |-- scaler.pkl
    |-- feature_cols.json
    |-- model_compare.csv
    |-- model_compare.json
    |-- sequence_meta.json
    |-- sequence_recent_games.csv
    `-- training_history.json
```

## 사용 특징

- 홈팀
- 원정팀
- 구장
- 월
- 요일
- 주말 여부
- 공휴일 여부
- 라이벌전 여부
- 시즌

## 로컬 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit은 실행 환경에 따라 `8501`, `8502`, `8503` 등 다른 포트로 열릴 수 있습니다.
