from __future__ import annotations

from pathlib import Path
import json
import pickle

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "kbo_attendance.csv"

DENSE_MODEL_PATH = MODEL_DIR / "attendance_dense_model.keras"
LSTM_MODEL_PATH = MODEL_DIR / "attendance_lstm_model.keras"
GRU_MODEL_PATH = MODEL_DIR / "attendance_gru_model.keras"
ENCODERS_FILE = ARTIFACT_DIR / "encoders.pkl"
SCALER_FILE = ARTIFACT_DIR / "scaler.pkl"
FEATURES_FILE = ARTIFACT_DIR / "feature_cols.json"
LSTM_SCALER_FILE = ARTIFACT_DIR / "lstm_target_scaler.pkl"
GRU_SCALER_FILE = ARTIFACT_DIR / "gru_target_scaler.pkl"

TEAM_TO_STADIUM = {
    "LG": "잠실",
    "두산": "잠실",
    "삼성": "대구",
    "KIA": "광주",
    "KT": "수원",
    "SSG": "문학",
    "롯데": "사직",
    "한화": "대전",
    "NC": "창원",
    "키움": "고척",
}
RIVAL_MATCHES = {tuple(sorted(["LG", "두산"]))}
WEEKDAY_TO_NUM = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
TEAMS = ["LG", "두산", "삼성", "KIA", "KT", "SSG", "롯데", "한화", "NC", "키움"]

st.set_page_config(page_title="KBO 관중 수 예측", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef6ff 0%, #f7fbff 40%, #ffffff 100%);
    }
    .hero {
        padding: 28px 32px;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f4c81 0%, #2374b7 55%, #66b2ff 100%);
        color: white;
        box-shadow: 0 16px 40px rgba(20, 68, 120, 0.18);
        margin-bottom: 22px;
    }
    .hero h1 {
        margin: 0 0 10px 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0;
        font-size: 1.02rem;
        line-height: 1.6;
        opacity: 0.96;
    }
    .panel {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(34, 91, 144, 0.10);
        border-radius: 22px;
        padding: 18px 20px 14px 20px;
        box-shadow: 0 12px 32px rgba(36, 78, 126, 0.08);
        backdrop-filter: blur(8px);
        margin-bottom: 18px;
    }
    .mini-card {
        background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 100%);
        border: 1px solid rgba(34, 91, 144, 0.10);
        border-radius: 18px;
        padding: 16px 16px 12px 16px;
        box-shadow: 0 10px 26px rgba(36, 78, 126, 0.06);
        min-height: 104px;
    }
    .mini-label {
        color: #5b6f85;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .mini-value {
        color: #143b63;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .mini-sub {
        color: #6f8194;
        font-size: 0.86rem;
        margin-top: 6px;
    }
    .section-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #183a5b;
        margin: 4px 0 14px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["home_team", "date"]).reset_index(drop=True)


@st.cache_resource
def load_dense_artifacts():
    if not (DENSE_MODEL_PATH.exists() and ENCODERS_FILE.exists() and SCALER_FILE.exists() and FEATURES_FILE.exists()):
        return None, None, None, None
    model = load_model(DENSE_MODEL_PATH)
    with open(ENCODERS_FILE, "rb") as f:
        encoders = pickle.load(f)
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    feature_cols = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))
    return model, encoders, scaler, feature_cols


@st.cache_resource
def load_sequence_artifacts():
    lstm_model = load_model(LSTM_MODEL_PATH) if LSTM_MODEL_PATH.exists() else None
    gru_model = load_model(GRU_MODEL_PATH) if GRU_MODEL_PATH.exists() else None
    lstm_scaler = None
    gru_scaler = None
    if LSTM_SCALER_FILE.exists():
        with open(LSTM_SCALER_FILE, "rb") as f:
            lstm_scaler = pickle.load(f)
    if GRU_SCALER_FILE.exists():
        with open(GRU_SCALER_FILE, "rb") as f:
            gru_scaler = pickle.load(f)
    return lstm_model, gru_model, lstm_scaler, gru_scaler


def render_hero(title: str, body: str):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{body}</p></div>", unsafe_allow_html=True)


def render_cards(items: list[tuple[str, str, str]]):
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='mini-card'><div class='mini-label'>{label}</div><div class='mini-value'>{value}</div><div class='mini-sub'>{sub}</div></div>",
                unsafe_allow_html=True,
            )


def predict_dense(home_team: str, away_team: str, month: int, weekday: str, is_holiday: bool, season: int) -> int | None:
    model, encoders, scaler, feature_cols = load_dense_artifacts()
    if model is None:
        return None
    stadium = TEAM_TO_STADIUM.get(home_team, "잠실")
    weekday_num = WEEKDAY_TO_NUM[weekday]
    is_weekend = int(weekday_num >= 5)
    is_rival_match = int(tuple(sorted([home_team, away_team])) in RIVAL_MATCHES)
    row = {
        "home_team_enc": int(encoders["home_team"].transform([home_team])[0]),
        "away_team_enc": int(encoders["away_team"].transform([away_team])[0]),
        "stadium_enc": int(encoders["stadium"].transform([stadium])[0]),
        "month": month,
        "weekday_num": weekday_num,
        "is_weekend": is_weekend,
        "is_holiday": int(is_holiday),
        "is_rival_match": is_rival_match,
        "season": season,
    }
    X = pd.DataFrame([[row[col] for col in feature_cols]], columns=feature_cols)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled, verbose=0).reshape(-1)[0]
    return max(0, int(round(float(pred))))


def get_recent_sequence(home_team: str, season: int, seq_len: int = 5):
    df = load_data()
    team_df = df[(df["home_team"] == home_team) & (df["season"] == season)].sort_values("date")
    if len(team_df) < seq_len:
        team_df = df[df["home_team"] == home_team].sort_values("date")
    if len(team_df) < seq_len:
        return None, None
    recent = team_df.tail(seq_len)[["date", "away_team", "attendance"]].copy()
    seq = recent["attendance"].astype(float).to_numpy().reshape(1, seq_len, 1)
    return recent, seq


def predict_sequence(model_type: str, home_team: str, season: int):
    lstm_model, gru_model, lstm_scaler, gru_scaler = load_sequence_artifacts()
    recent, seq = get_recent_sequence(home_team, season)
    if recent is None:
        return None, None
    if model_type == "LSTM" and lstm_model is not None and lstm_scaler is not None:
        pred_scaled = lstm_model.predict(seq, verbose=0).reshape(-1)[0]
        pred = lstm_scaler.inverse_transform(np.array([[pred_scaled]])).reshape(-1)[0]
        return max(0, int(round(float(pred)))), recent
    if model_type == "GRU" and gru_model is not None and gru_scaler is not None:
        pred_scaled = gru_model.predict(seq, verbose=0).reshape(-1)[0]
        pred = gru_scaler.inverse_transform(np.array([[pred_scaled]])).reshape(-1)[0]
        return max(0, int(round(float(pred)))), recent
    return None, None


def render_dense_page():
    render_hero(
        "Dense 조건 기반 예측",
        "홈팀, 원정팀, 월, 요일, 공휴일 여부를 조합해 특정 경기의 예상 관중 수를 빠르게 확인합니다.",
    )

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>경기 조건 입력</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("홈팀", TEAMS, key="dense_home")
        weekday = st.selectbox("요일", ["월", "화", "수", "목", "금", "토", "일"], index=5, key="dense_weekday")
        month = st.slider("경기 월 선택", min_value=3, max_value=10, value=7, key="dense_month")
    with c2:
        away_team = st.selectbox("원정팀", TEAMS, index=1, key="dense_away")
        is_holiday = st.toggle("공휴일 경기", value=False, key="dense_holiday")
        season = st.selectbox("시즌", [2024, 2025, 2026], index=2, key="dense_season")

    stadium = TEAM_TO_STADIUM.get(home_team, "미정")
    is_weekend = weekday in ["토", "일"]
    is_rival_match = int(tuple(sorted([home_team, away_team])) in RIVAL_MATCHES)
    st.markdown("</div>", unsafe_allow_html=True)

    render_cards([
        ("자동 매핑 구장", stadium, "홈팀 기준 구장 적용"),
        ("주말 여부", "주말" if is_weekend else "평일", "요일 자동 계산"),
        ("공휴일 여부", "공휴일" if is_holiday else "일반일", "직접 입력 반영"),
        ("라이벌전 여부", "라이벌전" if is_rival_match else "일반 경기", "LG-두산 자동 감지"),
    ])

    if st.button("Dense 예상 관중 확인", use_container_width=True):
        prediction = predict_dense(home_team, away_team, month, weekday, is_holiday, season)
        if prediction is None:
            st.info("Dense 모델 산출물이 없습니다. 먼저 학습을 실행해주세요.")
        else:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>예측 결과</div>", unsafe_allow_html=True)
            render_cards([
                ("예상 관중 수", f"{prediction:,}명", "Dense 회귀 모델 결과"),
                ("홈팀", home_team, f"원정팀 {away_team}"),
                ("경기 조건", f"{season} / {month}월 / {weekday}", "입력 조합 기준"),
            ])
            st.markdown("</div>", unsafe_allow_html=True)


def render_sequence_page(model_type: str):
    render_hero(
        f"{model_type} 시계열 예측",
        "최근 5경기 홈 관중 흐름을 시계열로 보고 다음 홈경기 관중 수를 예측합니다.",
    )

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>시계열 입력 조건</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("홈팀", TEAMS, key=f"{model_type}_home")
    with c2:
        season = st.selectbox("시즌", [2024, 2025, 2026], index=2, key=f"{model_type}_season")
    st.markdown("</div>", unsafe_allow_html=True)

    prediction, recent = predict_sequence(model_type, home_team, season)
    if prediction is None or recent is None:
        st.info("시계열 예측에 필요한 최근 5경기 데이터 또는 모델 산출물이 없습니다.")
        return

    render_cards([
        (f"{model_type} 예상 관중", f"{prediction:,}명", "다음 홈경기 예측값"),
        ("기준 홈팀", home_team, f"시즌 {season}"),
        ("시퀀스 길이", "최근 5경기", "홈 관중 흐름 사용"),
    ])

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>최근 5경기 홈 관중 흐름</div>", unsafe_allow_html=True)
    display_df = recent.copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df.columns = ["날짜", "원정팀", "관중 수"]
    left, right = st.columns([1.05, 1.25])
    with left:
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    with right:
        chart_df = recent.copy()
        chart_df["date"] = chart_df["date"].dt.strftime("%m-%d")
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=4, color="#1976d2")
            .encode(
                x=alt.X("date:N", title="경기일"),
                y=alt.Y("attendance:Q", title="관중 수"),
                tooltip=["date", "away_team", "attendance"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


st.sidebar.title("모델 선택")
page = st.sidebar.radio("예측 방식", ["Dense 예측", "LSTM 예측", "GRU 예측"])

if page == "Dense 예측":
    render_dense_page()
elif page == "LSTM 예측":
    render_sequence_page("LSTM")
else:
    render_sequence_page("GRU")
