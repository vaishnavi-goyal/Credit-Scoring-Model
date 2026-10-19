
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------- PAGE CONFIG ----------------
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "credit_scoring_model.joblib"
INFO_PATH = BASE_DIR / "model_info.json"

st.set_page_config(
    page_title="CreditGuard | Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LIGHT THEME CSS ----------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #eee9ff 0%, transparent 42%),
        radial-gradient(ellipse at 95% 20%, #e1f0ff 0%, transparent 38%),
        #f7f8fc;
    color: #202944;
}
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}
#MainMenu, footer {
    visibility: hidden;
}
.brand {
    font-size: 25px;
    font-weight: 800;
    color: #202944;
}
.brand span {
    color: #7655d9;
}
.hero {
    padding: 35px 10px 25px;
}
.eyebrow {
    color: #7655d9;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}
.hero h1 {
    font-size: clamp(32px, 5vw, 52px);
    line-height: 1.15;
    letter-spacing: -1.5px;
    color: #202944;
}
.gradient {
    background: linear-gradient(90deg, #7956d9, #3488db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.muted {
    color: #64718b;
    line-height: 1.7;
}
.panel {
    background: rgba(255,255,255,0.92);
    border: 1px solid #e1e6f2;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 6px 22px rgba(49,65,100,0.06);
}
.result {
    background: linear-gradient(140deg, #f0eaff, #e6f3ff);
    border: 1px solid #d9d3fa;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0e6f2;
    border-radius: 13px;
    padding: 15px;
    box-shadow: 0 4px 15px rgba(49,65,100,0.04);
}
div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #35415c;
    font-weight: 600;
}
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background: #ffffff;
    border-color: #d7deec;
    color: #202944;
    border-radius: 9px;
}
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 48px;
    background: linear-gradient(100deg, #7956d9, #438ce0);
    color: white;
    border: none;
    border-radius: 11px;
    font-weight: 700;
}
.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    color: white;
    border: 1px solid #7655d9;
    filter: brightness(1.04);
}
hr {
    border-color: #e0e6f2;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
div[data-testid="stForm"] {
    background: rgba(255,255,255,0.75);
    border: 1px solid #e1e6f2;
    padding: 20px;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_info():
    with open(INFO_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# Dataset features may be X1-X23 or descriptive names.
ALIASES = {
    "LIMIT_BAL": ["LIMIT_BAL", "X1"],
    "SEX": ["SEX", "X2"],
    "EDUCATION": ["EDUCATION", "X3"],
    "MARRIAGE": ["MARRIAGE", "X4"],
    "AGE": ["AGE", "X5"],
    "PAY_0": ["PAY_0", "X6"],
    "PAY_2": ["PAY_2", "X7"],
    "PAY_3": ["PAY_3", "X8"],
    "PAY_4": ["PAY_4", "X9"],
    "PAY_5": ["PAY_5", "X10"],
    "PAY_6": ["PAY_6", "X11"],
    "BILL_AMT1": ["BILL_AMT1", "X12"],
    "BILL_AMT2": ["BILL_AMT2", "X13"],
    "BILL_AMT3": ["BILL_AMT3", "X14"],
    "BILL_AMT4": ["BILL_AMT4", "X15"],
    "BILL_AMT5": ["BILL_AMT5", "X16"],
    "BILL_AMT6": ["BILL_AMT6", "X17"],
    "PAY_AMT1": ["PAY_AMT1", "X18"],
    "PAY_AMT2": ["PAY_AMT2", "X19"],
    "PAY_AMT3": ["PAY_AMT3", "X20"],
    "PAY_AMT4": ["PAY_AMT4", "X21"],
    "PAY_AMT5": ["PAY_AMT5", "X22"],
    "PAY_AMT6": ["PAY_AMT6", "X23"],
}


def build_feature_row(values, feature_names):
    row = {}

    for feature in feature_names:
        normalized = str(feature).strip().upper()
        matched_key = None

        for key, aliases in ALIASES.items():
            if normalized in aliases:
                matched_key = key
                break

        if matched_key is None or matched_key not in values:
            raise ValueError(
                f"Unsupported model feature: {feature}"
            )

        row[feature] = values[matched_key]

    return pd.DataFrame([row], columns=feature_names)


# ---------------- HEADER ----------------
st.markdown(
    '<div class="brand">◈ Credit<span>Guard</span></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ MACHINE LEARNING · CREDIT RISK</div>
    <h1>Understand risk.<br>
    <span class="gradient">Make informed decisions.</span></h1>
    <p class="muted">
        Explore credit-default risk predictions using customer credit limits,
        repayment history and bill/payment information.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Educational prototype only. This model is not a credit bureau score "
    "and must not be used as the sole basis for lending decisions."
)

# ---------------- CHECK MODEL FILES ----------------
if not MODEL_PATH.exists() or not INFO_PATH.exists():
    st.error(
        "Model files not found. Run train_model.py first and verify that "
        "models/ contains credit_scoring_model.joblib and model_info.json."
    )
    st.stop()

try:
    model = load_model()
    model_info = load_model_info()
    feature_names = model_info["features"]
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()


# ---------------- CUSTOMER INPUT FORM ----------------
st.markdown("### 01 · Customer financial profile")
st.markdown(
    '<p class="muted">Enter an example customer profile below.</p>',
    unsafe_allow_html=True
)

with st.form("credit_profile"):
    st.markdown("#### Credit and personal details")

    c1, c2, c3 = st.columns(3)

    with c1:
        limit_bal = st.number_input(
            "Credit limit (NT$)",
            min_value=1000,
            max_value=2000000,
            value=50000,
            step=1000
        )

    with c2:
        sex = st.selectbox(
            "Sex (dataset code)",
            [1, 2],
            format_func=lambda x: {
                1: "1 — Male",
                2: "2 — Female"
            }[x]
        )

    with c3:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    c4, c5 = st.columns(2)

    with c4:
        education = st.selectbox(
            "Education code",
            [1, 2, 3, 4, 5, 6, 0],
            format_func=lambda x: {
                1: "1 — Graduate school",
                2: "2 — University",
                3: "3 — High school",
                4: "4 — Other",
                5: "5 — Unknown category",
                6: "6 — Unknown category",
                0: "0 — Unknown category"
            }[x]
        )

    with c5:
        marriage = st.selectbox(
            "Marital status code",
            [1, 2, 3, 0],
            format_func=lambda x: {
                1: "1 — Married",
                2: "2 — Single",
                3: "3 — Other",
                0: "0 — Unknown category"
            }[x]
        )

    st.markdown("---")
    st.markdown("#### Repayment history")
    st.caption(
        "-1 = paid duly; 0 = revolving credit; positive values "
        "indicate payment delay in months."
    )

    payment_status = {}

    cols = st.columns(3)
    for i, col in zip([0, 1, 2], cols):
        with col:
            key = ["PAY_0", "PAY_2", "PAY_3"][i]
            label = [
                "Most recent month",
                "2 months ago",
                "3 months ago"
            ][i]
            payment_status[key] = st.selectbox(
                f"{label} ({key})",
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                index=2
            )

    cols = st.columns(3)
    for i, col in zip([3, 4, 5], cols):
        with col:
            key = ["PAY_4", "PAY_5", "PAY_6"][i - 3]
            label = [
                "4 months ago",
                "5 months ago",
                "6 months ago"
            ][i - 3]
            payment_status[key] = st.selectbox(
                f"{label} ({key})",
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                index=2
            )

    st.markdown("---")
    st.markdown("#### Recent bills and payments")
    st.caption("Enter amounts in New Taiwan dollars (NT$).")

    bills = {}
    payments = {}

    cols = st.columns(3)
    for i, col in zip(range(1, 4), cols):
        with col:
            bills[f"BILL_AMT{i}"] = st.number_input(
                f"Bill amount {i}",
                min_value=-1000000,
                max_value=10000000,
                value=10000 if i == 1 else 8000,
                step=1000
            )

    cols = st.columns(3)
    for i, col in zip(range(1, 4), cols):
        with col:
            payments[f"PAY_AMT{i}"] = st.number_input(
                f"Payment amount {i}",
                min_value=0,
                max_value=10000000,
                value=5000 if i == 1 else 3000,
                step=500
            )

    cols = st.columns(3)
    for i, col in zip(range(4, 7), cols):
        with col:
            bills[f"BILL_AMT{i}"] = st.number_input(
                f"Bill amount {i}",
                min_value=-1000000,
                max_value=10000000,
                value=7000,
                step=1000
            )

    cols = st.columns(3)
    for i, col in zip(range(4, 7), cols):
        with col:
            payments[f"PAY_AMT{i}"] = st.number_input(
                f"Payment amount {i}",
                min_value=0,
                max_value=10000000,
                value=2500,
                step=500
            )

    submitted = st.form_submit_button(
        "✦  Analyze Credit Risk",
        type="primary"
    )


# ---------------- PREDICTION RESULTS ----------------
if submitted:
    values = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **payment_status,
        **bills,
        **payments
    }

    try:
        input_df = build_feature_row(values, feature_names)

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        default_index = classes.index(1)
        no_default_index = classes.index(0)

        default_probability = float(probabilities[default_index])
        no_default_probability = float(probabilities[no_default_index])

        st.markdown("---")
        st.markdown("### 02 · Prediction results")

        r1, r2 = st.columns([1, 1])

        with r1:
            st.markdown('<div class="result">', unsafe_allow_html=True)

            if prediction == 1:
                st.markdown("#### Higher predicted default risk")
            else:
                st.markdown("#### Lower predicted default risk")

            st.metric(
                "Estimated default probability",
                f"{default_probability * 100:.1f}%"
            )
            st.progress(default_probability)

            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("#### Model interpretation")
            st.write(
                "The model compares this profile with patterns learned "
                "from historical customer data."
            )
            st.write(
                f"Predicted class: **{prediction}** "
                f"({'Default' if prediction == 1 else 'No default'})"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Attractive interactive probability graph
        probability_df = pd.DataFrame({
            "Outcome": ["No default", "Default"],
            "Probability (%)": [
                no_default_probability * 100,
                default_probability * 100
            ]
        })

        fig = go.Figure(
            go.Bar(
                x=probability_df["Outcome"],
                y=probability_df["Probability (%)"],
                marker=dict(
                    color=["#4F8FE8", "#8B6CE8"],
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                text=[
                    f"{value:.1f}%"
                    for value in probability_df["Probability (%)"]
                ],
                textposition="outside",
                textfont=dict(size=14, color="#35415C"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Probability: %{y:.2f}%<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title=dict(
                text="Credit Default Probability",
                font=dict(size=20, color="#202944")
            ),
            xaxis_title="Customer outcome",
            yaxis_title="Probability (%)",
            yaxis=dict(
                range=[0, 110],
                ticksuffix="%",
                gridcolor="#E6EAF3",
                zeroline=False
            ),
            xaxis=dict(
                tickfont=dict(color="#35415C", size=13)
            ),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(family="Arial", color="#35415C"),
            margin=dict(l=15, r=15, t=65, b=25),
            height=380,
            showlegend=False
        )

        st.markdown("### 03 · Prediction analytics")
        st.markdown(
            '<p class="muted">Compare the estimated probabilities '
            'for both possible outcomes.</p>',
            unsafe_allow_html=True
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "This is an educational estimate, not a guarantee of future "
            "repayment behavior. Model confidence does not establish "
            "creditworthiness or replace fair, regulated lending review."
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")

st.markdown("---")
st.markdown(
    '<p class="muted">CreditGuard · Educational ML project · '
    'Built with Python, scikit-learn, Plotly and Streamlit</p>',
    unsafe_allow_html=True
)
