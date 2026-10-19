
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="CreditGuard | Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =====================================================
# 2. LIGHT THEME AND CSS
# =====================================================

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
    font-size: 26px;
    font-weight: 800;
    color: #202944;
}

.brand span {
    color: #7655d9;
}

.hero {
    padding: 32px 5px 24px;
}

.eyebrow {
    color: #7655d9;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    font-size: clamp(32px, 5vw, 50px);
    line-height: 1.18;
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

div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dce3ef !important;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(49, 65, 100, 0.05);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p,
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] p,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #35415c !important;
    font-weight: 600;
}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #111827 !important;
    border-color: #d7deec !important;
    border-radius: 9px;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.85);
    border: 1px solid #e1e6f2;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 6px 22px rgba(49,65,100,0.04);
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    min-height: 48px;
    background: linear-gradient(100deg, #7956d9, #438ce0);
    color: #ffffff !important;
    border: none;
    border-radius: 11px;
    font-weight: 700;
}

hr {
    border-color: #e0e6f2;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. FIND MODEL FILES
# Supports both root and models/ directory
# =====================================================

model_candidates = [
    BASE_DIR / "credit_scoring_model.joblib",
    BASE_DIR / "models" / "credit_scoring_model.joblib"
]

info_candidates = [
    BASE_DIR / "model_info.json",
    BASE_DIR / "models" / "model_info.json"
]

MODEL_PATH = next(
    (path for path in model_candidates if path.is_file()),
    None
)

INFO_PATH = next(
    (path for path in info_candidates if path.is_file()),
    None
)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_model_info(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# =====================================================
# 4. FEATURE NAME MAPPING
# =====================================================

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
    "PAY_AMT6": ["PAY_AMT6", "X23"]
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
                f"Unsupported model feature: {feature}. "
                "Check model_info.json and train_model.py."
            )

        row[feature] = values[matched_key]

    return pd.DataFrame([row], columns=feature_names)


# =====================================================
# 5. HEADER
# =====================================================

st.markdown(
    '<div class="brand">◈ Credit<span>Guard</span></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <div class="eyebrow">MACHINE LEARNING · CREDIT RISK</div>
    <h1>Understand risk.<br>
    <span class="gradient">Make informed decisions.</span></h1>
    <p class="muted">
        Explore credit-default risk predictions using customer
        credit limits, repayment history and bill/payment information.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Educational prototype only. This model is not a credit bureau score "
    "and must not be used as the sole basis for lending decisions."
)


# =====================================================
# 6. CHECK AND LOAD MODEL
# =====================================================

if MODEL_PATH is None or INFO_PATH is None:
    st.error(
        "Model files not found. Ensure these files are available: "
        "credit_scoring_model.joblib and model_info.json. "
        "They can be in the project root or inside the models folder."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    model_info = load_model_info(str(INFO_PATH))
    feature_names = model_info["features"]

    if not hasattr(model, "predict_proba"):
        st.error("The saved model does not support predict_proba.")
        st.stop()

    if 0 not in model.classes_ or 1 not in model.classes_:
        st.error("The model must contain both target classes: 0 and 1.")
        st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# =====================================================
# 7. CUSTOMER INPUT FORM
# =====================================================

st.markdown("### 01 · Customer Financial Profile")

st.markdown(
    '<p class="muted">Enter example customer details below.</p>',
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
        "Dataset codes: -1 generally indicates paid duly, 0 indicates "
        "revolving credit, and positive values indicate payment delays."
    )

    payment_status = {}

    first_payment_cols = st.columns(3)

    payment_fields = [
        ("PAY_0", "Most recent month"),
        ("PAY_2", "2 months ago"),
        ("PAY_3", "3 months ago"),
        ("PAY_4", "4 months ago"),
        ("PAY_5", "5 months ago"),
        ("PAY_6", "6 months ago")
    ]

    for index, (key, label) in enumerate(payment_fields):
        if index < 3:
            column = first_payment_cols[index]
        else:
            if index == 3:
                second_payment_cols = st.columns(3)
            column = second_payment_cols[index - 3]

        with column:
            payment_status[key] = st.selectbox(
                f"{label} ({key})",
                [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                index=2,
                key=f"input_{key}"
            )

    st.markdown("---")
    st.markdown("#### Recent bills and payments")
    st.caption("Enter amounts in New Taiwan dollars (NT$).")

    bills = {}
    payments = {}

    bill_cols_1 = st.columns(3)

    for i, column in zip(range(1, 4), bill_cols_1):
        with column:
            bills[f"BILL_AMT{i}"] = st.number_input(
                f"Bill amount {i}",
                min_value=-1000000,
                max_value=10000000,
                value=10000 if i == 1 else 8000,
                step=1000,
                key=f"bill_{i}"
            )

    bill_cols_2 = st.columns(3)

    for i, column in zip(range(4, 7), bill_cols_2):
        with column:
            bills[f"BILL_AMT{i}"] = st.number_input(
                f"Bill amount {i}",
                min_value=-1000000,
                max_value=10000000,
                value=7000,
                step=1000,
                key=f"bill_{i}"
            )

    payment_cols_1 = st.columns(3)

    for i, column in zip(range(1, 4), payment_cols_1):
        with column:
            payments[f"PAY_AMT{i}"] = st.number_input(
                f"Payment amount {i}",
                min_value=0,
                max_value=10000000,
                value=5000 if i == 1 else 3000,
                step=500,
                key=f"payment_{i}"
            )

    payment_cols_2 = st.columns(3)

    for i, column in zip(range(4, 7), payment_cols_2):
        with column:
            payments[f"PAY_AMT{i}"] = st.number_input(
                f"Payment amount {i}",
                min_value=0,
                max_value=10000000,
                value=2500,
                step=500,
                key=f"payment_{i}"
            )

    submitted = st.form_submit_button(
        "✦ Analyze Credit Risk",
        type="primary"
    )


# =====================================================
# 8. PREDICTION AND INTERACTIVE GRAPH
# =====================================================

if submitted:
    try:
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

        input_df = build_feature_row(values, feature_names)

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        default_index = classes.index(1)
        no_default_index = classes.index(0)

        default_probability = float(probabilities[default_index])
        no_default_probability = float(probabilities[no_default_index])

        st.markdown("---")
        st.markdown("### 02 · Prediction Results")

        result_col, details_col = st.columns(2)

        with result_col:
            if prediction == 1:
                st.error("Higher predicted default risk")
            else:
                st.success("Lower predicted default risk")

            st.metric(
                "Estimated default probability",
                f"{default_probability * 100:.1f}%"
            )

            st.progress(default_probability)

        with details_col:
            st.markdown("#### Model interpretation")

            st.write(
                "The model compares this profile with patterns learned "
                "from historical customer data."
            )

            st.metric(
                "Predicted class",
                "Default" if prediction == 1 else "No Default"
            )

            st.caption(
                "This is a model prediction, not a guarantee of future behavior."
            )

        # -------------------------------------------------
        # INTERACTIVE PROBABILITY GRAPH
        # -------------------------------------------------

        st.markdown("---")
        st.markdown("### 03 · Prediction Analytics")

        st.markdown(
            """
            <p style="color:#64718b; font-size:15px;">
                Explore both possible outcomes. Hover over each bar
                to view the estimated probability.
            </p>
            """,
            unsafe_allow_html=True
        )

        probability_df = pd.DataFrame({
            "Outcome": ["No Default", "Default"],
            "Probability (%)": [
                no_default_probability * 100,
                default_probability * 100
            ]
        })

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=probability_df["Outcome"],
                y=probability_df["Probability (%)"],
                text=[
                    f"{value:.1f}%"
                    for value in probability_df["Probability (%)"]
                ],
                textposition="outside",
                textfont=dict(
                    family="Arial",
                    size=16,
                    color="#111827"
                ),
                marker=dict(
                    color=["#4F8FE8", "#8B6CE8"],
                    line=dict(color="#FFFFFF", width=2)
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Estimated probability: %{y:.2f}%"
                    "<extra></extra>"
                ),
                cliponaxis=False
            )
        )

        fig.update_layout(
            title=dict(
                text="Credit Default Probability",
                x=0.02,
                xanchor="left",
                font=dict(size=22, color="#111827")
            ),
            xaxis=dict(
                title=dict(
                    text="Predicted outcome",
                    font=dict(size=14, color="#111827")
                ),
                tickfont=dict(size=14, color="#111827"),
                showgrid=False,
                zeroline=False,
                fixedrange=True
            ),
            yaxis=dict(
                title=dict(
                    text="Probability (%)",
                    font=dict(size=14, color="#111827")
                ),
                tickfont=dict(size=12, color="#111827"),
                range=[0, 112],
                dtick=20,
                ticksuffix="%",
                gridcolor="#E1E7F0",
                zeroline=False,
                fixedrange=True
            ),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font=dict(
                family="Arial",
                color="#111827",
                size=13
            ),
            margin=dict(l=30, r=25, t=85, b=45),
            height=430,
            showlegend=False,
            hoverlabel=dict(
                bgcolor="#FFFFFF",
                bordercolor="#D9DFF0",
                font=dict(size=14, color="#111827")
            ),
            bargap=0.48
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "responsive": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "credit_probability_chart",
                    "height": 600,
                    "width": 1000,
                    "scale": 2
                }
            }
        )

        st.caption(
            "Probabilities are model estimates, not guarantees of future "
            "repayment behavior. Model estimates do not establish "
            "creditworthiness or replace a fair lending assessment."
        )

    except Exception as error:
        st.error(
            f"Prediction could not be completed: {error}"
        )


# =====================================================
# 9. FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <p style="color:#64718b; text-align:center;">
        CreditGuard · Educational Machine Learning Project<br>
        Built with Python, Scikit-learn, Plotly and Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
