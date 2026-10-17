from __future__ import annotations

import json
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import DATASET_RAW_PATH, EDUCATIONAL_DISCLAIMER, FEATURE_NAMES, METRICS_PATH, MODEL_PATH, PLOTS_DIR, TARGET_COLUMN
from src.data_loader import load_credit_data
from src.evaluation import evaluate_at_threshold
from src.prediction import batch_predict, predict_single
from src.preprocessing import split_train_test
from src.visualization import get_feature_importance_data

st.set_page_config(page_title="CredScore | Risk workspace", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --ink: #eef4fb;
        --muted: #9aacc0;
        --canvas: #101713;
        --paper: #19251e;
        --line: #34443a;
        --navy: #0c130f;
        --navy-soft: #1c2b22;
        --accent: #59c68a;
        --mint: #35d5a0;
        --coral: #ff7b72;
        --violet: #c3a0ff;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', 'Segoe UI', sans-serif; color: var(--ink); }
    .stApp, [data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 82% 4%, #20392b 0, var(--canvas) 42rem); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1480px; padding: 1.8rem 2.3rem 3rem; }
    [data-testid="stSidebar"] { background: var(--navy); border-right: 1px solid #27372e; width: 15rem !important; min-width: 15rem !important; }
    [data-testid="stSidebar"] > div:first-child { background: var(--navy); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #cbd7e5; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #cbd7e5; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border: 1px solid transparent; border-radius: 9px; padding: .5rem .7rem; margin: .15rem 0;
        color: #d2deea; transition: background .16s ease, border-color .16s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: var(--navy-soft); }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: #234c37; border-color: #3f7656; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { color: inherit; font-weight: 600; }
    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3 { color: var(--ink); font-family: 'Manrope', 'Segoe UI', sans-serif; letter-spacing: 0; }
    [data-testid="stMainBlockContainer"] h1 { font-size: 1.85rem; line-height: 1.2; margin-bottom: .2rem; }
    [data-testid="stMainBlockContainer"] h2 { font-size: 1.18rem; }
    [data-testid="stMainBlockContainer"] h3 { font-size: 1rem; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #1b2b22, #17241c); border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1rem; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Manrope', 'Segoe UI', sans-serif; }
    [data-testid="stMetricDelta"] { color: var(--mint); }
    [data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(145deg, #19271f, #16231b); border-color: var(--line); border-radius: 13px; }
    [data-testid="stDataFrame"] { background: #19271f; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
    [data-testid="stImage"] img { max-width: 100%; height: auto; object-fit: contain; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: var(--ink); }
    [data-testid="stAlert"] { background: #1a2b20; border: 1px solid var(--line); color: var(--ink); }
    div.stButton > button, div.stFormSubmitButton > button, [data-testid="stDownloadButton"] button {
        border-radius: 9px; font-weight: 600; min-height: 2.65rem; border-color: #3c5747; background: #1b2a21; color: var(--ink);
    }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(110deg, #28734e, #359465); border-color: #59c68a; color: #fff;
    }
    [data-testid="stProgressBar"] > div > div { background: var(--mint); }
    .brand-lockup { display: flex; align-items: center; gap: .7rem; padding: .4rem .25rem 1.2rem; border-bottom: 1px solid #2c4032; margin-bottom: 1.1rem; }
    .brand-mark { display: grid; place-items: center; width: 2.35rem; height: 2.35rem; border-radius: 9px; background: linear-gradient(145deg, #25d3a7, #59c68a); color: #101713; font: 800 .8rem 'Manrope', sans-serif; }
    .brand-name { color: #fff; font: 700 1rem 'Manrope', sans-serif; }
    .brand-caption { color: #a4b6a9; font-size: .68rem; margin-top: .12rem; }
    .sidebar-kicker { color: #90a395; font-size: .67rem; margin: .8rem .25rem .3rem; }
    .sidebar-foot { border-top: 1px solid #2c4032; margin-top: 1.5rem; padding: .9rem .25rem; color: #9aab9f; font-size: .75rem; line-height: 1.5; }
    .page-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; margin: 0 0 1.25rem; }
    .page-eyebrow { color: var(--mint); font-size: .7rem; font-weight: 700; margin: 0 0 .35rem; }
    .page-title { color: var(--ink); font: 800 1.85rem/1.2 'Manrope', 'Segoe UI', sans-serif; margin: 0; }
    .page-subtitle { color: var(--muted); font-size: .91rem; margin: .45rem 0 0; max-width: 760px; }
    .status-pill { white-space: nowrap; border: 1px solid #1e695d; background: #10352f; color: #74e0c4; border-radius: 999px; padding: .42rem .7rem; font-size: .73rem; font-weight: 700; margin-top: .2rem; }
    .status-dot { display: inline-block; width: .43rem; height: .43rem; background: var(--mint); border-radius: 50%; margin-right: .35rem; box-shadow: 0 0 10px #28d3ac88; }
    .stat-card { height: 100%; min-height: 112px; box-sizing: border-box; background: linear-gradient(145deg, #1b2b22, #17241c); border: 1px solid var(--line); border-top: 3px solid var(--accent); border-radius: 11px; padding: .9rem 1rem; }
    .stat-label { color: var(--muted); font-size: .72rem; font-weight: 700; margin-bottom: .55rem; }
    .stat-value { color: var(--ink); font: 800 1.45rem 'Manrope', sans-serif; }
    .stat-note { color: var(--muted); font-size: .74rem; margin-top: .35rem; }
    .section-kicker { color: var(--muted); font-size: .7rem; font-weight: 700; margin: 0 0 .35rem; }
    .result-card { border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 11px; background: linear-gradient(145deg, #1b2b22, #17241c); padding: 1.1rem 1.2rem; }
    .result-label { color: var(--muted); font-size: .75rem; font-weight: 700; }
    .result-value { color: var(--accent); font: 800 1.5rem 'Manrope', sans-serif; margin: .2rem 0; }
    .result-copy { color: var(--muted); font-size: .83rem; line-height: 1.5; }
    .quiet-note { color: var(--muted); font-size: .77rem; }
    .comparison-wrap { width: 100%; overflow-x: auto; }
    .comparison-grid { display: grid; grid-template-columns: minmax(5.5rem, .8fr) repeat(3, minmax(4.2rem, 1fr)); align-items: center; gap: .65rem .55rem; min-width: 0; }
    .comparison-heading { color: var(--muted); font-size: .68rem; font-weight: 700; line-height: 1.25; }
    .comparison-label { color: var(--ink); font-size: .75rem; font-weight: 700; }
    .comparison-cell { min-width: 0; }
    .comparison-value { color: var(--ink); font-size: .68rem; font-weight: 700; margin-bottom: .25rem; }
    .comparison-track, .risk-track { width: 100%; height: .34rem; border-radius: 99px; background: #314236; overflow: hidden; }
    .comparison-fill, .risk-fill { display: block; height: 100%; border-radius: inherit; }
    .risk-row { margin: .85rem 0; }
    .risk-row-head { display: flex; justify-content: space-between; gap: .75rem; color: var(--ink); font-size: .77rem; margin-bottom: .35rem; }
    .risk-row-head span:last-child { color: var(--muted); }
    .outcome-donut-layout { display: flex; align-items: center; justify-content: center; gap: 1.4rem; min-height: 250px; }
    .outcome-donut { display: grid; place-items: center; width: 176px; aspect-ratio: 1; flex: 0 0 auto; border-radius: 50%; background: conic-gradient(var(--mint) 0deg var(--good-angle), var(--coral) var(--good-angle) 360deg); }
    .outcome-donut-hole { display: flex; flex-direction: column; align-items: center; justify-content: center; width: 64%; aspect-ratio: 1; border-radius: 50%; background: #19251e; }
    .outcome-donut-value { color: var(--ink); font: 800 1.4rem 'Manrope', sans-serif; }
    .outcome-donut-caption { color: var(--muted); font-size: .7rem; }
    .outcome-legend { display: grid; gap: .95rem; }
    .outcome-legend-item { display: grid; grid-template-columns: .6rem auto; column-gap: .55rem; align-items: start; color: var(--ink); font-size: .77rem; }
    .outcome-swatch { width: .55rem; height: .55rem; margin-top: .18rem; border-radius: 50%; }
    .outcome-legend-count { grid-column: 2; color: var(--muted); font-size: .72rem; margin-top: .16rem; }
    @media (max-width: 760px) {
        [data-testid="stSidebar"] { width: min(16rem, 82vw) !important; min-width: min(16rem, 82vw) !important; }
        [data-testid="stMainBlockContainer"] { padding: 1rem .85rem 2rem; }
        .page-heading { display: block; }
        .status-pill { display: inline-block; margin-top: .8rem; }
        .page-title { font-size: 1.55rem; }
        .stat-card { min-height: 98px; padding: .75rem; }
        .stat-value { font-size: 1.2rem; }
        .comparison-grid { grid-template-columns: minmax(4rem, .8fr) repeat(3, minmax(3.4rem, 1fr)); gap: .6rem .35rem; }
        .comparison-heading { font-size: .61rem; }
        .comparison-label, .comparison-value { font-size: .65rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="brand-lockup"><div class="brand-mark">CS</div><div><div class="brand-name">CredScore</div><div class="brand-caption">RISK ANALYTICS WORKSPACE</div></div></div><div class="sidebar-kicker">WORKSPACE</div>',
    unsafe_allow_html=True,
)

FEATURE_DESCRIPTIONS = {
    "status_of_existing_checking_account": "Status of the existing checking account.",
    "duration_in_month": "Duration of the credit in months.",
    "credit_history": "Credit history of the applicant.",
    "purpose": "Purpose of the credit.",
    "credit_amount": "Credit amount requested.",
    "savings_account_bonds": "Savings account or bonds balance.",
    "present_employment_since": "Current employment duration.",
    "installment_rate_in_percentage_of_disposable_income": "Installment rate as a percentage of disposable income.",
    "personal_status_and_sex": "Personal status and sex.",
    "other_debtors_guarantors": "Other debtors or guarantors.",
    "residence_since": "Years at current residence.",
    "property": "Property ownership status.",
    "age_in_years": "Applicant age in years.",
    "other_installment_plans": "Other installment plans.",
    "housing": "Housing situation.",
    "number_of_existing_credits_at_this_bank": "Number of existing credits at the bank.",
    "job": "Job type/occupation.",
    "number_of_people_liable_to_provide_maintenance_for": "Number of dependents or people liable for maintenance.",
    "telephone": "Telephone availability.",
    "foreign_worker": "Whether the applicant is a foreign worker.",
}

OPTIONAL_CATEGORIES = {
    "status_of_existing_checking_account": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
    "savings_account_bonds": ["A61", "A62", "A63", "A64", "A65"],
    "present_employment_since": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status_and_sex": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors_guarantors": ["A101", "A102", "A103", "A104"],
    "property": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143", "A144"],
    "housing": ["A151", "A152", "A153", "A154"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

FEATURE_LABELS = {
    "status_of_existing_checking_account": "Checking account status",
    "duration_in_month": "Credit duration (months)",
    "credit_history": "Credit history",
    "purpose": "Credit purpose",
    "credit_amount": "Credit amount",
    "savings_account_bonds": "Savings and bonds",
    "present_employment_since": "Employment duration",
    "installment_rate_in_percentage_of_disposable_income": "Installment rate (1-4)",
    "personal_status_and_sex": "Personal status",
    "other_debtors_guarantors": "Co-applicant or guarantor",
    "residence_since": "Years at current residence",
    "property": "Property type",
    "age_in_years": "Age (years)",
    "other_installment_plans": "Other installment plans",
    "housing": "Housing status",
    "number_of_existing_credits_at_this_bank": "Existing credits at this bank",
    "job": "Employment category",
    "number_of_people_liable_to_provide_maintenance_for": "Dependents",
    "telephone": "Telephone status",
    "foreign_worker": "Foreign-worker status",
}

NUMERIC_INPUTS = {
    "duration_in_month": (4, 72, 24),
    "credit_amount": (250, 18424, 2500),
    "installment_rate_in_percentage_of_disposable_income": (1, 4, 2),
    "residence_since": (1, 4, 2),
    "age_in_years": (19, 75, 35),
    "number_of_existing_credits_at_this_bank": (1, 4, 1),
    "number_of_people_liable_to_provide_maintenance_for": (1, 2, 1),
}


def load_saved_metrics():
    if not METRICS_PATH.exists():
        return {}
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_credit_dataset():
    return load_credit_data(DATASET_RAW_PATH)


@st.cache_data(show_spinner=False)
def get_heldout_data():
    dataset = get_credit_dataset()
    _, X_test, _, y_test = split_train_test(dataset, target_column=TARGET_COLUMN, test_size=0.2, random_state=42)
    return X_test, y_test


def page_header(title, description, status="Model workspace"):
    st.markdown(
        f'<div class="page-heading"><div><p class="page-eyebrow">CREDScore / RISK OPERATIONS</p><h1 class="page-title">{escape(title)}</h1><p class="page-subtitle">{escape(description)}</p></div><span class="status-pill"><span class="status-dot"></span>{escape(status)}</span></div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, note, tone="blue"):
    tones = {"blue": "var(--accent)", "good": "var(--mint)", "bad": "var(--coral)", "violet": "var(--violet)"}
    accent = tones.get(tone, tones["blue"])
    st.markdown(
        f'<div class="stat-card" style="--accent:{accent}"><div class="stat-label">{escape(str(label).upper())}</div><div class="stat-value">{escape(str(value))}</div><div class="stat-note">{escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def render_model_comparison(test_results):
    metric_specs = [
        ("Accuracy", "accuracy", "#2563c8"),
        ("Precision", "precision", "#168f9a"),
        ("Recall", "recall", "#d75c66"),
        ("F1 score", "f1", "#7856c9"),
        ("ROC-AUC", "roc_auc", "#16846e"),
    ]
    model_names = list(test_results)
    model_labels = {"logistic_regression": "Logistic regression", "decision_tree": "Decision tree", "random_forest": "Random forest"}
    markup = ['<div class="comparison-wrap"><div class="comparison-grid"><div></div>']
    for name in model_names:
        markup.append(f'<div class="comparison-heading">{escape(model_labels.get(name, name.replace("_", " ").title()))}</div>')
    for label, key, color in metric_specs:
        markup.append(f'<div class="comparison-label">{label}</div>')
        for name in model_names:
            value = test_results[name].get(key)
            value = float(value) if value is not None else 0.0
            width = min(100, max(0, value * 100))
            markup.append(
                f'<div class="comparison-cell"><div class="comparison-value">{value:.0%}</div>'
                f'<div class="comparison-track"><span class="comparison-fill" style="width:{width:.1f}%;background:{color}"></span></div></div>'
            )
    markup.append("</div></div>")
    st.markdown("".join(markup), unsafe_allow_html=True)


def render_feature_importance(importance_rows, top_n):
    top_features = sorted(importance_rows, key=lambda row: float(row["importance"]), reverse=True)[:top_n]
    if not top_features:
        st.info("Feature-importance data is not available for this model.")
        return

    maximum = max(float(row["importance"]) for row in top_features) or 1.0
    markup = ['<div class="importance-list">']
    for row in top_features:
        feature = str(row["feature"])
        label = feature.replace("numeric__", "").replace("categorical__", "").replace("_", " ")
        value = float(row["importance"])
        width = value / maximum * 100
        markup.append(
            f'<div class="importance-row"><div class="importance-label" title="{escape(feature)}">{escape(label)}</div>'
            f'<div class="importance-track"><span class="importance-fill" style="width:{width:.1f}%"></span></div>'
            f'<div class="importance-value">{value:.3f}</div></div>'
        )
    markup.append("</div>")
    st.markdown("".join(markup), unsafe_allow_html=True)


def home_dashboard():
    metrics = load_saved_metrics()
    dataset = get_credit_dataset() if DATASET_RAW_PATH.exists() else None
    selected_name = metrics.get("selected_model", "not trained")
    test_results = metrics.get("test_results", {})
    selected_metrics = test_results.get(selected_name, {})
    good_count = int((dataset[TARGET_COLUMN] == 0).sum()) if dataset is not None else 0
    bad_count = int((dataset[TARGET_COLUMN] == 1).sum()) if dataset is not None else 0
    total_count = len(dataset) if dataset is not None else 0
    good_share = good_count / total_count if total_count else 0
    bad_share = bad_count / total_count if total_count else 0

    page_header(
        "Credit risk overview",
        "A transparent view of the UCI German Credit portfolio and the current model's held-out performance.",
        "Pipeline ready" if MODEL_PATH.exists() else "Training required",
    )

    cards = st.columns(4)
    with cards[0]:
        metric_card("Total records", f"{total_count:,}" if total_count else "Unavailable", "UCI German Credit dataset", "blue")
    with cards[1]:
        metric_card("Good credit", f"{good_count:,}" if total_count else "--", f"{good_share:.1%} of records" if total_count else "Target class 0", "good")
    with cards[2]:
        metric_card("Bad credit", f"{bad_count:,}" if total_count else "--", f"{bad_share:.1%} of records" if total_count else "Target class 1", "bad")
    with cards[3]:
        accuracy = selected_metrics.get("accuracy")
        metric_card("Test accuracy", f"{accuracy:.1%}" if accuracy is not None else "--", selected_name.replace("_", " ").title(), "violet")

    with st.container(border=True):
        st.markdown("### Explore a live portfolio segment")
        filter_columns = st.columns(4)
        with filter_columns[0]:
            age_range = st.slider(
                "Applicant age",
                int(dataset["age_in_years"].min()),
                int(dataset["age_in_years"].max()),
                (int(dataset["age_in_years"].min()), int(dataset["age_in_years"].max())),
                key="overview_age_range",
            )
        with filter_columns[1]:
            amount_range = st.slider(
                "Credit amount",
                int(dataset["credit_amount"].min()),
                int(dataset["credit_amount"].max()),
                (int(dataset["credit_amount"].min()), int(dataset["credit_amount"].max())),
                step=250,
                key="overview_amount_range",
            )
        with filter_columns[2]:
            purposes = sorted(dataset["purpose"].dropna().unique().tolist())
            selected_purpose = st.selectbox("Credit purpose", ["All purposes", *purposes], key="overview_purpose")
        with filter_columns[3]:
            selected_outcome = st.selectbox("Observed outcome", ["All outcomes", "Good credit", "Bad credit"], key="overview_outcome")

    filtered_data = dataset[
        dataset["age_in_years"].between(*age_range)
        & dataset["credit_amount"].between(*amount_range)
    ].copy()
    if selected_purpose != "All purposes":
        filtered_data = filtered_data[filtered_data["purpose"] == selected_purpose]
    if selected_outcome == "Good credit":
        filtered_data = filtered_data[filtered_data[TARGET_COLUMN] == 0]
    elif selected_outcome == "Bad credit":
        filtered_data = filtered_data[filtered_data[TARGET_COLUMN] == 1]

    segment_columns = st.columns(3)
    segment_bad_rate = float(filtered_data[TARGET_COLUMN].mean()) if len(filtered_data) else 0.0
    segment_average_amount = float(filtered_data["credit_amount"].mean()) if len(filtered_data) else 0.0
    with segment_columns[0]:
        st.metric("Filtered records", f"{len(filtered_data):,}", f"from {len(dataset):,} total records")
    with segment_columns[1]:
        st.metric("Observed bad-credit share", f"{segment_bad_rate:.1%}" if len(filtered_data) else "--", "Historical labels in this segment")
    with segment_columns[2]:
        st.metric("Mean requested amount", f"DM {segment_average_amount:,.0f}" if len(filtered_data) else "--", "Within the current filters")

    st.write("")
    left, right = st.columns([1.55, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("### Candidate model comparison")
        st.caption("Held-out test-set scores. Higher is better; the positive class is bad credit.")
        if test_results:
            render_model_comparison(test_results)
        else:
            st.info("Train the candidate models to see the comparison.")

    with right, st.container(border=True):
        st.markdown("### Segment outcome mix")
        st.caption("Updates immediately from the age, amount, purpose, and outcome filters above.")
        if len(filtered_data):
            outcome_counts = filtered_data[TARGET_COLUMN].map({0: "Good credit", 1: "Bad credit"}).value_counts().reindex(["Good credit", "Bad credit"], fill_value=0)
            segment_total = int(outcome_counts.sum())
            good_segment_count = int(outcome_counts["Good credit"])
            bad_segment_count = int(outcome_counts["Bad credit"])
            good_segment_share = good_segment_count / segment_total if segment_total else 0.0
            st.markdown(
                f'<div class="outcome-donut-layout" role="img" aria-label="Good credit: {good_segment_count} of {segment_total}; bad credit: {bad_segment_count} of {segment_total}">'
                f'<div class="outcome-donut" style="--good-angle:{good_segment_share * 360:.2f}deg"><div class="outcome-donut-hole"><span class="outcome-donut-value">{good_segment_share:.0%}</span><span class="outcome-donut-caption">Good credit</span></div></div>'
                f'<div class="outcome-legend"><div class="outcome-legend-item"><span class="outcome-swatch" style="background:var(--mint)"></span><span>Good credit</span><span class="outcome-legend-count">{good_segment_count:,} · {good_segment_share:.1%}</span></div>'
                f'<div class="outcome-legend-item"><span class="outcome-swatch" style="background:var(--coral)"></span><span>Bad credit</span><span class="outcome-legend-count">{bad_segment_count:,} · {1 - good_segment_share:.1%}</span></div></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No records match these filters. Widen the age or credit-amount range.")

    st.markdown("### Live segment patterns")
    st.caption("These plots are recalculated from the filtered source rows, not static training images.")
    trend_left, trend_right = st.columns(2, gap="large")
    with trend_left, st.container(border=True):
        st.markdown("#### Bad-credit rate by age band")
        if len(filtered_data):
            age_bands = pd.cut(
                filtered_data["age_in_years"],
                bins=[18, 25, 35, 45, 55, 65, 100],
                labels=["19-25", "26-35", "36-45", "46-55", "56-65", "66+"],
                include_lowest=True,
            )
            bad_rate_by_age = (
                filtered_data.assign(age_band=age_bands)
                .groupby("age_band", observed=True)[TARGET_COLUMN]
                .mean()
                .mul(100)
                .rename("Bad-credit rate (%)")
                .to_frame()
            )
            st.bar_chart(bad_rate_by_age, height=290)
        else:
            st.info("No age-band data for this selection.")
    with trend_right, st.container(border=True):
        st.markdown("#### Credit amount vs duration")
        if len(filtered_data):
            scatter_data = filtered_data[["duration_in_month", "credit_amount", "age_in_years", TARGET_COLUMN]].copy()
            scatter_data["Observed outcome"] = scatter_data[TARGET_COLUMN].map({0: "Good credit", 1: "Bad credit"})
            st.scatter_chart(
                scatter_data.sample(min(500, len(scatter_data)), random_state=42),
                x="duration_in_month",
                y="credit_amount",
                color="Observed outcome",
                size="age_in_years",
                height=290,
            )
        else:
            st.info("No records to plot for this selection.")

    cv_results = metrics.get("cv_results", {})
    lower_left, lower_right = st.columns([1.2, 1], gap="large")
    with lower_left, st.container(border=True):
        st.markdown("### Cross-validation")
        cv_table = pd.DataFrame.from_dict(cv_results, orient="index")
        if not cv_table.empty:
            cv_table = cv_table.rename(columns={"roc_auc_mean": "Mean ROC-AUC", "roc_auc_std": "Std. dev."})
            cv_table.index = [name.replace("_", " ").title() for name in cv_table.index]
            visible_columns = [column for column in ("Mean ROC-AUC", "Std. dev.") if column in cv_table]
            st.dataframe(cv_table[visible_columns], use_container_width=True)
        else:
            st.info("Cross-validation results are not available yet.")

    with lower_right, st.container(border=True):
        st.markdown("### Selected model")
        st.markdown(f"**{selected_name.replace('_', ' ').title()}**")
        cv_score = cv_results.get(selected_name, {}).get("roc_auc_mean")
        st.metric("Cross-validation ROC-AUC", f"{cv_score:.3f}" if cv_score is not None else "--")
        recall = selected_metrics.get("recall")
        if recall is not None:
            st.caption(f"Bad-credit recall on the held-out set: {recall:.1%}. Review this alongside false positives and the intended use.")

    st.caption(EDUCATIONAL_DISCLAIMER)


def render_prediction_input(field, row):
    label = FEATURE_LABELS[field]
    help_text = FEATURE_DESCRIPTIONS.get(field)
    if field in OPTIONAL_CATEGORIES:
        row[field] = st.selectbox(label, OPTIONAL_CATEGORIES[field], help=help_text, key=f"prediction_{field}")
        return

    minimum, maximum, default = NUMERIC_INPUTS[field]
    row[field] = st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=1,
        help=help_text,
        key=f"prediction_{field}",
    )


def single_prediction_page():
    page_header(
        "Applicant risk estimate",
        "Enter the dataset's original applicant attributes to obtain an experimental good/bad classification.",
        "Model ready" if MODEL_PATH.exists() else "Training required",
    )
    st.caption("Complete the applicant details, then choose Predict Result to run the model.")

    with st.form("prediction_form"):
        row = {}
        groups = [
            ("Credit and repayment", [
                "status_of_existing_checking_account", "duration_in_month", "credit_history", "purpose",
                "credit_amount", "savings_account_bonds", "present_employment_since",
                "installment_rate_in_percentage_of_disposable_income",
            ]),
            ("Applicant profile", [
                "personal_status_and_sex", "other_debtors_guarantors", "residence_since", "property",
                "age_in_years", "other_installment_plans", "housing",
                "number_of_existing_credits_at_this_bank", "job",
                "number_of_people_liable_to_provide_maintenance_for", "telephone", "foreign_worker",
            ]),
        ]
        for group_title, fields in groups:
            with st.container(border=True):
                st.markdown(f"### {group_title}")
                for index in range(0, len(fields), 2):
                    columns = st.columns(2)
                    for column, field in zip(columns, fields[index:index + 2]):
                        with column:
                            render_prediction_input(field, row)

        numeric_feature = st.selectbox(
            "Feature to vary in sensitivity chart",
            list(NUMERIC_INPUTS),
            format_func=lambda feature: FEATURE_LABELS[feature],
            key="sensitivity_feature",
        )
        submit = st.form_submit_button("Predict Result", type="primary", use_container_width=True)

    if not submit:
        st.info("Your prediction will appear here after you click Predict Result.")
        return

    model = load_model()
    if model is None:
        st.warning("The trained model artifact is unavailable. Run `python train_model.py` from the project root first.")
        return
    try:
        result = predict_single(row, model=model)
    except (TypeError, ValueError) as exc:
        st.error(f"Please review the applicant details: {exc}")
        return

    probability = result["predicted_probability"]
    bad_prediction = result["predicted_class"] == 1
    accent = "var(--coral)" if bad_prediction else "var(--mint)"
    st.markdown(
        f'<div class="result-card" style="--accent:{accent}"><div class="result-label">MODEL CLASSIFICATION</div><div class="result-value">{escape(result["risk_label"])}</div><div class="result-copy">{escape(result["explanation"])}</div></div>',
        unsafe_allow_html=True,
    )
    st.metric("Estimated bad-credit probability", f"{probability:.1%}")
    st.progress(probability, text="Estimated probability of the bad-credit class")

    with st.container(border=True):
        st.markdown("### One-feature sensitivity")
        st.caption("Vary one numeric input while holding the other applicant fields fixed. This is model sensitivity, not a causal explanation.")
        minimum, maximum, _ = NUMERIC_INPUTS[numeric_feature]
        variation_values = sorted({
            int(round(minimum + (maximum - minimum) * step / 12))
            for step in range(13)
        })
        sensitivity_records = []
        for value in variation_values:
            record = dict(row)
            record[numeric_feature] = value
            sensitivity_records.append(record)
        sensitivity_probabilities = model.predict_proba(pd.DataFrame(sensitivity_records))[:, 1]
        sensitivity_data = pd.DataFrame({
            FEATURE_LABELS[numeric_feature]: variation_values,
            "Bad-credit probability": sensitivity_probabilities,
        })
        st.line_chart(
            sensitivity_data,
            x=FEATURE_LABELS[numeric_feature],
            y="Bad-credit probability",
            height=280,
        )
    st.caption("This is an educational model output, not a credit score, lending recommendation, or estimate of an individual's financial worth.")


def dataset_explorer():
    page_header("Dataset explorer", "Inspect the validated source data, filter records, and review feature definitions.", "1,000 source records")
    dataset = get_credit_dataset()
    stats = st.columns(3)
    with stats[0]:
        metric_card("Records", f"{len(dataset):,}", "UCI Statlog German Credit", "blue")
    with stats[1]:
        metric_card("Input features", str(len(FEATURE_NAMES) - 1), "Validated model fields", "good")
    with stats[2]:
        metric_card("Missing values", f"{int(dataset.isna().sum().sum()):,}", "Across all source columns", "violet")

    st.write("")
    class_filter, amount_filter = st.columns([1, 2])
    with class_filter:
        selected_class = st.selectbox("Credit outcome", ["All outcomes", "Good credit", "Bad credit"])
    amount_min = int(dataset["credit_amount"].min())
    amount_max = int(dataset["credit_amount"].max())
    with amount_filter:
        selected_amounts = st.slider("Credit amount range", amount_min, amount_max, (amount_min, amount_max), step=250)

    filtered = dataset[dataset["credit_amount"].between(*selected_amounts)]
    if selected_class == "Good credit":
        filtered = filtered[filtered[TARGET_COLUMN] == 0]
    elif selected_class == "Bad credit":
        filtered = filtered[filtered[TARGET_COLUMN] == 1]
    display_data = filtered.rename(columns=FEATURE_LABELS).copy()
    display_data["Credit outcome"] = filtered[TARGET_COLUMN].map({0: "Good credit", 1: "Bad credit"})
    if TARGET_COLUMN in display_data:
        display_data = display_data.drop(columns=[TARGET_COLUMN])
    st.caption(f"Showing {len(filtered):,} of {len(dataset):,} records")
    st.dataframe(display_data.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered records",
        data=display_data.to_csv(index=False).encode("utf-8"),
        file_name="german_credit_filtered.csv",
        mime="text/csv",
    )

    left, right = st.columns([1, 1.4], gap="large")
    with left, st.container(border=True):
        st.markdown("### Data quality")
        st.write(f"Duplicate rows: **{int(dataset.duplicated().sum())}**")
        st.write(f"Rows after filters: **{len(filtered):,}**")
        st.write("Target mapping: **Good=0, Bad=1**")
    with right, st.container(border=True):
        st.markdown("### Feature dictionary")
        feature_summary = [
            {"Feature": FEATURE_LABELS[field], "Description": FEATURE_DESCRIPTIONS.get(field, "No description available.")}
            for field in FEATURE_NAMES if field != TARGET_COLUMN
        ]
        st.dataframe(pd.DataFrame(feature_summary), use_container_width=True, hide_index=True, height=300)


def model_performance():
    metrics = load_saved_metrics()
    preferred_model = metrics.get("selected_model", "unknown")
    page_header("Model performance", "Compare candidate models on held-out data and inspect saved diagnostics.", preferred_model.replace("_", " ").title())
    if not metrics:
        st.warning("No saved evaluation metrics were found. Train the model first.")
        return

    heldout = metrics.get("heldout_evaluation", {})
    y_true = heldout.get("y_true", [])
    candidate_probabilities = {
        name: values
        for name, values in heldout.get("probabilities", {}).items()
        if y_true and len(values) == len(y_true)
    }
    if not candidate_probabilities:
        model = load_model()
        if model is not None:
            X_test, y_test = get_heldout_data()
            y_true = [int(value) for value in y_test]
            candidate_probabilities = {preferred_model: model.predict_proba(X_test)[:, 1].tolist()}

    model_names = list(candidate_probabilities) or list(metrics.get("test_results", {}))
    if not model_names:
        st.warning("No candidate-model evaluation results were found. Train the model first.")
        return

    preferred_index = model_names.index(preferred_model) if preferred_model in model_names else 0
    selection_columns = st.columns([1, 1, 1.2])
    with selection_columns[0]:
        inspected_model = st.selectbox(
            "Model to inspect",
            model_names,
            index=preferred_index,
            format_func=lambda name: name.replace("_", " ").title(),
        )
    with selection_columns[1]:
        threshold = st.slider(
            "Bad-credit decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            format="%.2f",
            help="A record is classified as bad credit when its estimated bad-credit probability exceeds this value.",
        )

    dynamic_results = {}
    if y_true and candidate_probabilities:
        for name, probabilities in candidate_probabilities.items():
            result = evaluate_at_threshold(y_true, probabilities, threshold)
            result["roc_auc"] = float(roc_auc_score(y_true, probabilities))
            dynamic_results[name] = result

    selected_metrics = dynamic_results.get(inspected_model, metrics.get("test_results", {}).get(inspected_model, {}))
    threshold_results = selected_metrics
    importance_rows = metrics.get("feature_importance", {}).get(inspected_model, [])
    if not importance_rows and inspected_model == preferred_model:
        selected_pipeline = load_model()
        if selected_pipeline is not None:
            importance_rows = get_feature_importance_data(selected_pipeline).to_dict(orient="records")
    with selection_columns[2]:
        max_features = min(20, len(importance_rows))
        top_n = st.slider(
            "Features to display",
            min_value=min(5, max_features) if max_features else 1,
            max_value=max_features if max_features else 1,
            value=min(10, max_features) if max_features else 1,
            step=1,
            disabled=not importance_rows,
        )

    st.caption("The candidate comparison and confusion matrix recalculate at the selected threshold. The ROC curve and feature importance follow the selected model; ROC-AUC itself is threshold-independent.")
    metric_columns = st.columns(5)
    for column, label, key in zip(metric_columns, ("Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"), ("accuracy", "precision", "recall", "f1", "roc_auc")):
        with column:
            value = selected_metrics.get(key)
            st.metric(label, f"{value:.3f}" if value is not None else "--")

    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left, st.container(border=True):
        st.markdown("### Model comparison at current threshold")
        if dynamic_results:
            render_model_comparison(dynamic_results)
        else:
            table = [
                {"Model": name.replace("_", " ").title(), **{key: payload.get(key, 0) for key in ("accuracy", "precision", "recall", "f1", "roc_auc")}}
                for name, payload in metrics.get("test_results", {}).items()
            ]
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption("False positives are good-credit records predicted as bad; false negatives are bad-credit records predicted as good.")
    with right, st.container(border=True):
        st.markdown("### Cross-validation")
        cv_rows = [
            {"Model": name.replace("_", " ").title(), "Mean ROC-AUC": values.get("roc_auc_mean"), "Std. dev.": values.get("roc_auc_std")}
            for name, values in metrics.get("cv_results", {}).items()
        ]
        if cv_rows:
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)
        st.caption("Model selection uses training-fold cross-validation; final metrics use a separate held-out test split.")

    st.markdown("### Interactive diagnostics")
    diagnostic_columns = st.columns(2, gap="large")
    with diagnostic_columns[0], st.container(border=True):
        st.markdown("#### Confusion matrix at current threshold")
        if threshold_results.get("confusion_matrix") is not None:
            matrix_values = threshold_results["confusion_matrix"]
            confusion = pd.DataFrame(
                matrix_values,
                index=["Actual good credit", "Actual bad credit"],
                columns=["Predicted good", "Predicted bad"],
            )
            st.dataframe(confusion, use_container_width=True)
            error_columns = st.columns(2)
            with error_columns[0]:
                st.metric("False positives", matrix_values[0][1])
            with error_columns[1]:
                st.metric("False negatives", matrix_values[1][0])

    with diagnostic_columns[1], st.container(border=True):
        st.markdown(f"#### ROC curve · {inspected_model.replace('_', ' ').title()}")
        probabilities = candidate_probabilities.get(inspected_model)
        if y_true and probabilities:
            false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
            roc_data = pd.DataFrame({
                "False positive rate": false_positive_rate,
                "True positive rate": true_positive_rate,
            })
            st.line_chart(roc_data, x="False positive rate", y="True positive rate", height=300)
            st.metric("ROC-AUC", f"{selected_metrics.get('roc_auc', 0):.3f}")
        else:
            st.info("Held-out probabilities are not available for this model.")

    with st.container(border=True):
        st.markdown(f"#### Precision / recall trade-off · {inspected_model.replace('_', ' ').title()}")
        probabilities = candidate_probabilities.get(inspected_model)
        if y_true and probabilities:
            tradeoff_rows = []
            for candidate_threshold in (step / 100 for step in range(5, 100, 5)):
                point = evaluate_at_threshold(y_true, probabilities, candidate_threshold)
                tradeoff_rows.append({
                    "Threshold": candidate_threshold,
                    "Precision": point["precision"],
                    "Recall": point["recall"],
                    "F1 score": point["f1"],
                })
            tradeoff_data = pd.DataFrame(tradeoff_rows)
            st.line_chart(
                tradeoff_data,
                x="Threshold",
                y=["Precision", "Recall", "F1 score"],
                height=300,
            )
            st.caption(f"Selected threshold: {threshold:.2f}. Moving the slider above updates the error counts and headline metrics; this curve shows the model's held-out trade-off across cutoffs.")
        else:
            st.info("Held-out probabilities are not available for this model.")

    with st.container(border=True):
        st.markdown(f"#### Feature importance · {inspected_model.replace('_', ' ').title()}")
        render_feature_importance(importance_rows, top_n)


def batch_scoring():
    page_header("Batch scoring", "Score a CSV against the saved pipeline and export the enriched records.", "CSV workflow")
    required_columns = [field for field in FEATURE_NAMES if field != TARGET_COLUMN]
    template = pd.DataFrame(columns=required_columns).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV schema",
        data=template,
        file_name="credit_scoring_input_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload applicant records as CSV", type=["csv"], help="The header row must contain all 20 German Credit input fields.")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            model = load_model()
            if model is None:
                st.warning("A trained model is required. Run `python train_model.py` from the project root.")
                return
            scored = batch_predict(df, model=model)
            scored["risk_label"] = scored["predicted_class"].map({0: "Good credit", 1: "Bad credit"})
            summary = st.columns(3)
            with summary[0]:
                metric_card("Records scored", f"{len(scored):,}", uploaded.name, "blue")
            with summary[1]:
                bad_count = int((scored["predicted_class"] == 1).sum())
                metric_card("Predicted bad", f"{bad_count:,}", f"{bad_count / len(scored):.1%} of uploaded records" if len(scored) else "No records", "bad")
            with summary[2]:
                metric_card("Predicted good", f"{len(scored) - bad_count:,}", f"{1 - bad_count / len(scored):.1%} of uploaded records" if len(scored) else "No records", "good")
            st.write("")
            st.dataframe(scored.head(100), use_container_width=True, hide_index=True)
            if len(scored):
                st.bar_chart(scored["risk_label"].value_counts().rename_axis("Predicted outcome").to_frame("Records"))
            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button("Download scored results", data=csv_bytes, file_name="scored_predictions.csv", mime="text/csv", type="primary")
        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"The uploaded CSV is incompatible with the model schema: {exc}")


def about_page():
    page_header("Responsible use", "Understand the dataset scope, model behavior, and safeguards required before any consequential use.", "Education only")
    st.warning(EDUCATIONAL_DISCLAIMER)
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("### Data and model scope")
        st.markdown(
            """
            - Source: UCI Statlog German Credit dataset, with 1,000 historical records.
            - Target mapping: original label 1 becomes Good=0; original label 2 becomes Bad=1.
            - Candidate models: Logistic Regression, Decision Tree, and Random Forest, compared with stratified cross-validation.
            - The score describes a model classification for this dataset, not a verified probability of future default.
            """
        )
    with right, st.container(border=True):
        st.markdown("### Limits and safeguards")
        st.markdown(
            """
            - The sample is small, historical, and may encode social and institutional bias.
            - No fairness, legal, regulatory, or production-readiness certification is provided.
            - Real financial use would require representative data, independent validation, privacy controls, documented governance, ongoing monitoring, and qualified human review.
            - Do not use these outputs to approve, deny, price, or prioritize real credit applications.
            """
        )
    st.caption("This project is a learning and model-evaluation exercise. It is not a lending decision system.")


pages = {
    "Overview": home_dashboard,
    "Predict": single_prediction_page,
    "Dataset": dataset_explorer,
    "Performance": model_performance,
    "Batch scoring": batch_scoring,
    "Responsible use": about_page,
}

page_name = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
st.sidebar.markdown(
    '<div class="sidebar-foot">UCI German Credit<br>Experimental pipeline · Good=0 / Bad=1</div>',
    unsafe_allow_html=True,
)
pages[page_name]()
