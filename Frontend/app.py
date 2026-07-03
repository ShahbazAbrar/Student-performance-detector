"""Student Performance Detector – Streamlit Frontend.

A clean, product-style interface that accepts student details and returns
a performance prediction with a personalised recommendation plan.
No technical ML jargon is exposed to the user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import streamlit as st


# ─── Paths (resolved dynamically) ────────────────────────────────────────────

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
_OUTPUT_DIR = _PROJECT_ROOT / "Output"
_DATA_PATH = _PROJECT_ROOT / "Data" / "xAPI-Edu-Data.csv"


# ─── Page configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Performance Detector",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─── Premium CSS ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 860px;
    }

    /* ── Hero ──────────────────────────────────────────── */
    .hero {
        text-align: center;
        padding: 2.8rem 1rem 1.6rem 1rem;
    }
    .hero-icon {
        font-size: 3.4rem;
        margin-bottom: 0.35rem;
        filter: drop-shadow(0 4px 12px rgba(99,102,241,.35));
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        font-size: 2.55rem;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        line-height: 1.15;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.02rem;
        font-weight: 400;
        margin-top: 0.45rem;
        letter-spacing: 0.01em;
    }

    /* ── Divider ───────────────────────────────────────── */
    .gradient-divider {
        height: 3px;
        border: none;
        border-radius: 3px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #f97316);
        margin: 1.8rem 0 2rem 0;
        opacity: 0.55;
    }

    /* ── Section titles ────────────────────────────────── */
    .section-label {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #e2e8f0;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-label .icon {
        font-size: 1.25rem;
    }

    /* ── Form card ─────────────────────────────────────── */
    [data-testid="stForm"] {
        background: rgba(30, 32, 48, 0.55);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }

    /* ── Submit button ─────────────────────────────────── */
    [data-testid="stForm"] button[kind="formSubmit"] {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        color: #fff;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.08rem;
        letter-spacing: 0.03em;
        border: none;
        border-radius: 14px;
        padding: 0.75rem 2rem;
        margin-top: 0.8rem;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    [data-testid="stForm"] button[kind="formSubmit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.45);
    }
    [data-testid="stForm"] button[kind="formSubmit"]:active {
        transform: translateY(0);
    }

    /* ── Result cards ──────────────────────────────────── */
    .result-container {
        margin-top: 2rem;
    }
    .result-card {
        background: rgba(30, 32, 48, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 20px;
        padding: 2rem 2rem 1.6rem 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.22);
        animation: fadeSlideUp 0.5s ease-out;
    }
    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.55rem;
        margin-bottom: 0.2rem;
    }
    .result-header.high  { color: #34d399; }
    .result-header.medium { color: #fbbf24; }
    .result-header.low   { color: #f87171; }
    .result-risk {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.3rem;
    }

    /* ── Metric pills ──────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-bottom: 1.4rem;
    }
    .metric-pill {
        flex: 1;
        min-width: 130px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(148,163,184,0.1);
        border-radius: 14px;
        padding: 1rem 0.9rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    .metric-pill:hover {
        border-color: rgba(139, 92, 246, 0.4);
    }
    .metric-pill .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .metric-pill .value {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #e2e8f0;
    }

    /* ── Path steps ────────────────────────────────────── */
    .path-container {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin: 0.8rem 0 1.2rem 0;
    }
    .path-step {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.1));
        border: 1px solid rgba(139,92,246,0.2);
        color: #c4b5fd;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 0.4rem 0.85rem;
        border-radius: 10px;
        white-space: nowrap;
    }
    .path-arrow {
        color: #475569;
        font-size: 0.85rem;
    }

    /* ── Recommendation box ────────────────────────────── */
    .reco-box {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(236,72,153,0.06));
        border-left: 3px solid #a855f7;
        border-radius: 0 14px 14px 0;
        padding: 1rem 1.2rem;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ── Footer ────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        margin-top: 3rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(148,163,184,0.08);
    }

    /* ── Hide streamlit extras ─────────────────────────── */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Hero ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">🎓</div>
        <div class="hero-title">Student Performance Detector</div>
        <div class="hero-sub">Enter student details below and get an instant performance prediction with a personalised action plan.</div>
    </div>
    <hr class="gradient-divider">
    """,
    unsafe_allow_html=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _map_class(predicted_class: str) -> Dict[str, str]:
    """Return human-friendly performance label and risk level."""
    mapping = {
        "H": {"performance": "High Performer", "risk": "Low Risk", "css": "high"},
        "M": {"performance": "Average Performer", "risk": "Medium Risk", "css": "medium"},
        "L": {"performance": "Needs Improvement", "risk": "High Risk", "css": "low"},
    }
    return mapping.get(predicted_class, {"performance": "Unknown", "risk": "Unknown", "css": "medium"})


def _recommendation_path(risk_level: str) -> List[str]:
    paths = {
        "High Risk": [
            "Improve Attendance",
            "Increase Resource Usage",
            "Join Tutoring",
            "Weekly Monitoring",
            "Academic Safe Zone",
        ],
        "Medium Risk": [
            "Increase Resource Usage",
            "Boost Class Participation",
            "Weekly Monitoring",
            "Academic Safe Zone",
        ],
        "Low Risk": ["Maintain Current Plan", "Academic Safe Zone"],
    }
    return paths.get(risk_level, ["Academic Safe Zone"])


def _recommendation_text(risk: str) -> str:
    texts = {
        "High Risk": (
            "This student needs immediate support. Focus on improving attendance, "
            "increasing access to learning resources, and enrolling in tutoring sessions. "
            "Weekly monitoring is strongly recommended until performance stabilises."
        ),
        "Medium Risk": (
            "This student is performing at an average level but has room for improvement. "
            "Encourage more active use of learning resources, boost class participation, "
            "and set up periodic check-ins to track progress."
        ),
        "Low Risk": (
            "Great job! This student is performing well. Continue with the current study plan "
            "and maintain consistent engagement to stay on track."
        ),
    }
    return texts.get(risk, "No specific recommendation available.")


# ─── Country display helpers ──────────────────────────────────────────────────

_COUNTRY_LABELS = {
    "Egypt": "Egypt",
    "Iran": "Iran",
    "Iraq": "Iraq",
    "Jordan": "Jordan",
    "KW": "Kuwait",
    "KuwaIT": "Kuwait",
    "Lybia": "Libya",
    "Morocco": "Morocco",
    "Palestine": "Palestine",
    "SaudiArabia": "Saudi Arabia",
    "Syria": "Syria",
    "Tunis": "Tunisia",
    "USA": "United States",
    "lebanon": "Lebanon",
    "venzuela": "Venezuela",
}

_STAGE_LABELS = {
    "lowerlevel": "Primary School",
    "MiddleSchool": "Middle School",
    "HighSchool": "High School",
}

_SEMESTER_LABELS = {"F": "First Semester", "S": "Second Semester"}
_GENDER_LABELS = {"M": "Male", "F": "Female"}
_RELATION_LABELS = {"Father": "Father", "Mum": "Mother"}
_SURVEY_LABELS = {"Yes": "Yes", "No": "No"}
_SATISFACTION_LABELS = {"Good": "Satisfied", "Bad": "Not Satisfied"}
_ABSENCE_LABELS = {"Under-7": "Less than 7 days", "Above-7": "7 days or more"}


def _reverse_lookup(labels: Dict[str, str], display: str) -> str:
    """Find the original data value from a display label."""
    for key, value in labels.items():
        if value == display:
            return key
    return display


# ─── Load saved artefacts ─────────────────────────────────────────────────────

@st.cache_resource
def _load_model():
    model_path = _OUTPUT_DIR / "best_model.pkl"
    preprocessor_path = _OUTPUT_DIR / "preprocessor.pkl"
    if not model_path.exists() or not preprocessor_path.exists():
        return None, None
    return joblib.load(model_path), joblib.load(preprocessor_path)


@st.cache_data
def _load_dataset():
    if not _DATA_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(_DATA_PATH)
    except Exception:
        return pd.DataFrame()


model, preprocessor = _load_model()
dataset_df = _load_dataset()

if not dataset_df.empty and "Class" in dataset_df.columns:
    class_order = sorted(dataset_df["Class"].astype(str).unique().tolist())
else:
    class_order = ["H", "L", "M"]


# ─── Nationality / birthplace options ────────────────────────────────────────

_nat_raw = sorted(dataset_df["NationalITy"].astype(str).unique().tolist()) if (not dataset_df.empty and "NationalITy" in dataset_df.columns) else list(_COUNTRY_LABELS.keys())
_bp_raw = sorted(dataset_df["PlaceofBirth"].astype(str).unique().tolist()) if (not dataset_df.empty and "PlaceofBirth" in dataset_df.columns) else list(_COUNTRY_LABELS.keys())

_nat_display = [_COUNTRY_LABELS.get(n, n) for n in _nat_raw]
_bp_display = [_COUNTRY_LABELS.get(b, b) for b in _bp_raw]


# ─── Build the input form ────────────────────────────────────────────────────

st.markdown('<div class="section-label"><span class="icon">📋</span> Student Information</div>', unsafe_allow_html=True)

with st.form("predict_form"):
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        gender_display = st.selectbox("Gender", options=list(_GENDER_LABELS.values()), index=0)
        nationality_display = st.selectbox("Nationality", options=_nat_display, index=_nat_display.index("Kuwait") if "Kuwait" in _nat_display else 0)
        birthplace_display = st.selectbox("Place of Birth", options=_bp_display, index=_bp_display.index("Kuwait") if "Kuwait" in _bp_display else 0)
        stage_display = st.selectbox("Education Stage", options=list(_STAGE_LABELS.values()), index=1)
        grade = st.selectbox("Grade", options=sorted(dataset_df["GradeID"].astype(str).unique().tolist()) if (not dataset_df.empty and "GradeID" in dataset_df.columns) else ["G-08"], index=0)
        section = st.selectbox("Section", options=sorted(dataset_df["SectionID"].astype(str).unique().tolist()) if (not dataset_df.empty and "SectionID" in dataset_df.columns) else ["A", "B", "C"], index=0)
        topic = st.selectbox("Subject / Topic", options=sorted(dataset_df["Topic"].astype(str).unique().tolist()) if (not dataset_df.empty and "Topic" in dataset_df.columns) else ["IT"], index=0)
        semester_display = st.selectbox("Semester", options=list(_SEMESTER_LABELS.values()), index=0)

    with col_right:
        relation_display = st.selectbox("Guardian", options=list(_RELATION_LABELS.values()), index=0)
        raisedhands = st.slider("Hands Raised in Class", 0, 100, 50)
        visited = st.slider("Resources Visited", 0, 100, 50)
        announcements = st.slider("Announcements Viewed", 0, 100, 30)
        discussion = st.slider("Discussion Participation", 0, 100, 25)
        parent_survey_display = st.selectbox("Parent Answered Survey?", options=list(_SURVEY_LABELS.values()), index=0)
        parent_satisfaction_display = st.selectbox("Parent School Satisfaction", options=list(_SATISFACTION_LABELS.values()), index=0)
        absence_display = st.selectbox("Student Absence Days", options=list(_ABSENCE_LABELS.values()), index=0)

    submitted = st.form_submit_button("🔍  Detect Performance")


# ─── Run prediction ──────────────────────────────────────────────────────────

if submitted:
    if model is None or preprocessor is None:
        st.error("The prediction model could not be loaded. Please make sure the project pipeline has been run at least once.")
    else:
        # Reverse display labels back to raw data values
        gender_raw = _reverse_lookup(_GENDER_LABELS, gender_display)
        nationality_raw = _nat_raw[_nat_display.index(nationality_display)]
        birthplace_raw = _bp_raw[_bp_display.index(birthplace_display)]
        stage_raw = _reverse_lookup(_STAGE_LABELS, stage_display)
        semester_raw = _reverse_lookup(_SEMESTER_LABELS, semester_display)
        relation_raw = _reverse_lookup(_RELATION_LABELS, relation_display)
        survey_raw = _reverse_lookup(_SURVEY_LABELS, parent_survey_display)
        satisfaction_raw = _reverse_lookup(_SATISFACTION_LABELS, parent_satisfaction_display)
        absence_raw = _reverse_lookup(_ABSENCE_LABELS, absence_display)

        input_row = pd.DataFrame([{
            "gender": gender_raw,
            "NationalITy": nationality_raw,
            "PlaceofBirth": birthplace_raw,
            "StageID": stage_raw,
            "GradeID": grade,
            "SectionID": section,
            "Topic": topic,
            "Semester": semester_raw,
            "Relation": relation_raw,
            "raisedhands": raisedhands,
            "VisITedResources": visited,
            "AnnouncementsView": announcements,
            "Discussion": discussion,
            "ParentAnsweringSurvey": survey_raw,
            "ParentschoolSatisfaction": satisfaction_raw,
            "StudentAbsenceDays": absence_raw,
        }])

        try:
            processed = preprocessor.transform(input_row)
            predicted_index = int(model.predict(processed)[0])
            predicted_class = str(class_order[predicted_index]) if 0 <= predicted_index < len(class_order) else "Unknown"

            info = _map_class(predicted_class)
            risk = info["risk"]
            css_class = info["css"]
            path_steps = _recommendation_path(risk)
            reco_text = _recommendation_text(risk)

            # Store in session state so it persists
            st.session_state["result"] = {
                "performance": info["performance"],
                "risk": risk,
                "css": css_class,
                "path": path_steps,
                "recommendation": reco_text,
                "inputs": {
                    "hands_raised": raisedhands,
                    "resources_visited": visited,
                    "announcements_viewed": announcements,
                    "discussion": discussion,
                },
            }
        except Exception as err:
            st.error(f"Something went wrong during prediction: {err}")


# ─── Display result ──────────────────────────────────────────────────────────

if "result" in st.session_state:
    r = st.session_state["result"]

    # Build path HTML — no indentation so markdown won't treat it as code
    path_parts = []
    for i, step in enumerate(r["path"]):
        path_parts.append(f'<span class="path-step">{step}</span>')
        if i < len(r["path"]) - 1:
            path_parts.append('<span class="path-arrow">→</span>')
    path_html = "".join(path_parts)

    # Header
    st.markdown(f'<div class="result-container"><div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="result-header {r["css"]}">{r["performance"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-risk">Risk Level: {r["risk"]}</div>', unsafe_allow_html=True)

    # Metric pills
    st.markdown(
        f'<div class="metric-row">'
        f'<div class="metric-pill"><div class="label">Hands Raised</div><div class="value">{r["inputs"]["hands_raised"]}</div></div>'
        f'<div class="metric-pill"><div class="label">Resources</div><div class="value">{r["inputs"]["resources_visited"]}</div></div>'
        f'<div class="metric-pill"><div class="label">Announcements</div><div class="value">{r["inputs"]["announcements_viewed"]}</div></div>'
        f'<div class="metric-pill"><div class="label">Discussion</div><div class="value">{r["inputs"]["discussion"]}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Action path
    st.markdown(f'<div class="section-label"><span class="icon">🗺️</span> Recommended Action Path</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="path-container">{path_html}</div>', unsafe_allow_html=True)

    # Recommendation
    st.markdown(f'<div class="section-label"><span class="icon">💡</span> Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="reco-box">{r["recommendation"]}</div>', unsafe_allow_html=True)

    # Close card
    st.markdown('</div></div>', unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">Student Performance Detector &nbsp;·&nbsp; Built with ❤️</div>',
    unsafe_allow_html=True,
)
