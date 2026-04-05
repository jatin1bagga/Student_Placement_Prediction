import pickle
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ── Page config & CSS ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Placement Prediction", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .main { background: #f7f9fc; }
  .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
  .hero {
    background: linear-gradient(90deg, #183b66, #1f7a5d);
    color: white; padding: 1.3rem 1.4rem;
    border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.12); margin-bottom: 1rem;
  }
  .card {
    background: white; padding: 1rem; border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06); border: 1px solid #e9edf3; margin-bottom: 1rem;
  }
  .subtle { color: #5b6573; font-size: 0.95rem; }
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BINARY_COLS = ["Internships(Y/N)", "Training(Y/N)", "Innovative Project(Y/N)",
               "Technical Course(Y/N)", "Backlog in 5th sem"]
YES_NO = {"Yes": 1, "No": 0}

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_binary(v: str) -> str:
    return "Yes" if str(v).strip().lower() in {"yes", "y", "1", "true"} else "No"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].apply(normalize_binary)
        df[col + "_num"] = df[col].map(YES_NO)
    df["profile_score"] = (
        0.20 * (df["10th marks"] / 100) + 0.20 * (df["12th marks"] / 100)
        + 0.40 * (df["Cgpa"] / 10) + 0.10 * df["Internships(Y/N)_num"]
        + 0.05 * df["Training(Y/N)_num"] + 0.05 * df["Innovative Project(Y/N)_num"]
    )
    df["experience_index"] = (
        0.30 * df["Internships(Y/N)_num"] + 0.20 * df["Training(Y/N)_num"]
        + 0.20 * df["Innovative Project(Y/N)_num"] + 0.15 * df["Technical Course(Y/N)_num"]
        - 0.15 * df["Backlog in 5th sem_num"]
    )
    return df

def build_row(vals: dict) -> pd.DataFrame:
    return add_features(pd.DataFrame([vals]))

def predict_proba(mdl, row):
    return float(mdl.predict_proba(row)[0][1]) if hasattr(mdl, "predict_proba") else None

def make_recommendations(cgpa, m10, m12, internships, training, innovative, technical, backlog):
    checks = [
        (cgpa < 6.5,            "Improve CGPA with focused study and revision."),
        (m10 < 65 or m12 < 65,  "Strengthen fundamentals and aptitude practice."),
        (internships == "No",   "Try to complete at least one internship or live project."),
        (training == "No",      "Add job-oriented training like Python, SQL, Excel, or aptitude."),
        (innovative == "No",    "Work on one solid mini-project to show practical exposure."),
        (technical == "No",     "Add a technical certification or online course."),
        (backlog == "Yes",      "Clear backlog as soon as possible to avoid repeated gaps."),
    ]
    recs = [msg for cond, msg in checks if cond]
    return recs or ["Profile looks balanced. Focus on resume quality and mock interviews."]

@st.cache_resource
def load_pkl(path: Path):
    return pickle.load(open(path, "rb")) if path.exists() else None

def run_shap(mdl, bg_data, inp_df):
    """Returns pyplot figure or raises."""
    bg = bg_data.sample(min(80, len(bg_data)), random_state=42)
    pre = mdl.named_steps["preprocessor"]
    clf = mdl.named_steps["model"]
    X_bg  = pre.transform(bg)
    X_inp = pre.transform(inp_df)
    if hasattr(X_bg,  "toarray"): X_bg  = X_bg.toarray()
    if hasattr(X_inp, "toarray"): X_inp = X_inp.toarray()
    exp = shap.Explainer(clf.predict_proba, X_bg)(X_inp)
    try:
        exp.feature_names = list(pre.get_feature_names_out())
    except Exception:
        pass
    try:
        shap.plots.waterfall(exp[0, :, 1], show=False)
    except Exception:
        shap.plots.bar(exp[0, :, 1], show=False)
    return plt.gcf()

# ── Load artifacts ─────────────────────────────────────────────────────────────
model      = load_pkl(Path("models/best_model.pkl"))
background = load_pkl(Path("models/shap_background.pkl"))

if model is None:
    st.error("Model file not found. Please place `best_model.pkl` inside `models/`.")
    st.stop()

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1 style="margin:0;">🎓 Student Placement Prediction System</h1>
  <p style="margin:0.35rem 0 0 0; font-size:1.02rem;">
    Prediction + probability + what-if analysis + open-source explainability
  </p>
</div>""", unsafe_allow_html=True)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
sb = st.sidebar
sb.title("Student Profile"); sb.caption("Enter the details below.")

cgpa      = sb.slider("CGPA", 0.0, 10.0, 7.2, 0.1)
sb.markdown("### Demographics & Academics")
gender    = sb.selectbox("Gender", ["Male", "Female"])
board_10  = sb.selectbox("10th Board", ["State Board", "CBSE", "ICSE", "WBBSE", "Other"])
marks_10  = sb.slider("10th Marks (%)", 0.0, 100.0, 75.0, 1.0)
board_12  = sb.selectbox("12th Board", ["State Board", "CBSE", "ISC", "WBCHSE", "Diploma", "Other"])
marks_12  = sb.slider("12th Marks (%)", 0.0, 100.0, 72.0, 1.0)
stream    = sb.selectbox("Stream", [
    "Computer Science and Engineering", "Information Technology",
    "Electronics and Communication Engineering", "Mechanical Engineering",
    "Computer Science in AIML", "Other",
])
sb.markdown("### Profile Details")
internships = sb.selectbox("Internships",        ["Yes", "No"])
training    = sb.selectbox("Training",           ["Yes", "No"])
innovative  = sb.selectbox("Innovative Project", ["Yes", "No"])
technical   = sb.selectbox("Technical Course",   ["Yes", "No"])
comm_level  = sb.slider("Communication Level (1-5)", 1, 5, 3, 1)
backlog     = sb.selectbox("Backlog in 5th Sem", ["Yes", "No"], index=1)

WHAT_IF_OPTS = ["CGPA", "10th Marks (%)", "12th Marks (%)", "Internships",
                "Training", "Innovative Project", "Technical Course", "Backlog in 5th Sem"]
what_if_opt = sb.selectbox("What-if factor", WHAT_IF_OPTS)
num_opts = {"CGPA": (0.0, 10.0, cgpa, 0.1), "10th Marks (%)": (0.0, 100.0, marks_10, 1.0),
            "12th Marks (%)": (0.0, 100.0, marks_12, 1.0)}
what_if_val = (sb.slider("New value", *num_opts[what_if_opt]) if what_if_opt in num_opts
               else sb.selectbox("New value", ["Yes", "No"]))

predict_btn = sb.button("Predict Placement", width="stretch")

# ── Core data ─────────────────────────────────────────────────────────────────
raw = {"Gender": gender, "10th board": board_10, "10th marks": marks_10,
       "12th board": board_12, "12th marks": marks_12, "Stream": stream,
       "Cgpa": cgpa, "Internships(Y/N)": internships, "Training(Y/N)": training,
       "Innovative Project(Y/N)": innovative, "Technical Course(Y/N)": technical,
       "Communication level": comm_level, "Backlog in 5th sem": backlog}

input_df   = build_row(raw)
prediction = model.predict(input_df)[0]
probability = predict_proba(model, input_df)

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input Summary")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("CGPA", f"{cgpa:.1f}"); st.metric("10th Marks", f"{marks_10:.0f}%")
    with c2: st.metric("12th Marks", f"{marks_12:.0f}%"); st.metric("Internships", internships)
    with c3: st.metric("Training", training); st.metric("Backlog", backlog)
    st.info("Same notebook flow: cleaning → feature engineering → preprocessing → prediction.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What-if Analysis")
    if predict_btn and probability is not None:
        altered = raw.copy()
        col_map = {"CGPA": "Cgpa", "10th Marks (%)": "10th marks", "12th Marks (%)": "12th marks",
                   "Internships": "Internships(Y/N)", "Training": "Training(Y/N)",
                   "Innovative Project": "Innovative Project(Y/N)", "Technical Course": "Technical Course(Y/N)",
                   "Backlog in 5th Sem": "Backlog in 5th sem"}
        altered[col_map[what_if_opt]] = float(what_if_val) if what_if_opt in num_opts else what_if_val
        alt_prob = predict_proba(model, build_row(altered))
        st.metric("Current Probability", f"{probability * 100:.2f}%")
        st.metric("What-if Probability", f"{alt_prob * 100:.2f}%")
        st.write(f"Change: **{(alt_prob - probability) * 100:+.2f}%**")
    else:
        st.write("Run prediction to compare a scenario.")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Panel")
    if predict_btn:
        (st.success if prediction == 1 else st.error)("Placed" if prediction == 1 else "Not Placed")
        if probability is not None:
            st.markdown("### Placement Probability")
            st.progress(max(0.0, min(1.0, probability)))
            st.write(f"**{probability * 100:.2f}%**")

        st.markdown("### Personalized Recommendations")
        for rec in make_recommendations(cgpa, marks_10, marks_12, internships, training, innovative, technical, backlog):
            st.write(f"- {rec}")

        st.markdown("### Open-source Explanation")
        if SHAP_AVAILABLE and background is not None:
            try:
                st.pyplot(run_shap(model, background, input_df), clear_figure=True)
            except Exception as e:
                st.warning(f"SHAP could not render: {e}")
        else:
            st.caption("SHAP not available — showing profile scores.")
            st.dataframe(input_df[["profile_score", "experience_index"]], width="stretch", hide_index=True)

        st.markdown("### Inputs used")
        st.dataframe(input_df, width="stretch", hide_index=True)
    else:
        st.write("Click **Predict Placement** in the sidebar.")
        st.caption("Tip: adjust sliders, then use What-if to test scenarios.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="subtle">Built for student placement prediction using a notebook-aligned ML pipeline.</div>',
            unsafe_allow_html=True)
