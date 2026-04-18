"""
app.py
=======
GenAI Capstone — Unified Streamlit UI
Tab 1: Milestone 1 — ML Question Difficulty Predictor
Tab 2: Milestone 2 — Agentic Assessment Design Assistant
"""

# ── NumPy 2.0 Monkey-Patch (MUST be first) ──
import numpy as np
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64

import os
import re
import joblib
import scipy.sparse as sp
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from scipy.sparse import hstack

# ── Milestone 2 Imports ──
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

# ── Load Environment ──
load_dotenv()

# ── Artifacts Path ──
ARTIFACTS_DIR = Path("artifacts")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Intelligent Exam Question Analyser",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (teammate's dark academic UI) ──
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  :root {
    --bg-deep:    #0d1117;
    --bg-card:    #161b22;
    --bg-hover:   #1c2130;
    --accent-gold:#d4a853;
    --accent-teal:#3bbfad;
    --accent-rose:#e05c7a;
    --accent-blue:#4d9de0;
    --text-main:  #e6edf3;
    --text-muted: #8b949e;
    --border:     #30363d;
  }

  .stApp { background-color: var(--bg-deep); }

  .app-header {
    background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 50%, #0d1117 100%);
    border-bottom: 2px solid var(--accent-gold);
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
  }
  .app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(212,168,83,0.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .app-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--accent-gold);
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
  }
  .app-header p { color: var(--text-muted); font-size: 0.95rem; font-weight: 300; margin: 0; }

  .badge {
    display: inline-block;
    background: rgba(212,168,83,0.12);
    border: 1px solid var(--accent-gold);
    color: var(--accent-gold);
    font-size: 0.72rem;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2px 8px;
    border-radius: 3px;
    margin-right: 6px;
    letter-spacing: 0.5px;
  }

  .result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    position: relative;
    transition: border-color 0.2s;
  }
  .result-card:hover { border-color: #40464f; }

  .card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-label.gold  { color: var(--accent-gold); }
  .card-label.teal  { color: var(--accent-teal); }
  .card-label.rose  { color: var(--accent-rose); }
  .card-label.blue  { color: var(--accent-blue); }
  .card-label.muted { color: var(--text-muted);  }

  .metric-tile {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent-gold);
    line-height: 1;
    margin-bottom: 0.3rem;
  }
  .metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .disclaimer-box {
    background: rgba(77,157,224,0.06);
    border: 1px solid rgba(77,157,224,0.25);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-size: 0.82rem;
    color: #8b949e;
    line-height: 1.7;
  }

  .step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.82rem;
    color: var(--text-muted);
  }
  .step-done   { color: var(--accent-teal); }
  .step-active { color: var(--accent-gold); }

  section[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
  }

  .stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
  }
  .stTextArea textarea:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 1px rgba(212,168,83,0.3) !important;
  }
  .stButton button {
    background: var(--accent-gold) !important;
    color: #0d1117 !important;
    font-weight: 700 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.55rem 1.5rem !important;
    font-size: 0.9rem !important;
    transition: all 0.2s !important;
  }
  .stButton button:hover {
    background: #e0b96a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(212,168,83,0.3) !important;
  }
  button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
  }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-deep); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="app-header">
  <div>
    <span class="badge">MILESTONE 1 + 2</span>
    <span class="badge">LANGGRAPH</span>
    <span class="badge">RAG · CHROMADB</span>
    <span class="badge">GROQ · LLAMA3</span>
  </div>
  <h1>🎓 Intelligent Exam Question Analyser</h1>
  <p>ML-based difficulty prediction (M1) · Agentic AI assessment design assistant (M2)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; color:#d4a853; font-size:0.75rem;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem;">
      ⚙ System Info
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; color:#8b949e; font-size:0.75rem; line-height:2;">
      <b style="color:#d4a853;">M1 Model:</b> XGBoost / RF / LR<br>
      <b style="color:#d4a853;">M2 LLM:</b> llama-3.3-70b (Groq)<br>
      <b style="color:#d4a853;">RAG:</b> ChromaDB + MiniLM<br>
      <b style="color:#d4a853;">Workflow:</b> LangGraph<br>
      <b style="color:#d4a853;">Features:</b> TF-IDF + Word/Char Count
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; color:#d4a853; font-size:0.75rem;
    letter-spacing:2px; text-transform:uppercase; margin-bottom:0.8rem;">
      ● M2 Agent Pipeline
    </div>""", unsafe_allow_html=True)

    PIPELINE_STEPS = [
        ("rag_retriever",    "1. RAG Retrieval",    "ChromaDB pedagogy lookup"),
        ("llm_analyser",     "2. LLM Analysis",     "Groq llama3 reasoning"),
        ("output_formatter", "3. Format Output",    "Structured report assembly"),
    ]
    completed = st.session_state.get("steps_completed", [])
    for step_id, step_name, step_desc in PIPELINE_STEPS:
        icon = "✓" if step_id in completed else "○"
        css  = "step-done" if step_id in completed else ""
        st.markdown(f"""
        <div class="step-item {css}">
          <span style="font-family:'IBM Plex Mono',monospace;font-weight:600;">{icon}</span>
          <div>
            <div style="font-weight:500;">{step_name}</div>
            <div style="font-size:0.72rem;opacity:0.7;">{step_desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2 = st.tabs([
    "📊 Milestone 1 — ML Difficulty Predictor",
    "🤖 Milestone 2 — Agentic Assessment Designer"
])


# ═════════════════════════════════════════════
# MILESTONE 1 — ML Predictor
# ═════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="result-card">
      <div class="card-label gold">● ML-Based Question Difficulty Predictor</div>
      <span style="color:#8b949e; font-size:0.88rem;">
        Uses classical ML (TF-IDF + Word Count + Char Length) trained on your exam dataset
        to predict question difficulty as Easy / Medium / Hard.
      </span>
    </div>""", unsafe_allow_html=True)

    @st.cache_resource
    def load_ml_artifacts():
        try:
            vectorizer    = joblib.load(ARTIFACTS_DIR / "vectorizer.pkl")
            scaler        = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
            label_encoder = joblib.load(ARTIFACTS_DIR / "encoder.pkl")
            model         = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
            return vectorizer, scaler, label_encoder, model
        except FileNotFoundError as e:
            st.error(f"Model artifact not found: {e}. Make sure .pkl files are in the artifacts/ folder.")
            return None, None, None, None

    vectorizer, scaler, label_encoder, model = load_ml_artifacts()

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.strip()

    st.markdown("### Enter Your Question")
    question_text = st.text_area(
        "Question Text",
        height=150,
        placeholder="Enter exam question here...",
        label_visibility="collapsed",
    )

    if st.button("🔍 Predict Difficulty", use_container_width=False):
        if not question_text.strip():
            st.warning("Please enter a question.")
        elif model is None:
            st.error("ML model not loaded.")
        else:
            try:
                cleaned_q  = clean_text(question_text)
                word_count = len(cleaned_q.split())
                char_len   = len(cleaned_q)

                tfidf_features = vectorizer.transform([cleaned_q])
                num_features   = scaler.transform([[word_count, char_len]])
                X              = hstack([tfidf_features, sp.csr_matrix(num_features)])

                prediction = model.predict(X)
                difficulty = label_encoder.inverse_transform(prediction)[0]
                proba      = model.predict_proba(X)[0]
                confidence = round(max(proba) * 100, 2)

                # ── Metrics ──
                m1, m2, m3 = st.columns(3)
                color_map = {"Easy": "#3bbfad", "Medium": "#d4a853", "Hard": "#e05c7a"}
                color = color_map.get(difficulty, "#8b949e")
                with m1:
                    st.markdown(f"""<div class="metric-tile">
                    <div class="metric-value" style="color:{color};">{difficulty}</div>
                    <div class="metric-label">Predicted Difficulty</div></div>""",
                    unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="metric-tile">
                    <div class="metric-value">{confidence}<span style="font-size:1rem;color:#8b949e;">%</span></div>
                    <div class="metric-label">Confidence</div></div>""",
                    unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="metric-tile">
                    <div class="metric-value">{word_count}</div>
                    <div class="metric-label">Word Count</div></div>""",
                    unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Probability bar ──
                st.markdown("""<div class="result-card">
                <div class="card-label gold">● Probability Distribution</div>""",
                unsafe_allow_html=True)

                prob_df = pd.DataFrame({
                    "Difficulty": label_encoder.classes_,
                    "Probability (%)": [round(p * 100, 2) for p in proba]
                })
                st.bar_chart(prob_df.set_index("Difficulty"))
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ═════════════════════════════════════════════
# MILESTONE 2 — Agentic Assessment Designer
# ═════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="result-card">
      <div class="card-label teal">● Agentic Assessment Design Assistant</div>
      <span style="color:#8b949e; font-size:0.88rem;">
        Uses LangGraph + RAG (ChromaDB) + Groq LLM to autonomously reason about
        assessment quality and generate structured improvement recommendations.
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box" style="margin-bottom:1.2rem;">
      ✅ &nbsp;Analyse exam question quality &amp; Bloom's taxonomy coverage &nbsp;·&nbsp;
      ✅ &nbsp;Identify learning gaps &nbsp;·&nbsp;
      ✅ &nbsp;Suggest pedagogically-grounded improvements &nbsp;·&nbsp;
      ❌ &nbsp;Not for dataset queries or pass rate lookups (use Tab 1)
    </div>""", unsafe_allow_html=True)

    @st.cache_resource
    def load_agent():
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            vector_store = Chroma(
                persist_directory="chroma_db",
                embedding_function=embeddings,
                collection_name="pedagogy_kb",
            )
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                api_key=os.getenv("GROQ_API_KEY"),
            )
            return retriever, llm
        except Exception as e:
            st.error(f"Agent load error: {e}")
            return None, None

    retriever, llm = load_agent()

    class AssessmentState(TypedDict):
        exam_questions : List[str]
        retrieved_docs : List[str]
        analysis       : str
        summary        : str
        gaps           : str
        advice         : str
        refs           : str
        disclaimer     : str

    def rag_retriever_node(state: AssessmentState) -> AssessmentState:
        combined_query = " ".join(state["exam_questions"])
        docs           = retriever.invoke(combined_query)
        st.session_state["steps_completed"] = ["rag_retriever"]
        return {"retrieved_docs": [doc.page_content for doc in docs]}

    def llm_analyser_node(state: AssessmentState) -> AssessmentState:
        context   = "\n\n".join(state["retrieved_docs"])
        questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(state["exam_questions"])])
        prompt    = f"""
You are an expert educational assessment designer.
Using the pedagogy guidelines below, analyse the given exam questions.

PEDAGOGY GUIDELINES:
{context}

EXAM QUESTIONS:
{questions}

Provide your analysis in EXACTLY this format:

SUMMARY: (overall quality and difficulty distribution)

GAPS: (learning gaps or missing Bloom's taxonomy levels)

ADVICE: (specific improvements for each question)

REFS: (which pedagogy guidelines you referenced)

DISCLAIMER: (educational and ethical notices about this assessment)
"""
        response = llm.invoke(prompt)
        st.session_state["steps_completed"] = ["rag_retriever", "llm_analyser"]
        return {"analysis": response.content}

    def output_formatter_node(state: AssessmentState) -> AssessmentState:
        analysis = state["analysis"]

        def extract_section(text, section):
            try:
                start    = text.index(f"{section}:") + len(f"{section}:")
                sections = ["SUMMARY", "GAPS", "ADVICE", "REFS", "DISCLAIMER"]
                end      = len(text)
                for ns in [s for s in sections if s != section]:
                    try:
                        pos = text.index(f"{ns}:", start)
                        if pos < end:
                            end = pos
                    except ValueError:
                        continue
                return text[start:end].strip()
            except ValueError:
                return "Not available."

        st.session_state["steps_completed"] = ["rag_retriever", "llm_analyser", "output_formatter"]
        return {
            "summary"    : extract_section(analysis, "SUMMARY"),
            "gaps"       : extract_section(analysis, "GAPS"),
            "advice"     : extract_section(analysis, "ADVICE"),
            "refs"       : extract_section(analysis, "REFS"),
            "disclaimer" : extract_section(analysis, "DISCLAIMER"),
        }

    @st.cache_resource
    def build_agent(_retriever, _llm):
        graph_builder = StateGraph(AssessmentState)
        graph_builder.add_node("rag_retriever",    rag_retriever_node)
        graph_builder.add_node("llm_analyser",     llm_analyser_node)
        graph_builder.add_node("output_formatter", output_formatter_node)
        graph_builder.set_entry_point("rag_retriever")
        graph_builder.add_edge("rag_retriever",    "llm_analyser")
        graph_builder.add_edge("llm_analyser",     "output_formatter")
        graph_builder.add_edge("output_formatter", END)
        return graph_builder.compile()

    agent = build_agent(retriever, llm)

    # ── Input Section ──
    st.markdown("### 📝 Input Exam Questions")
    input_method = st.radio(
        "Choose input method:",
        ["Type manually", "Load from dataset CSV"],
        horizontal=True
    )

    exam_questions = []

    if input_method == "Type manually":
        raw_input = st.text_area(
            "Enter questions (one per line):",
            height=180,
            placeholder="What is Newton's second law?\nExplain the process of photosynthesis.\nDesign an experiment to test...",
            label_visibility="collapsed",
        )
        if raw_input.strip():
            exam_questions = [q.strip() for q in raw_input.strip().split("\n") if q.strip()]
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#3bbfad; margin-top:0.4rem;">
              ✓ {len(exam_questions)} question(s) detected
            </div>""", unsafe_allow_html=True)
    else:
        csv_path = st.text_input("CSV file path:", value="Dataset/raw_exam_data.csv")
        n_sample = st.slider("Number of questions to sample:", 3, 20, 5)
        if st.button("📂 Load from CSV"):
            try:
                df             = pd.read_csv(csv_path)
                exam_questions = df["Question_Text"].dropna().sample(n_sample, random_state=42).tolist()
                st.session_state["loaded_questions"] = exam_questions
                st.success(f"✅ {len(exam_questions)} questions loaded.")
            except Exception as e:
                st.error(f"CSV load error: {e}")

        if "loaded_questions" in st.session_state:
            exam_questions = st.session_state["loaded_questions"]
            st.markdown("""<div class="result-card">
            <div class="card-label teal">● Loaded Questions</div>""", unsafe_allow_html=True)
            for i, q in enumerate(exam_questions, 1):
                st.markdown(f"**{i}.** {q[:120]}...")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("▶️ Run Assessment Agent", use_container_width=False):
        if not exam_questions:
            st.warning("Please enter or load questions first.")
        elif retriever is None or llm is None:
            st.error("Agent not loaded. Check your API key and ChromaDB.")
        else:
            with st.spinner("🤖 Agentic pipeline running..."):
                try:
                    st.session_state["steps_completed"] = []
                    initial_state = AssessmentState(
                        exam_questions = exam_questions,
                        retrieved_docs = [],
                        analysis       = "",
                        summary        = "",
                        gaps           = "",
                        advice         = "",
                        refs           = "",
                        disclaimer     = "",
                    )
                    result = agent.invoke(initial_state)

                    # ── Report Tabs ──
                    st.markdown("""
                    <div style="font-family:'Playfair Display',serif; font-size:1.4rem;
                    color:#d4a853; margin:1.5rem 0 1rem 0;">
                      📋 Assessment Analysis Report
                    </div>""", unsafe_allow_html=True)

                    r_tab1, r_tab2, r_tab3, r_tab4, r_tab5 = st.tabs([
                        "📊 Summary", "⚠️ Gaps", "💡 Advice", "📚 References", "⚖️ Disclaimer"
                    ])

                    with r_tab1:
                        st.markdown("""<div class="result-card">
                        <div class="card-label gold">● Assessment Quality Summary</div>""",
                        unsafe_allow_html=True)
                        st.markdown(result["summary"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    with r_tab2:
                        st.markdown("""<div class="result-card">
                        <div class="card-label rose">● Identified Learning Gaps</div>""",
                        unsafe_allow_html=True)
                        st.markdown(result["gaps"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    with r_tab3:
                        st.markdown("""<div class="result-card">
                        <div class="card-label teal">● Improvement Recommendations</div>""",
                        unsafe_allow_html=True)
                        st.markdown(result["advice"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    with r_tab4:
                        st.markdown("""<div class="result-card">
                        <div class="card-label blue">● Pedagogical References</div>""",
                        unsafe_allow_html=True)
                        st.markdown(result["refs"])
                        st.markdown("</div>", unsafe_allow_html=True)

                    with r_tab5:
                        st.markdown(f"""
                        <div class="disclaimer-box">
                          {result["disclaimer"].replace(chr(10), "<br>")}
                        </div>""", unsafe_allow_html=True)

                    # ── Download ──
                    report_text = f"""
INTELLIGENT ASSESSMENT ANALYSIS REPORT
=======================================

SUMMARY:
{result["summary"]}

GAPS:
{result["gaps"]}

ADVICE:
{result["advice"]}

REFS:
{result["refs"]}

DISCLAIMER:
{result["disclaimer"]}
"""
                    st.download_button(
                        label="⬇️ Download Report (.txt)",
                        data=report_text,
                        file_name="assessment_report.txt",
                        mime="text/plain",
                    )

                except Exception as e:
                    st.error(f"Agent error: {e}")

# ── Footer ──
st.markdown("---")
st.markdown(
    """<div style="text-align:center; font-family:'IBM Plex Mono',monospace;
    font-size:0.75rem; color:#8b949e; padding:1rem 0;">
    GenAI Capstone Project &nbsp;·&nbsp; Milestone 1 + Milestone 2 &nbsp;·&nbsp;
    Built with LangGraph · Groq · ChromaDB · Streamlit
    </div>""",
    unsafe_allow_html=True
)