import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from scipy.sparse import hstack
import os

# Configure the page
st.set_page_config(
    page_title="Intelligent Exam Question Analyzer", 
    page_icon="🎓", 
    layout="centered"
)

# Custom CSS for a premium design
st.markdown("""
<style>
    /* Styling the main container */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }
    
    /* Input Text Area Styling */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-size: 16px;
        padding: 15px;
        transition: all 0.3s ease;
    }
    .stTextArea textarea::placeholder {
        color: #a0aec0 !important;
    }
    .stTextArea textarea:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600;
        font-size: 18px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Result Widget Styling */
    .result-widget {
        background: #ffffff !important;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 30px;
        margin-bottom: 20px;
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Difficulty Colors */
    .diff-Easy { color: #10b981 !important; }
    .diff-Medium { color: #f59e0b !important; }
    .diff-Hard { color: #ef4444 !important; }
    
    /* Fix for text colors in the white widget */
    .result-widget p {
        color: #718096 !important; 
    }
    
    /* Subheaders */
    h1 {
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #a0aec0 !important;
        font-size: 18px;
        margin-bottom: 40px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    try:
        vectorizer = joblib.load('artifacts/vectorizer.pkl')
        scaler = joblib.load('artifacts/scaler.pkl')
        label_encoder = joblib.load('artifacts/label_encoder.pkl')
        best_model = joblib.load('artifacts/best_model.pkl')
        return vectorizer, scaler, label_encoder, best_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'[^\w\s]', '', text) 
    return text

# Header Section
st.markdown("<h1>🎓 Question Difficulty Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by Classical Machine Learning</p>", unsafe_allow_html=True)

vectorizer, scaler, label_encoder, best_model = load_artifacts()

if best_model is None:
    st.error("Model artifacts not found. Please ensure that the 'artifacts' folder exists with vectorizer.pkl, scaler.pkl, label_encoder.pkl, and best_model.pkl.")
else:
    # Input area
    question = st.text_area(
        "", 
        height=160, 
        placeholder="Type or paste your exam question here...\n\ne.g. Synthesize asymptotic complexity with mathematical proofs."
    )
    
    if st.button("Analyze Question"):
        if not question.strip():
            st.warning("⚠️ Please enter a question to analyze.")
        else:
            with st.spinner("Analyzing complexity and patterns..."):
                # 1. Preprocessing
                cleaned_text = clean_text(question)
                word_count = len(cleaned_text.split())
                char_length = len(cleaned_text)
                
                # 2. Feature engineering
                X_text_tfidf = vectorizer.transform([cleaned_text])
                X_num_scaled = scaler.transform([[word_count, char_length]])
                X_final = hstack([X_text_tfidf, X_num_scaled])
                
                # 3. Prediction
                pred_idx = best_model.predict(X_final)[0]
                difficulty = label_encoder.inverse_transform([pred_idx])[0]
                
                # 4. Probabilities (if model supports predict_proba)
                conf_text = ""
                if hasattr(best_model, "predict_proba"):
                    probs = best_model.predict_proba(X_final)[0]
                    confidence = np.max(probs) * 100
                    conf_text = f"Confidence Score: {confidence:.1f}%"
                
                # Map formatting
                emoji_map = {"Easy": "🟢", "Medium": "🟠", "Hard": "🔴"}
                emoji = emoji_map.get(difficulty, "⚪")
                
                # 5. Display Result
                st.markdown(f"""
                <div class="result-widget">
                    <p style="color: #718096; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Predicted Difficulty</p>
                    <h2 class="diff-{difficulty}" style="font-size: 48px; margin: 10px 0;">{emoji} {difficulty}</h2>
                    <p style="color: #a0aec0; font-size: 16px; font-weight: 500;">{conf_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional Stats
                st.markdown("### 📊 Text Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Words", word_count)
                
                with col2:
                    st.metric("Total Characters", char_length)
