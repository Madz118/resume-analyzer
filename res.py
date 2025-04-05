import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Extract text from PDF ---
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# --- Clean text ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

# --- Match skills ---
def match_skills(resume_text, job_desc_text):
    vectorizer = CountVectorizer().fit_transform([resume_text, job_desc_text])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    match_score = round(similarity[0][1] * 100, 2)

    resume_words = set(resume_text.split())
    job_desc_words = set(job_desc_text.split())

    matched_skills = list(job_desc_words & resume_words)
    missing_skills = list(job_desc_words - resume_words)

    return match_score, matched_skills, missing_skills

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Analyzer", layout="centered")
st.title("📄 AI Resume Analyzer")

uploaded_resume = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
job_desc_input = st.text_area("Paste the Job Description")

if st.button("Analyze"):
    if uploaded_resume and job_desc_input:
        with st.spinner("Analyzing..."):
            resume_text = clean_text(extract_text_from_pdf(uploaded_resume))
            job_desc_text = clean_text(job_desc_input)

            match_percent, matched, missing = match_skills(resume_text, job_desc_text)

            st.subheader("✅ Results")
            st.write(f"**Match Percentage:** {match_percent}%")
            st.write(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
            st.write(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")
    else:
        st.error("Please upload your resume and paste the job description.")
