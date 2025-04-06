import streamlit as st
import pdfplumber
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Function to extract text from uploaded PDF
@st.cache_data
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

# Function to clean and tokenize text
def clean_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    words = [word for word in tokens if word.isalpha()]
    return ' '.join(words)

# Function to compute match percentage
def match_percentage(resume, jd):
    cv = CountVectorizer()
    count_matrix = cv.fit_transform([resume, jd])
    match_score = cosine_similarity(count_matrix)[0][1] * 100
    return round(match_score, 2)

# Function to send email (Dummy SMTP - needs real credentials to work)
def send_email_report(to_email, subject, body):
    sender_email = "your_email@example.com"
    password = "your_password"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
        return True
    except Exception as e:
        return False

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🤖 AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Smartly match your resume to any job description</h4>", unsafe_allow_html=True)

with st.sidebar:
    st.title("Resume Analyzer Panel")
    st.markdown("""
    ### Instructions:
    1. Upload one or more resumes in PDF format.
    2. Paste the full job description.
    3. (Optional) Enter your email to receive a report.
    4. Click **Analyze Resumes** below.
    """)
    uploaded_resumes = st.file_uploader("\U0001F4E4 Upload Resumes (PDF only)", type=["pdf"], accept_multiple_files=True)
    job_description = st.text_area("\U0001F4DD Job Description", height=250, placeholder="Paste the job description here...")
    email_input = st.text_input("\U0001F4E7 Email for Report (optional)")
    analyze_clicked = st.button("\U0001F50D Analyze Resumes")

if analyze_clicked:
    if uploaded_resumes and job_description.strip():
        st.session_state.resume_texts = [extract_text_from_pdf(resume) for resume in uploaded_resumes]
        st.session_state.resume_names = [resume.name for resume in uploaded_resumes]
        st.session_state.jd_text = job_description
        st.session_state.email = email_input
        st.session_state.analysis_ready = True
        st.success("✅ Analysis complete! View results below.")
    else:
        st.error("🚨 Upload at least one resume and provide a job description.")

if st.session_state.get("analysis_ready", False):
    jd_words = clean_text(st.session_state.jd_text)
    results = []

    for i, resume_text in enumerate(st.session_state.resume_texts):
        resume_words = clean_text(resume_text)
        match = match_percentage(resume_words, jd_words)
        resume_set = set(resume_words.split())
        jd_set = set(jd_words.split())
        missing_skills = jd_set - resume_set
        results.append({
            "name": st.session_state.resume_names[i],
            "match": match,
            "missing": list(missing_skills)
        })

    st.header("📊 Resume Analysis Results")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Match Percentage Bar Chart")
        fig, ax = plt.subplots()
        bars = sns.barplot(x=[res["name"] for res in results], y=[res["match"] for res in results], palette="coolwarm", ax=ax)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        ax.set_ylabel("Match %")
        ax.set_xlabel("Resume")
        ax.set_title("Resume vs Job Description Match")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("Match Percentage Pie Chart")
        if results:
            fig2, ax2 = plt.subplots()
            labels = [res["name"] for res in results]
            sizes = [res["match"] for res in results]
            explode = [0.05]*len(labels) if len(labels) <= 3 else None
            if any(sizes):
                ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode)
                ax2.axis('equal')
                st.pyplot(fig2)
            else:
                st.warning("⚠️ Cannot generate pie chart: All match percentages are 0%.")
        else:
            st.warning("⚠️ No results to display in pie chart.")

    st.markdown("---")
    st.subheader("📚 Resume Suggestions")
    for res in results:
        st.markdown(f"### \U0001F4CC {res['name']}")
        match_color = "green" if res['match'] >= 75 else "orange" if res['match'] >= 40 else "red"
        st.markdown(f"<span style='color:{match_color}; font-weight:bold;'>Match: {res['match']}%</span>", unsafe_allow_html=True)

        if res['missing']:
            top_missing = res['missing'][:10]
            st.write("**Missing Skills (Top 10):**")
            for skill in top_missing:
                st.markdown(f"- {skill} \U0001F4A1 [Learn more](https://www.google.com/search?q=how+to+learn+{skill})")
            st.info("💡 Suggestions: Try incorporating these keywords/skills in your resume if they apply to you.")
        else:
            st.success("Great! Resume covers all required skills.")

    if st.session_state.email:
        report_body = "\n\n".join([f"Resume: {r['name']}\nMatch: {r['match']}%\nMissing: {', '.join(r['missing'])}" for r in results])
        if send_email_report(st.session_state.email, "Your Resume Match Report", report_body):
            st.success("📧 Report emailed successfully!")
        else:
            st.warning("❌ Failed to send email. Check credentials or try again later.")

    summary_text = "\n\n".join([f"{r['name']} - Match: {r['match']}%\nMissing Skills: {', '.join(r['missing'])}" for r in results])
    b64 = base64.b64encode(summary_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="match_summary.txt">📥 Download Summary Report</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    st.info("👈 Use the sidebar to upload resumes and job description, then click Analyze.")
