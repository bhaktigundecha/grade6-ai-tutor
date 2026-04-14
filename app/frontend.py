import streamlit as st
import requests
import speech_recognition as sr

st.set_page_config(page_title="AI Tutor", layout="centered")

st.title("📚 AI Tutor (Grade 6)")

# Session
if "query" not in st.session_state:
    st.session_state.query = ""

# SUBJECT SELECTOR
subject = st.selectbox(
    "Select Subject",
    ["", "English", "Hindi", "Mathematics", "Science", "Social Science"]
)

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = r.listen(source)

    try:
        return r.recognize_google(audio)
    except:
        return ""

# Input
query = st.text_input("Type your question", value=st.session_state.query)

# Voice
if st.button("🎤 Speak"):
    spoken = speech_to_text()
    if spoken:
        st.session_state.query = spoken
        st.success(f"You said: {spoken}")

# Submit
if st.button("Submit"):
    if not query.strip():
        st.warning("Enter a question")
    else:
        try:
            res = requests.get(
                "http://127.0.0.1:8000/ask",
                params={
                    "query": query,
                    "subject": subject.lower() if subject else ""
                }
            )

            data = res.json()

            st.markdown("### 💬 Answer")
            st.success(data["answer"])

            st.markdown("### 📚 Source")
            st.info(f"{data['subject']} – {data['source']}")

        except:
            st.error("Backend not running")