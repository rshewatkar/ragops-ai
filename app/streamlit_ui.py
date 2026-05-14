import streamlit as st
import requests
import os
import re

# PAGE CONFIG

st.set_page_config(
    page_title="AI Resume Assistant",
    page_icon="📄",
    layout="centered"
)

# FASTAPI BACKEND URL

API_URL = os.getenv("API_URL", "http://localhost:8000/ask")

# For local testing without Docker, use:
# API_URL = "http://localhost:8000/ask"


# HEADER

st.markdown("""
<h1 style='text-align: center;'>📄 AI Resume Assistant</h1>
<p style='text-align: center; color: gray;'>
Ask anything about Rahul's profile, skills, experience, or projects
</p>
""", unsafe_allow_html=True)

st.divider()


# INPUT SECTION

query = st.text_input(
    "🔍 Search resume",
    placeholder="e.g. What are his skills? What projects has he built?"
)

col1, col2 = st.columns([1, 1])

with col1:
    ask_btn = st.button("Ask")

with col2:
    clear_btn = st.button("Clear")


# SESSION STATE

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# FUNCTION TO CALL API

def clean_display_text(text):
    text = str(text or "")
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"</?b>", "", text)
    return text.strip()


def ask_backend(question):
    try:
        response = requests.post(
            API_URL,
            json={"query": question},
            timeout=300
        )

        if response.status_code == 200:
            data = response.json()
            return {
                "answer": data.get("answer", "No answer returned by API")
            }

        return {
            "answer": f"Error: {response.status_code} - {response.text}"
        }

    except ValueError:
        return {
            "answer": "Error: API returned an invalid JSON response"
        }

    except Exception as e:
        return {
            "answer": f"Connection Error: {str(e)}"
        }


# ACTIONS

if ask_btn and query:

    with st.spinner("Thinking..."):

        response = ask_backend(query)

        st.session_state.answer = response["answer"]
        st.session_state.last_query = query


if clear_btn:
    st.session_state.answer = ""
    st.session_state.last_query = ""


# OUTPUT SECTION

if st.session_state.answer:

    st.markdown("### 📌 Result")
    answer = clean_display_text(st.session_state.answer)

    with st.container(border=True):
        st.markdown(f"**Query:** {st.session_state.last_query}")
        st.markdown("**Answer:**")
        st.markdown(answer)


# SUGGESTED QUESTIONS

st.divider()

st.markdown("### 💡 Try these:")

suggestions = [
    "What are his skills?",
    "What is his experience?",
    "What projects has he built?",
    "Tell me about his profile",
    "Which ML libraries does he know?"
]

cols = st.columns(2)

for i, q in enumerate(suggestions):

    if cols[i % 2].button(q):

        with st.spinner("Thinking..."):

            response = ask_backend(q)

            st.session_state.last_query = q
            st.session_state.answer = response["answer"]

            st.rerun()


# FOOTER

st.divider()

st.caption("Built with RAG + FastAPI + MLflow | Rahul Shewatkar")
