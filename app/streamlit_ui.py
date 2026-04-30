import streamlit as st

try:
    from rag_chain import ask_rag
except ModuleNotFoundError as exc:
    if exc.name != "rag_chain":
        raise
    from app.rag_chain import ask_rag

# page config
st.set_page_config(
    page_title = "RAG Resume Assistant",
    page_icon = "📃",
    layout = "centered"
)

#Title
st.title("📃 RAG Resume Assistance")
st.markdown("Ask questions about Rahuls's resume")

# Input Box
query = st.text_input("🫡Enter your Questions:")

# Button
if st.button("🔍Ask"):
    if query.strip() == "":
         st.warning("Please enter a question")
    else:
        with st.spinner("Thinking.."):
            answer = ask_rag(query)
            
        st.success("answer:")
        st.write(answer)

# Footer
st.markdown("---") 
st.caption("Built with ❤️ using RAG + FastAPI + Streamlit")
