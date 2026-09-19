import streamlit as st
import requests

st.set_page_config(page_title="Document AI Assistant", layout="centered")

st.title("📄 Local Document Assistant")
st.caption("Upload a PDF or TXT file, then chat directly with it.")

# 1. File Upload Section
uploaded_file = st.file_uploader("Upload your document here", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Processing and indexing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post("http://127.0.0.1:8000/upload", files=files)
                if response.status_code == 200:
                    st.success(f"Indexed '{uploaded_file.name}' successfully!")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server: {e}")

st.divider()

# 2. Chat Query Section
user_question = st.text_input("Ask a question about your document:")

if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please type a question first.")
    else:
        with st.spinner("Finding answer..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={"query": user_question}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    
                    with st.expander("Show Retrieved Source Context"):
                        for i, chunk in enumerate(data["retrieved_context"]):
                            st.write(f"**Snippet {i+1}:** {chunk}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server: {e}")
