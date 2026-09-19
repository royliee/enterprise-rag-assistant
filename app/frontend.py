import streamlit as st
import requests

# FastAPI backend URLs
BACKEND_CHAT_URL = "http://127.0.0.1:8000/chat"
BACKEND_UPLOAD_URL = "http://127.0.0.1:8000/upload"

st.set_page_config(page_title="Enterprise Doc AI", page_icon="🤖", layout="wide")

# --- Sidebar: Document Management ---
with st.sidebar:
    st.title("📁 Document Knowledge Base")
    st.markdown("Upload documents (PDF, TXT) to ground the AI responses.")
    
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    if uploaded_file and st.button("Index Document", use_container_width=True):
        with st.spinner("Processing & vectorizing into ChromaDB..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            res = requests.post(BACKEND_UPLOAD_URL, files=files)
            if res.status_code == 200:
                data = res.json()
                st.success(f"Indexed {data.get('total_chunks', 0)} chunks successfully!")
            else:
                st.error(f"Error indexing file: {res.text}")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main Window: ChatGPT Interface ---
st.header("💬 Enterprise Assistant")

# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to answer any questions about your indexed documents."}
    ]

# Display all prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Chunk {idx}:**")
                    st.text(src)

# Bottom chat input bar (ChatGPT style)
if user_prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching context & generating response..."):
            try:
                response = requests.post(BACKEND_CHAT_URL, json={"query": user_prompt})
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer returned.")
                    sources = result.get("retrieved_context", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 View Retrieved Sources"):
                            for idx, src in enumerate(sources, 1):
                                st.caption(f"**Chunk {idx}:**")
                                st.text(src)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Backend error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to FastAPI backend. Make sure Uvicorn is running on port 8000.")
