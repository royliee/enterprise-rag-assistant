import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="Enterprise Doc AI", page_icon="🤖", layout="wide")

# API Key handling
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Missing Gemini API Key. Please configure it in your secrets or .env file.")
    st.stop()

DB_DIR = "data/chroma_db"
DOCS_DIR = "data/docs"
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

@st.cache_resource
def load_vector_store():
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())

# --- Sidebar: Document Management ---
with st.sidebar:
    st.title("📁 Document Knowledge Base")
    st.markdown("Upload documents (PDF, TXT) to ground the AI responses.")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    if uploaded_file and st.button("Index Document", use_container_width=True):
        with st.spinner("Processing & vectorizing into ChromaDB..."):
            save_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if save_path.endswith(".pdf"):
                loader = PyPDFLoader(save_path)
            else:
                loader = TextLoader(save_path)

            raw_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
            chunks = splitter.split_documents(raw_docs)

            vector_store = load_vector_store()
            vector_store.add_documents(chunks)
            st.cache_resource.clear()
            st.success(f"Successfully indexed {len(chunks)} chunks from {uploaded_file.name}!")

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am ready to answer any questions about your indexed documents."}
        ]
        st.rerun()

# --- Main Window: ChatGPT Interface ---
st.header("💬 Enterprise Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am ready to answer any questions about your indexed documents."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Chunk {idx}:**")
                    st.text(src)

if user_prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching context & generating response..."):
            vector_store = load_vector_store()
            retriever = vector_store.as_retriever(search_kwargs={"k": 6})
            retrieved_docs = retriever.invoke(user_prompt)
            sources = [doc.page_content for doc in retrieved_docs]

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=api_key
            )

            prompt = ChatPromptTemplate.from_template(
                "You are an enterprise AI assistant. Answer using ONLY the provided context:\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}"
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(user_prompt)
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
