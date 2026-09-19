import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Ensure .env is loaded from the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=API_KEY
    )
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def ingest_uploaded_file(file_path: str):
    """Parses an uploaded TXT or PDF, chunks it, and indexes it into ChromaDB."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .txt or .pdf file.")

    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(raw_docs)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    return {
        "filename": os.path.basename(file_path),
        "total_chunks": len(chunks),
        "status": "Successfully ingested and indexed into vector store."
    }

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag(user_query: str):
    """Retrieves matching chunks and passes them to Gemini for grounded answering."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(user_query)
    sources = [doc.page_content for doc in retrieved_docs]

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=API_KEY
    )

    prompt = ChatPromptTemplate.from_template(
        "You are an enterprise AI assistant. Answer using ONLY the provided context:\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    rag_chain = (
        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(user_query)
    return {"answer": answer, "retrieved_context": sources}
