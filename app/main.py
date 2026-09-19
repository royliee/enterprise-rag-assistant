import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from app.rag import query_rag, ingest_uploaded_file

app = FastAPI(
    title="Enterprise Local RAG Agent",
    description="Containerized, network-isolated RAG service with dynamic document ingestion."
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: list[str]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a .pdf or .txt document directly via API and embed it."""
    valid_extensions = [".txt", ".pdf"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in valid_extensions:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest and index
    try:
        result = ingest_uploaded_file(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=QueryResponse)
def chat(payload: QueryRequest):
    try:
        return query_rag(payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))