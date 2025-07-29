from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import sys
import shutil
import uuid
import logging



from src.document_processor import process_document_path
from src.vector_store import store_embeddings_in_chroma
from src.Chat import retrieve_similar_chunks, ask_gpt

app = FastAPI(
    title="GenAI RAG Chatbot",
    docs_url="/chatbot"       #  default docs to /chatbot
    
)

# Enable CORS (Optional if connecting from frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "txt", "pptx", "docx"]:
        return {"error": "Unsupported file type"}

    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded file saved to: {file_path}")

    try:
        chunks = process_document_path(file_path)
        store_embeddings_in_chroma(chunks)
        return {"message": f"{len(chunks)} chunks created and stored in vector DB."}
    except Exception as e:
        logger.exception("Error processing document")
        return {"error": str(e)}


@app.post("/chat/")
async def chat_with_bot(query: str = Form(...)):
    try:
        docs = retrieve_similar_chunks(query)
        response = ask_gpt(query, docs)
        return response
    except Exception as e:
        logger.exception("Error during chat process")
        return {"error": str(e)}
