# src/chat.py

import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import AzureOpenAI
from src.vector_store import get_chroma_vectorstore

# Load environment variables
load_dotenv("config/.env")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################### Azure GPT setup #########################################
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)
AZURE_GPT_DEPLOYMENT = os.getenv("AZURE_GPT4_DEPLOYMENT") 

def retrieve_similar_chunks(query: str, k: int = 4) -> List[Document]:
    """Retrieve top-k similar chunks using ChromaDB."""
    vectorstore = get_chroma_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    print(f"Retrieved {len(results)} chunks for query: {query}")
    return results
#################################"Format chunks into a string for GPT context.###################
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])

def ask_gpt(query: str, context_docs: List[Document]) -> Dict[str, Any]:
  
    context = format_docs(context_docs)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer using only the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]

    logger.info("Sending prompt to GPT model...")

    response = client.chat.completions.create(
        model=AZURE_GPT_DEPLOYMENT,
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()
    
############### Store the refernce #########################
    top_doc = context_docs[0] if context_docs else None
    reference = {}
    if top_doc:
        reference = {
            "source": top_doc.metadata.get("source", "unknown"),
            "page": top_doc.metadata.get("page", "N/A"),
            
        }

    return {
        "answer": answer,
        "documents": context_docs,
        "references": [reference] if reference else []
    }