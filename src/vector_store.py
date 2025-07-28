import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

################# Load environment variables#############################
load_dotenv("config/.env")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

########################## ChromaDB Folder ####################################
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "Collections_data"  

# Optional logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################## Azure OpenAI embedding function ####################################
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_API_KEY,
    api_version="2025-01-01-preview"
)

##################################""Embed and store document chunks in ChromaDB.""" #######################
def store_embeddings_in_chroma(chunks: List[Document], persist_dir: str = PERSIST_DIR):
    """Embed and store document chunks in ChromaDB."""
    logger.info("Storing documents into ChromaDB...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()  #  Persist to disk
    logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")
    return vectorstore

################################## Get Chroma vector store instance ####################################
def get_chroma_vectorstore(persist_dir: str = PERSIST_DIR):
    """Load Chroma vector store with Azure embeddings."""
    logger.info("Loading ChromaDB with Azure embeddings...")
    return Chroma(
        collection_name=COLLECTION_NAME,  
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    logger.info("ChromaDB loaded successfully.")



