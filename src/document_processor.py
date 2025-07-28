####################### File upload and processing section ######################
import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPowerPointLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


################# Set up logging ########################################
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, 'chatbot.log')

############   Remove existing handlers  if any logs present ###############
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # Also print to terminal/Streamlit logs
    ]
)

##################  Processs Document Function ##########################
def process_document_path(filepath: str):
    """Loads supported document, splits into chunks for RAG, and returns them."""
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"{filepath} not found")

    logging.info(f"Processing file: {filepath}")
    file_ext = filepath.lower().split('.')[-1]
    try:
        if file_ext == 'pdf':
            loader = PyPDFLoader(filepath)
        elif file_ext == 'pptx':
            loader = UnstructuredPowerPointLoader(filepath)
        elif file_ext == 'docx':
            loader = UnstructuredWordDocumentLoader(filepath)
        elif file_ext == 'txt':
            loader = TextLoader(filepath)
        else:
            logging.error(f"Unsupported file type: {filepath}")
            raise ValueError(f" Unsupported file type: {file_ext}")
    except Exception as e:
        logging.exception("Error initializing document loader")
        raise e

    try:
        documents = loader.load()
    except Exception as e:
        logging.exception("Error loading document")
        raise e

    logging.info(f"Loaded {len(documents)} document(s)")

####################### Splitting into chunks #############################
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_documents(documents)

    logging.info(f"Generated {len(chunks)} chunks from document")
    print(f"Total Chunks Created: {len(chunks)}")

    return chunks
