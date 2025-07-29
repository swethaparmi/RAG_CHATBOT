RAG Chatbot Processing Pipeline
 Upload
The user uploads one or more documents using the Streamlit UI.
Supported formats: PDF, DOCX, PPTX, TXT.

 Parsing
Uploaded files are parsed using Unstructured loaders to extract and clean raw text.

Chunking
The cleaned text is split into small, manageable chunks (e.g., 300–500 tokens) for embedding.

Embedding
Each chunk is converted into a high-dimensional vector using Azure OpenAI’s text-embedding-3-small model.

Storage
The embeddings are stored in a local ChromaDB vector database for fast similarity search.

Search (Retrieval)
When a user asks a question, the query is also embedded, and semantic similarity is used to retrieve the most relevant document chunks.

Response Generation
Retrieved chunks are passed to GPT-4o, which uses them as context to generate a grounded and context-aware answer.

Referenced Answer Output
The chatbot response includes:

The answer

The source document name

The page number or slide (if available)

