The following steps outline the processing pipeline:
•	1. Upload: User uploads documents via Streamlit UI (PDF, DOCX, PPTX, TXT).
•	2. Parsing: Extract and clean raw text from documents using unstructured loaders.
•	3. Chunking: Split the text into small manageable pieces 
•	4. Embedding: Use Azure OpenAI's `text-embedding-3-small ` model to convert text chunks into high-dimensional vectors.
•	5. Storage: Save the vectors in a local ChromaDB database.
•	6. Search: Convert user query into embedding and find similar documents using vector similarity.
•	7. Response Generation: GPT-4o uses retrieved chunks as context to generate a grounded answer.
•	8. Response includes document source, page, and snippet.
# RAG_CHATBOT
