import streamlit as st
from src.Chat import retrieve_similar_chunks, ask_gpt
from src.document_processor import process_document_path
from src.vector_store import store_embeddings_in_chroma
import os

# ######################## Page Config #####################################
st.set_page_config(page_title="GenAI Chatbot", layout="wide")

####################### Session State Init #######################################
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("greeted", False)

######################## Sidebar: Upload ################################
st.sidebar.title("Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a document (PDF, DOCX, PPTX)",
    type=["pdf", "docx", "pptx"]
)

if uploaded_file:
    os.makedirs("data", exist_ok=True)  # ensure folder exists
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    chunks = process_document_path(file_path)
    store_embeddings_in_chroma(chunks)
    st.sidebar.success("✅ Document uploaded and processed!")

################################ Sidebar: Actions ###################################- #
if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.greeted = False
    st.experimental_rerun()

if st.sidebar.button("🚪 Exit App"):
    st.stop()

########################## Main UI #########################
st.title("GenAI Chatbot")

# Greeting Step
if not st.session_state.greeted:
    greeting = st.text_input(
        "Say hi to begin...",
        placeholder="Type 'Hi' or 'Hello' to start chatting"
    )
    if greeting and greeting.strip().lower() in ["hi", "hello"]:
        st.session_state.greeted = True
        st.success("Hi there! Please upload a document and start chatting.")
else:
    # Chat Interface
    query = st.text_input("Ask a question about your document")

    if query:
        with st.spinner("Searching and generating an answer..."):
            docs = retrieve_similar_chunks(query)
            result = ask_gpt(query, docs)

            # Display Answer
            st.markdown("###  Answer:")
            st.success(result["answer"])

            # Save to history
            st.session_state.chat_history.append(
                {"question": query, "answer": result["answer"], "references": result.get("references", [])}
            )

            # References
            if result.get("references"):
                with st.expander("References"):
                    for i, ref in enumerate(result["references"], start=1):
                        st.markdown(f"**Reference {i}:** [{ref}]")

    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.write(f"**You:** {chat['question']}")
            st.write(f"**Bot:** {chat['answer']}")
