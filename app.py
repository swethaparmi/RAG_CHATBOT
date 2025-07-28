import streamlit as st
from src.Chat import retrieve_similar_chunks, ask_gpt
from src.document_processor import process_document_path
from src.vector_store import store_embeddings_in_chroma


import os

st.set_page_config(page_title="GenAI Chatbot", layout="wide")

############################### Session state for uploaded file and chat messages################################
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar UI
st.sidebar.title("Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, DOCX, PPTX)", type=["pdf", "docx", "pptx"])

if uploaded_file:
    with open(os.path.join("data", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.read())

    # Process & store embeddings
    chunks = process_document_path(os.path.join("data", uploaded_file.name))
    store_embeddings_in_chroma(chunks)
    st.sidebar.success("Document uploaded and processed!")

######################## Clear Chat Button ############################################
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

############################## Exit Button #######################################################
if st.sidebar.button(" Exit App"):
    st.sidebar.info("Please close the browser tab to exit the app.")
    st.stop()

# Main UI
st.title("Chatbot")
st.markdown("Ask questions based on your uploaded document.")

query = st.text_input("Type your question:")

if query:
    with st.spinner("Searching and generating answer..."):
        docs = retrieve_similar_chunks(query)
        result = ask_gpt(query, docs)

        # Display answer
        st.markdown("### Answer:")
        st.success(result["answer"])

 ###############################       # Save to chat history #########################################
        st.session_state.chat_history.append(
            {"question": query, "answer": result["answer"], "references": result.get("references", [])}
        )

        # References (if any)
        if result.get("references"):
            with st.expander("References"):
                for i, ref in enumerate(result["references"], start=1):
                    st.markdown(f"**Reference {i}:**")

                    

############################  to enable chat history #####################################
# for chat in st.session_state.chat_history:
#     st.write("**Prompt :**", chat["question"])
#     st.write("**Answer**", chat["answer"])
