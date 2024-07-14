import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
DEVICE_TYPE="cpu"
model_kwargs = {"device": DEVICE_TYPE}
k = st.sidebar.selectbox("Top K docs", [1,2,5,10])
with st.spinner("### Loading documents for rag .. "):
    st.session_state.bge_large_embed = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs)
    st.session_state.db = FAISS.load_local("fiass_index", st.session_state.bge_large_embed)
    if prompt := st.chat_input(""):
        context=st.session_state.db.similarity_search_with_score(prompt,k=k)
        for i in context:
            st.markdown(f"* Distance : {i[1]}")
            st.write([i[0].page_content])
            st.divider()