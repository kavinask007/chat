import streamlit as st
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, UnstructuredPDFLoader,PDFMinerLoader, TextLoader, UnstructuredExcelLoader,UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# !apt upgrade -y
# !apt install antiword libreoffice -y  need for word document loader
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredPDFLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlxs": UnstructuredExcelLoader,
    ".doc": UnstructuredWordDocumentLoader,
    ".docx":UnstructuredWordDocumentLoader
}
INGEST_THREADS = os.cpu_count() or 8
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
DEVICE_TYPE="cpu"
model_kwargs = {"device": DEVICE_TYPE}
bge_large_embed = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs)
def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    print(file_path)
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    return loader.load()[0]

st.set_page_config(
    page_title="Upload",
    page_icon="🧠"
)
# App title and description
st.title("Document Upload")
st.write("Upload a document to chat with it - DONOT upload any personal or sensitive information")

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf","docx","doc"])  
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    st.write("File Details:", file_details)
    file_type="."+uploaded_file.type.split("/")[-1]
    loader_class = DOCUMENT_MAP.get(file_type,None)
    if not loader_class:
        st.write("file format not supported")
    temp_file_path = "temp_file.txt"

    # Save the uploaded file to the temporary location
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    loader=loader_class(temp_file_path)
    data=loader.load()[0]
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20,length_function= len , is_separator_regex=False)
    data=text_splitter.split_documents([data])
    st.write("### **Success!** Your data parsing completed.")
    st.write("uploading to db")
    db = FAISS.from_documents(data, embedding=bge_large_embed)
    db.save_local("fiass_index")
    st.write("uploaded 👍👍")   
    st.write([{"data":i.page_content} for i in data ])
