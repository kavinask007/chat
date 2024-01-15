import streamlit as st
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader, UnstructuredPDFLoader,PDFMinerLoader, TextLoader, UnstructuredExcelLoader,UnstructuredWordDocumentLoader
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
st.write("Upload a document to chat about it")
# c=st.container()
# for i in range(10):
#     colss=st.columns(3)
#     with colss[1]:
#         st.markdown("Assume some data hereasdfffff asdf asdadfasd  as dfa sdfa sdf asf as dfas dfa sdfa dsf")
#         tmp=st.columns(4)
#         tmp[0].button("👍",key=i)
#         tmp[1].button("👎",key="down"+str(i))
#         st.divider()
# Upload document
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
    st.write("### **Success!** Your data parsing completed.")
    if st.button("upload to db?"):
        st.write("uploading to db")
        db = FAISS.from_documents([data], embedding=bge_large_embed)
        db.save_local("fiass_index")
        st.write("uploaded 👍👍")   
    st.write({"data":data.page_content})
    # Read and display the content of the uploaded document
    if uploaded_file.type == "text/plain":
        text = uploaded_file.read()
        st.subheader("Content of the Document")
        st.write(text)
# name=st.text_input("name")
# if name:
#     with st.spinner("please wait ..."):
#         time.sleep(5)
#         st.write([1,2,3,4]) 
# 
# 
#  
