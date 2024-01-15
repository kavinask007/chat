from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import VitsModel, AutoTokenizer
import torch

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
DEVICE_TYPE="cpu"
model_kwargs = {"device": DEVICE_TYPE}
st.set_page_config(
    page_title="ChatBot",
    page_icon="🫡"
)
def convert_bytes_to_gb(size_in_bytes):
    return round(size_in_bytes / (1024 ** 3),2)

if "model_list" not in st.session_state:
    import os
    for root,_,files in os.walk('models'):
        st.session_state.model_list=files
        st.session_state.model_map={}
        for file in files:
            st.session_state.model_map[file]=convert_bytes_to_gb(os.path.getsize(os.path.join(root, file)))

if "audio_model" not in st.session_state: 
    st.session_state.audio_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    st.session_state.audio_tokenizer = tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

selected_model = st.sidebar.selectbox("Choose the model", st.session_state.model_list)
is_RAG = st.sidebar.selectbox("RAG Enabled 🤔", [False,True])
is_audio=st.sidebar.selectbox("Audio after completion🔉",[False,True])
if is_audio:
    audio_model=st.sidebar.selectbox("Audio Model🎚️",["facebook/mms-tts-eng"])
system_prompt=st.sidebar.text_area("System Prompt", value="You're a helpful AI assistant")
if "selected_model" not in st.session_state or st.session_state.selected_model_name!=selected_model:
    with st.spinner(f"### Loading -> { selected_model }, Size -> "+str(st.session_state.model_map[selected_model])+"GB",):
        st.session_state.selected_model_name=selected_model
        st.session_state.selected_model=LlamaCpp(
        model_path="/home/ubuntu/ai/models/"+selected_model,
        temperature=0.75,
        max_tokens=2000,
        top_p=1,
        n_ctx=3000,
        n_batch=10
    )
if is_RAG and "db" not in st.session_state:
    with st.spinner("### Loading documents for rag .. "):
        st.session_state.bge_large_embed = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs)
        st.session_state.db = FAISS.load_local("fiass_index", st.session_state.bge_large_embed)

base_prompt = """
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
st.title("Chat")    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input(""):
    # Add user message to chat history
    formatted_chat="###### "+st.session_state.selected_model_name+"\n"+prompt
    st.session_state.messages.append({"role": "user", "content": formatted_chat})
    # Display user message in chat message container
    
    with st.chat_message("user",avatar="👨"):
        st.markdown(formatted_chat)
        if is_RAG:
            context=st.session_state.db.similarity_search_with_score(prompt,k=1)[0][0]
            st.write([context.page_content])

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="💡"):
        message_placeholder = st.empty()
        full_response = ""
        if is_RAG:
            prompt =f""" ```{context.page_content}```.{prompt}"""
        _prompt = base_prompt.format(
            system_message=system_prompt,
            prompt=prompt,
        )
        is_im=False
        is_=False
        for chunk in st.session_state.selected_model.stream(_prompt):
            if chunk=="im":is_im=True
            if chunk=="-":is_=True
            if "end" in chunk and is_im and is_:
                break
            full_response += chunk
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        if is_audio:
            with st.spinner("audio loading .... "):
                inputs = st.session_state.audio_tokenizer(full_response, return_tensors="pt")
                with torch.no_grad():
                    output = st.session_state.audio_model(**inputs).waveform
                st.audio(output[0].numpy(),sample_rate=st.session_state.audio_model.config.sampling_rate)
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
