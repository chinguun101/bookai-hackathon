import streamlit as st
from typing import Generator
from groq import Groq
import os
import pandas as pd
from io import StringIO
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_icon="📖 ", layout="wide",
                   page_title="Book Assistant AI")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

##
os.environ['GROQ_API_KEY'] = 'YOUR_API_KEY' #Insert API
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)
model = "mixtral-8x7b-32768"
##
icon("🏎️")

st.subheader("Groq Book Chat Streamlit App", divider="rainbow", anchor=False)
# Initialize chat history and selected model

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

## file upload    
string_data = None
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
book_text = string_data 

if book_text is not None:
    chunk_size = 1000
    chunk_overlap = 100
    # Split the book text into chunks

    def split_text(text, chunk_size, chunk_overlap):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks
    
    text_chunks = split_text(book_text, chunk_size, chunk_overlap)
    # Tokenize each chunk
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    tokenized_chunks = [tokenizer.encode(chunk, return_tensors='pt', max_length=512, truncation=True) for chunk in text_chunks]

    print(f"Number of chunks: {len(tokenized_chunks)}")

    chunk_embeddings = []

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    for chunk in text_chunks:
        chunk_embeddings.append(embedding_function.embed_query(chunk))
        
    documents = []
    for idx, chunk in enumerate(tokenized_chunks):
        chunk_text = tokenizer.decode(chunk[0], skip_special_tokens=True) 
        header = f"Chunk {idx+1}\n\n"
        documents.append(Document(page_content=header + chunk_text, metadata={"source": "book"}))

    print(len(documents))
    docsearch = Chroma.from_documents(documents, embedding_function)
else:
    response = "Please Upload a File"

# Define model details      
models = {
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
}
# Layout for model selection and max_tokens slider

col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=4 
    )
# Detect model change and clear chat history if model has changed

if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    # Adjust max_tokens slider dynamically based on the selected model

    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512, 
        max_value=max_tokens_range,
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""

    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

if prompt := st.chat_input("Enter your prompt here..."):
    relevant_docs = docsearch.similarity_search(prompt, k=20)
    
    unique_excerpts = list({doc.page_content for doc in relevant_docs[:10]})
    relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join(unique_excerpts)
    
    combined_prompt = f"User Question: {prompt}\n\nRelevant Excerpt(s):\n\n{relevant_excerpts}"
    st.session_state.messages.append({"role": "user", "content": combined_prompt})
    
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(combined_prompt)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a book assistant. Given the user's question and relevant excerpts from a book, answer the question by including direct quotes from the book."
                },
                *[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    } for m in st.session_state.messages
                ],
                {
                    "role": "user",
                    "content": combined_prompt
                }
            ],
            model=model_option,
            stream=True
        )

        # Use the generator function with st.write_stream

        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
            
    except Exception as e:
        st.error(e, icon="🚨")
     # Append the full response to session_state.messages

    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
