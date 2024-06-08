import streamlit as st
from typing import Generator
from groq import Groq
import os
from io import StringIO
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity



st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Groq Goes Brrrrrrrr...")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

##
os.environ['GROQ_API_KEY'] = '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'

groq_api_key = os.getenv('GROQ_API_KEY')

client = Groq(api_key=groq_api_key)

model = "mixtral-8x7b-32768"
##
icon("🏎️")

st.subheader("Groq Book Chat Streamlit App", divider="rainbow", anchor=False)

#client = Groq(
#    api_key=st.secrets["GROQ_API_KEY"],
#)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
    
    
## file upload    
string_data = None
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)
    #print(stringio)

    #To read file as string:
    string_data = stringio.read()
    #st.write(string_data)
book_text = string_data 


if book_text is not None:
    # Define the chunk size and overlap for text splitting
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
        chunk_text = tokenizer.decode(chunk[0], skip_special_tokens=True)  # Assuming [0] is the index of the tensor
        header = f"Chunk {idx+1}\n\n"
        documents.append(Document(page_content=header + chunk_text, metadata={"source": "book"}))

    print(len(documents))
    docsearch = Chroma.from_documents(documents, embedding_function)


    

    # def book_chat_completion(client, model, user_question, relevant_excerpts):
    #     chat_completion = client.chat.completions.create(
    #         messages=[
    #             # {
    #             #     "role": m["system"],
    #             #     "content": m["You are a book expert. Given the user's question and relevant excerpts from a book, answer the question by including direct quotes from the book."]
    #             # },
    #             {
    #                 "role": m["user"],
    #                 "content": m["User Question: " + user_question + "\n\nRelevant Excerpt(s):\n\n" + relevant_excerpts],
    #             }
    #             for m in st.session_state.messages

    #         ],
    #         model=model,
    #         stream = True
    #     )
        
            #     chat_completion = client.chat.completions.create(
    #         model=model_option,
    #         messages=[
    #             {
    #                 "role": m["role"],
    #                 "content": m["content"]
    #             }
    #             for m in st.session_state.messages
    #         ],
    #         max_tokens=max_tokens,
    #         stream=True
    #     )


        # response = chat_completion.choices[0].message.content
        # return response

    # if user_question := st.text_input("Enter your prompt here..."):
    #     relevant_docs = docsearch.similarity_search(user_question)

    #     relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevant_docs[:3]])        
        
    #     st.session_state.messages.append({"role": "user", "content": user_question})

    #     with st.chat_message("user", avatar='👨‍💻'):
    #         st.markdown(user_question)

    #     # Fetch response from Groq API
    #     try:
    #         # Use the generator function with st.write_stream
    #         with st.chat_message("assistant", avatar="🤖"):
    #             chat_responses_generator = book_chat_completion(client, model, user_question, relevant_excerpts)
    #             full_response = st.write_stream(chat_responses_generator)
    #     except Exception as e:
    #         st.error(e, icon="🚨")

    #     # Append the full response to session_state.messages
    #     if isinstance(full_response, str):
    #         st.session_state.messages.append(
    #             {"role": "assistant", "content": full_response})
    #     else:
    #         # Handle the case where full_response is not a string
    #         combined_response = "\n".join(str(item) for item in full_response)
    #         st.session_state.messages.append(
                
    #             {"role": "assistant", "content": combined_response})
        
    # user_question = st.chat_input("Enter your prompt here...")
    # relevant_docs = docsearch.similarity_search(user_question)

    # relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevant_docs[:3]])
    # response = book_chat_completion(client, model, user_question, relevant_excerpts)
    
        # st.write(full_response)
        
        
        

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
        index=4  # Default to mixtral
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
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
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
    
    print(len(relevant_docs))

    relevant_excerpts = '\n\n------------------------------------------------------\n\n'.join([doc.page_content for doc in relevant_docs[:10]])        
    
    prompt = "User Question: " + prompt + "\n\nRelevant Excerpt(s):\n\n" + relevant_excerpts
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)
    # Fetch response from Groq API

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a book expert. Given the user's question and relevant excerpts from a book, answer the question by including direct quotes from the book."
                },
                *[
                    {
                        "role": m["role"],
                        "content": m["content"]
                    } for m in st.session_state.messages
                ]
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
# if prompt := st.chat_input("Enter your prompt here..."):
    # st.session_state.messages.append({"role": "user", "content": prompt})

    # with st.chat_message("user", avatar='👨‍💻'):
    #     st.markdown(prompt)

    # # Fetch response from Groq API
    # try:
    #     chat_completion = client.chat.completions.create(
    #         model=model_option,
    #         messages=[
    #             {
    #                 "role": m["role"],
    #                 "content": m["content"]
    #             }
    #             for m in st.session_state.messages
    #         ],
    #         max_tokens=max_tokens,
    #         stream=True
    #     )

    #     # Use the generator function with st.write_stream
    #     with st.chat_message("assistant", avatar="🤖"):
    #         chat_responses_generator = generate_chat_responses(chat_completion)
    #         full_response = st.write_stream(chat_responses_generator)
    # except Exception as e:
    #     st.error(e, icon="🚨")

    # # Append the full response to session_state.messages
    # if isinstance(full_response, str):
    #     st.session_state.messages.append(
    #         {"role": "assistant", "content": full_response})
    # else:
    #     # Handle the case where full_response is not a string
    #     combined_response = "\n".join(str(item) for item in full_response)
    #     st.session_state.messages.append(
            
    #         {"role": "assistant", "content": combined_response})
        
        