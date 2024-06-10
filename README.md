# Groq Book Chat Streamlit App

## Overview

The Groq Book Chat Streamlit App is a web application designed to assist users in querying and extracting relevant information from uploaded book texts using advanced language models. It leverages the capabilities of Groq API, LangChain, and Sentence Transformer embeddings to provide an interactive and insightful chat experience. Users can upload book texts, which are then processed and split into manageable chunks for efficient querying. The app features an interactive chat interface where users can select from multiple advanced language models to ask questions and receive relevant responses from the text.

## Setup and Usage

To run this application:
- create your own repository of the project
- create a virtual environment: `python -m venv myenv`
- activate the virtual environment: `source myenv/bin/activate`
- ensure you have Python installed and set up the required libraries listed in `requirements.txt` using `pip install -r requirements.txt`.
-  Additionally, set the `GROQ_API_KEY` as an environment variable to authenticate with the Groq API.
-  and run `streamlit run book.py`
-  After setting up, launch the Streamlit app, upload your book text, and start querying. The app supports dynamic model selection and token management, providing a seamless user experience for extracting information from large text documents.

## Demo video
https://youtu.be/mDIn5UKb_mE
