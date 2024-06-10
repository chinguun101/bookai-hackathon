import os
import streamlit as st
from groq import Groq

st.set_page_config(page_icon="📖 ", layout="wide", page_title="Book Assistant AI")

# Environment variables
os.environ['GROQ_API_KEY'] = 'YOUR_API_KEY'  # Insert API key
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)
model = "mixtral-8x7b-32768"
