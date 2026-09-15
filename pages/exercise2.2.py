import os
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

st.title("Exercise 2.2 ")

# 1. Allow users to copy and paste 2 different texts
if "text_1" not in st.session_state:
    st.session_state.text_1 = ""

if "text_2" not in st.session_state:
    st.session_state.text_2 = ""

text_1 = st.text_area("Enter Text 1:", value=st.session_state.text_1, key="input_text_1")
text_2 = st.text_area("Enter Text 2:", value=st.session_state.text_2, key="input_text_2")

# Initialize OpenAI Client
client = OpenAI()

# Button to trigger embedding generation and similarity computation
if st.button("Calculate Similarity"):
    if not text_1.strip() or not text_2.strip():
        st.warning("Please enter text into both fields before calculating.")
    else:
        with st.spinner("Generating embeddings..."):
            # 2. Create an embedding for each text input
            response_1 = client.embeddings.create(
                model="text-embedding-3-large",
                input=text_1
            )
            embedding_1 = response_1.data[0].embedding

            response_2 = client.embeddings.create(
                model="text-embedding-3-large",
                input=text_2
            )
            embedding_2 = response_2.data[0].embedding

        # 3. Display the cosine similarities between two texts
        vec1 = np.array(embedding_1)
        vec2 = np.array(embedding_2)

        # Compute cosine similarity: (A · B) / (||A|| * ||B||)
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        st.subheader("Results")
        st.metric(label="Cosine Similarity Score", value=f"{cosine_sim:.4f}")

        # 3. Display the cosine similarities between two texts
        vec1 = np.array(embedding_1)
        vec2 = np.array(embedding_2)

        # Compute cosine similarity
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Directly display the raw number
        st.write(cosine_sim)