import os
import chromadb
import streamlit as st
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("Exercise 2.4 - Advanced RAG with ChromaDB")

client = OpenAI()

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="chromadb_store")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

# Connect to existing ChromaDB collection
try:
    collection = chroma_client.get_collection(
        name="document_chunks",
        embedding_function=openai_ef
    )
    db_ready = True
except Exception:
    db_ready = False

if not db_ready or collection.count() == 0:
    st.warning("No document collection found in ChromaDB. Please process and save chunks in Exercise 2.1 first.")
else:
    st.success(f"ChromaDB connected successfully. Total indexed chunks: {collection.count()}")

    user_query = st.text_input("Ask a question about your uploaded document:")

    if st.button("Get Answer via ChromaDB"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving from ChromaDB and generating answer..."):
                # Query ChromaDB for Top-4 relevant chunks
                results = collection.query(
                    query_texts=[user_query],
                    n_results=min(4, collection.count())
                )

                retrieved_chunks = results["documents"][0]
                retrieved_ids = results["ids"][0]
                retrieved_context = "\n\n--- Chunk ---\n".join(retrieved_chunks)

                # LLM Completion
                system_prompt = (
                    "You are a helpful assistant for document analysis. "
                    "Analyze the provided context directly to answer the question. "
                    "Treat introductory lead-ins like 'MPIA Article 5 provides:' followed by quoted or italicized text as the valid text for that article. "
                    "Provide answers directly without unnecessary disclaimers."
                )
                user_prompt = f"Context:\n{retrieved_context}\n\nQuestion: {user_query}"

                chat_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                answer = chat_response.choices[0].message.content

            # Display retrieved contexts
            st.subheader("ChromaDB Retrieved Chunks")
            for idx, (doc_id, doc_text) in enumerate(zip(retrieved_ids, retrieved_chunks), start=1):
                with st.expander(f"📌 {doc_id} (Rank #{idx})"):
                    st.write(doc_text)

            st.subheader("Answer")
            st.write(answer)