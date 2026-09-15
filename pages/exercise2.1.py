import os
import re
import datetime
import chromadb
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.title("Exercise 2.1 - ChromaDB Ingestion")

# Define target directories
output_dir = "chunks"
os.makedirs(output_dir, exist_ok=True)

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="chromadb_store")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

collection = chroma_client.get_or_create_collection(
    name="document_chunks",
    embedding_function=openai_ef
)

if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []

# Helper function for natural numerical sorting
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# ----------------------------------------------------
# 1. View Previously Saved Chunks (Sorted Dropdown)
# ----------------------------------------------------
st.write("### View Previously Saved Chunks")

saved_files = sorted(
    [f for f in os.listdir(output_dir) if f.endswith(".txt")],
    key=natural_sort_key
)

if saved_files:
    selected_file = st.selectbox("Select a chunk to view:", saved_files)
    if selected_file:
        file_path = os.path.join(output_dir, selected_file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.code(content, language="text")
else:
    st.info("No saved chunks found in the project directory yet.")

st.write("---")

# ----------------------------------------------------
# 2. Document Upload & Chunk Processing
# ----------------------------------------------------
st.write("### Upload Document and Save to ChromaDB")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        bytes_data = uploaded_file.getvalue()
        text = bytes_data.decode("utf-8")

    st.write("#### Document Preview")
    st.write(text[:500] + "..." if len(text) > 500 else text)

    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=150, step=10)

    if st.button("Generate & Save Chunks"):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        batch_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare lists for ChromaDB batch insertion
        ids = []
        documents = []
        metadatas = []

        for idx, chunk in enumerate(chunks, start=1):
            file_name = f"batch_{batch_prefix}_chunk_{idx}.txt"
            file_path = os.path.join(output_dir, file_name)
            
            # Save txt file locally for dropdown visualization
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk)

            ids.append(file_name)
            documents.append(chunk)
            metadatas.append({"batch_id": batch_prefix, "chunk_index": idx})

        # Save to ChromaDB
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        st.session_state.latest_chunks = [
            (f"batch_{batch_prefix}_chunk_{idx}.txt", chunk) 
            for idx, chunk in enumerate(chunks, start=1)
        ]

        st.success(f"Successfully processed {len(chunks)} chunks and indexed them in ChromaDB!")
        st.rerun()

# ----------------------------------------------------
# 3. Display Newly Generated Chunks Below
# ----------------------------------------------------
if st.session_state.latest_chunks:
    st.write("---")
    st.write("### Newly Generated Chunks Display")
    
    first_file_name = st.session_state.latest_chunks[0][0]
    st.info(f"🚀 **New Batch Starting Marker:** First new chunk saved as `{first_file_name}`")

    for file_name, chunk in st.session_state.latest_chunks:
        with st.expander(f"📌 {file_name} (Length: {len(chunk)} characters)"):
            st.code(chunk, language="text")