import os
import chromadb
import streamlit as st
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

st.title("Exercise 2.3 - Document Q&A (ChromaDB)")

client = OpenAI()

# 1. 连接到本地 ChromaDB 数据库
chroma_client = chromadb.PersistentClient(path="chromadb_store")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

# 读取已有集合
collection = chroma_client.get_collection(
    name="document_chunks",
    embedding_function=openai_ef
)

user_query = st.text_input("Ask a question about your uploaded document:")

if st.button("Get Answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching with ChromaDB and generating answer..."):
            # 2. 从 Chroma 中检索最相似的 Top-4 切片（自动处理 Embedding 生成与距离计算）
            results = collection.query(
                query_texts=[user_query],
                n_results=4
            )

            # 提取检索出来的文本列表
            retrieved_chunks = results["documents"][0]
            retrieved_context = "\n\n--- Chunk ---\n".join(retrieved_chunks)

            # 3. 提交给 LLM 答题
            system_prompt = (
                "You are a helpful assistant for document analysis. "
                "Analyze the provided context directly to answer the question. "
                "Treat lead-ins like 'MPIA Article 5 provides:' followed by text as the valid text for that article. "
                "Do not add disclaimers about missing context if the information is presented."
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

        # 展示检索到的 Chunk 内容
        st.subheader("Retrieved Context Chunks")
        for idx, doc in enumerate(retrieved_chunks, start=1):
            with st.expander(f"📌 Retrieved Chunk #{idx}"):
                st.write(doc)

        st.subheader("Answer")
        st.write(answer)
        