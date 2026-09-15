import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("Exercise 2.1")

# 设定保存目录
output_dir = "chunks"

# ----------------------------------------------------
# 1. 历史 Chunk 文件查看（下拉菜单）
# ----------------------------------------------------
st.write("### 查看已保存的 Chunk 文件 (Saved Chunks)")

if os.path.exists(output_dir):
    # 获取文件夹内所有的 .txt 文件并排序
    chunk_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".txt")])
    
    if chunk_files:
        # 下拉菜单供用户选择文件
        selected_file = st.selectbox("选择要查看的历史 Chunk 文件：", chunk_files)
        
        # 读取并展示选中的 Chunk 内容
        if selected_file:
            file_path = os.path.join(output_dir, selected_file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            st.code(content, language="text")
    else:
        st.info("目前没有找到已保存的 Chunk 文件。")
else:
    st.info("尚未创建 chunks 文件夹，请先在下方上传并生成 Chunk。")

st.write("---")

# ----------------------------------------------------
# 2. 上传文件与生成新 Chunk
# ----------------------------------------------------
st.write("### 上传新文件并生成 Chunk")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        bytes_data = uploaded_file.getvalue()
        text = bytes_data.decode("utf-8")
    
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10)
    
    if st.button("生成并保存到项目"):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, chunk in enumerate(chunks):
            file_name = f"chunk_{idx + 1}.txt"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk)
                
        st.success(f"成功保存 {len(chunks)} 个 Chunk！重新选择顶部的下拉菜单即可查看。")
        st.rerun()  # 自动刷新页面以立刻更新下拉菜单列表