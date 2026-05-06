import os
import tempfile
import datetime
import streamlit as st

from langchain_community.document_loaders import (
    PDFPlumberLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embedding + Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 1. PROCESS DOCUMENT → RETURN CHUNKS
def process_document(uploaded_file, chunk_size=800, chunk_overlap=60):
    """
    Xử lý tài liệu:
    - Hỗ trợ PDF / DOCX
    - Chunking
    - KHÔNG embed ở đây
    """

    file_extension = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # LOAD DOCUMENT
        if file_extension == "pdf":
            loader = PDFPlumberLoader(tmp_file_path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(tmp_file_path)
        else:
            st.error("Chỉ hỗ trợ PDF và DOCX")
            return None

        docs = loader.load()

        if not docs:
            st.warning("Không có nội dung trong file")
            return None

        # METADATA
        uploaded_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name = uploaded_file.name

        # CHUNKING
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                " ",
                ""
            ]
        )

        chunks = text_splitter.split_documents(docs)

        # CLEAN + METADATA
        for chunk in chunks:
            chunk.page_content = chunk.page_content.strip()

            chunk.metadata["source_file"] = file_name
            chunk.metadata["uploaded_date"] = uploaded_date
            chunk.metadata["file_type"] = file_extension

            if "page" not in chunk.metadata:
                chunk.metadata["page"] = chunk.metadata.get("page_number", "N/A")

        return chunks

    except Exception as e:
        st.error(f"Lỗi xử lý file: {str(e)}")
        return None

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# CREATE VECTOR STORE (EMBED 1 LẦN DUY NHẤT)
def create_vector_store(chunks):
    """
    Tạo embedding + FAISS từ chunks
    """

    if not chunks:
        return None

    try:
        with st.spinner("Đang tạo embedding..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        with st.spinner("Đang tạo vector database (FAISS)..."):
            vectorstore = FAISS.from_documents(chunks, embeddings)

        return vectorstore

    except Exception as e:
        st.error(f"Lỗi tạo vector store: {str(e)}")
        return None
