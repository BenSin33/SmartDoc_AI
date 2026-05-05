import os
import tempfile
import datetime
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(uploaded_file, chunk_size, chunk_overlap):
    """
    Hàm nhận file upload từ Streamlit, đọc nội dung và chia nhỏ thành các chunks.
    Hỗ trợ cả PDF và DOCX.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Tạo file tạm thời để LangChain Loaders có thể đọc
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Bước 1: Document Loader
        if file_extension == 'pdf':
            loader = PDFPlumberLoader(tmp_file_path)
        elif file_extension == 'docx':
            loader = Docx2txtLoader(tmp_file_path) 
        else:
            st.error("Định dạng file không được hỗ trợ. Vui lòng tải lên PDF hoặc DOCX.")
            return None
        
        docs = loader.load()
        
        # Nếu file rỗng hoặc không có text
        if not docs:
            st.warning("Không tìm thấy nội dung văn bản nào trong tài liệu này.")
            return None
        
        # Câu 8.2.8: Multi-documents RAG với meatadata filtering 
        uploaded_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fileName = uploaded_file.name

        for doc in docs:
            doc.metadata["source_file"] = fileName
            doc.metadata["uploaded_date"] = uploaded_date
            doc.metadata["file_type"] = file_extension

        # Bước 2: Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = text_splitter.split_documents(docs)

        for chunk in chunks:
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = chunk.metadata.get("page_number", "N/A")
        return chunks

    except Exception as e:
        # Bắt lỗi và hiển thị rõ ràng lên Streamlit để dễ debug
        st.error(f"Đã xảy ra lỗi khi đọc file {file_extension.upper()}: {str(e)}")

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)