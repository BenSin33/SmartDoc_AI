import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(uploaded_file):
    """
    Hàm nhận file upload từ Streamlit, đọc nội dung và chia nhỏ thành các chunks.
    Hỗ trợ cả PDF và DOCX.
    """
    # Tạo file tạm thời để LangChain Loaders có thể đọc
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Bước 1: Document Loader (Hỗ trợ PDF và DOCX)
        if file_extension == 'pdf':
            loader = PDFPlumberLoader(tmp_file_path)
        elif file_extension == 'docx':
            # Hoàn thành yêu cầu phát triển số 1: Hỗ trợ file DOCX
            loader = Docx2txtLoader(tmp_file_path) # 
        else:
            st.error("Định dạng file không được hỗ trợ. Vui lòng tải lên PDF hoặc DOCX.")
            return None

        docs = loader.load()

        # Bước 2: Text Splitter (Băm nhỏ văn bản)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # Độ dài tối đa mỗi chunk 
            chunk_overlap=100   # Ký tự trùng lặp giữa các chunks liên tiếp 
        )
        
        chunks = text_splitter.split_documents(docs)
        return chunks

    finally:
        # Dọn file rác
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

# UI Streamlit
st.title("📄 SmartDoc AI - Document Processor")

uploaded_file = st.file_uploader("Tải lên tài liệu của bạn (PDF, DOCX)", type=["pdf", "docx"])

if uploaded_file is not None:
    with st.spinner("Đang xử lý tài liệu..."):
        document_chunks = process_document(uploaded_file)
        
        if document_chunks:
            st.success(f"Đã xử lý thành công! Chia thành {len(document_chunks)} đoạn (chunks).")
            
            # Hiển thị thử nội dung của chunk đầu tiên để kiểm tra
            with st.expander("Xem trước Chunk đầu tiên"):
                st.write(document_chunks[0].page_content)

            with st.expander("Xem trước Chunk thứ hai"):
                st.write(document_chunks[1].page_content)
            
            with st.expander("Xem trước Chunk thứ ba"):
                st.write(document_chunks[2].page_content)