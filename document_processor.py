import os
import tempfile
import datetime
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(uploaded_file, chunk_size, chunk_overlap):
    """
    Hàm xử lý tài liệu giữ nguyên PDFPlumber nhưng đã được tối ưu hóa vòng lặp
    và cấu trúc Metadata để tăng tốc độ xử lý phần mềm.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Tạo file tạm thời (Bắt buộc đối với các Loader mặc định của LangChain)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Bước 1: Khởi tạo Document Loader tương ứng
        if file_extension == 'pdf':
            loader = PDFPlumberLoader(tmp_file_path)
        elif file_extension == 'docx':
            loader = Docx2txtLoader(tmp_file_path) 
        else:
            st.error("Định dạng file không được hỗ trợ. Vui lòng tải lên PDF hoặc DOCX.")
            return None
        
        docs = loader.load()
        
        if not docs:
            st.warning("Không tìm thấy nội dung văn bản nào trong tài liệu này.")
            return None
        
        # Lấy thông tin thời gian và tên file một lần duy nhất ngoài vòng lặp
        uploaded_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_name = uploaded_file.name

        # Bước 2: Khởi tạo Text Splitter trước khi xử lý metadata 
        # (Để tận dụng việc tách trực tiếp từ mảng docs đã nạp sẵn)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        chunks = text_splitter.split_documents(docs)

        # Bước 3: Chuẩn hóa toàn bộ Metadata trên các Chunks (Gộp 2 vòng lặp cũ thành 1)
        for chunk in chunks:
            chunk.metadata["source_file"] = file_name
            chunk.metadata["uploaded_date"] = uploaded_date
            chunk.metadata["file_type"] = file_extension
            # Chuẩn hóa key "page" để tránh lỗi không tìm thấy trang khi hiển thị UI
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = chunk.metadata.get("page_number", "N/A")

        return chunks

    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi đọc file {file_extension.upper()}: {str(e)}")
        return None

    finally:
        # Đảm bảo file tạm luôn được xóa kể cả khi xảy ra lỗi giữa chừng
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)