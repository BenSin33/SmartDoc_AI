import logging
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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

# cấu hình logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):

        logger.info(f"Initializing embeddings with model: {model_name}...")

        try:

            self.embedder = HuggingFaceEmbeddings(
                model_name = model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            logger.info("Embeddings initialized successfully!")

        except Exception as e:

            logger.error(f"Error downloading Hugging Face embeddings: {e}")
            raise

        self.vector_store = None
    
    def create_and_get_retriever(self, documents):

        # Initialize Vector stroe and return retriever
        logger.info(f"Creating vector store with {len(documents)} documents chunks into FAISS...")

        try:
            self.vector_store = FAISS.from_documents(documents, self.embedder)
            logger.info("FAISS Vector store created successfully!")

            retriever = self.vector_store.as_retriever(
                search_type = "similarity",
                search_kwargs = {"k":3}
            )
            return retriever
        
        except Exception as e:
            logger.error(f"Error creating FAISS vector store: {e}")
            raise

    def save_local_index(self, folder_path: str ="faiss_index"):

        if self.vector_store:
            self.vector_store.save_local(folder_path)
            logger.info(f"FAISS index saved locally at: {folder_path}")

        else:
            logger.warning(f"No vector store to save. Please create the vector store first.")

    def load_local_index(self, folder_path: str = "faiss_index"):

        try:
            self.vector_store = FAISS.load_local(
                folder_path,
                self.embedder,
                allow_dangerous_deserialization = True
            )

            logger.info(f"FAISS index loaded successfully from: {folder_path}")
            return self.vector_store.as_retriever(search_kwargs={"k": 3})

        except Exception as e:
            logger.error(f"Error loading FAISS index from {folder_path}: {e}")
            return None

# =========================================================
# PHẦN TEST CHẠY THỬ MÔ HÌNH (Viết sát lề trái)
# =========================================================
if __name__ == "__main__":
    from langchain_core.documents import Document
    
    print("="*50)
    print("BẮT ĐẦU TEST VECTOR ENGINE (PHẦN 3.3.3 & 3.3.4)")
    print("="*50)
    
    # 1. Khởi tạo engine
    engine = VectorEngine()
    
    # 2. Tạo một vài document giả lập để test
    print("\n--- Đang tạo dữ liệu mẫu ---")
    sample_docs = [
        Document(page_content="Streamlit là một thư viện Python giúp tạo giao diện web app cho machine learning rất nhanh chóng.", metadata={"source": "doc1"}),
        Document(page_content="LangChain là một framework giúp kết nối các mô hình ngôn ngữ lớn (LLM) với dữ liệu bên ngoài.", metadata={"source": "doc2"}),
        Document(page_content="FAISS là thư viện mã nguồn mở của Facebook dùng để tìm kiếm sự tương đồng vector với tốc độ cao.", metadata={"source": "doc3"})
    ]
    
    # 3. Chạy thử hàm tạo retriever
    print("\n--- Chạy hàm tạo Vector Store ---")
    test_retriever = engine.create_and_get_retriever(sample_docs)
    
    # 4. Kiểm tra kết quả
    if test_retriever:
        print("\n🎉 THÀNH CÔNG! Đã setup xong Retriever.")
        
        # Test thử truy vấn
        query = "Thư viện nào tạo web app?"
        print(f"\n--- Thử tìm kiếm với câu hỏi: '{query}' ---")
        results = test_retriever.invoke(query) # Dùng .invoke() cho bản Langchain mới
        
        for i, doc in enumerate(results):
            print(f"Kết quả {i+1}: {doc.page_content}")
        
        # Test lưu dữ liệu xuống máy
        print("\n--- Lưu Database xuống ổ cứng ---")
        engine.save_local_index("test_faiss_index")
        print("\nHoàn tất bài test! Bạn có thể xem thư mục 'test_faiss_index' vừa được tạo ra.")