import logging
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def generate_answer(user_input: str, retriever):
    # 1. Lấy context từ VectorDB
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Xây dựng logic nhận diện ngôn ngữ
    vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
    is_vietnamese = any(char in user_input.lower() for char in vietnamese_chars)
    
    if is_vietnamese:
        prompt_text = """Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.
Nếu bạn không biết, chỉ cần nói là bạn không biết.
Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời:"""
    else:
        prompt_text = """Use the following context to answer the question.
If you don't know the answer, just say you don't know.
Keep answer concise (3-4 sentences).

Context: {context}

Question: {question}

Answer:"""
        
    prompt = PromptTemplate.from_template(prompt_text)
    
    # 3. Khởi tạo LLM và chạy chuỗi
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"context": context, "question": user_input}), docs

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
# UI Streamlit
# =========================================================
st.set_page_config(page_title="SmartDoc AI", page_icon="📄")
st.title("📄 SmartDoc AI - RAG & Chat")

# Lưu trữ retriever và lịch sử chat trong session_state (ngăn chặn reset khi app chạy lại)
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar để quản lý tải lên
with st.sidebar:
    st.header("🗂️ Quản lý Tài liệu")
    uploaded_file = st.file_uploader("Tải lên tài liệu (PDF, DOCX)", type=["pdf", "docx"])
    
    if uploaded_file is not None:
        if st.button("Xử lý tài liệu"):
            with st.spinner("Đang băm nhỏ dữ liệu và nhúng Vector..."):
                document_chunks = process_document(uploaded_file)
                if document_chunks:
                    engine = VectorEngine()
                    st.session_state.retriever = engine.create_and_get_retriever(document_chunks)
                    st.success(f"Khởi tạo thành công! ({len(document_chunks)} chunks).")

# Phần hiển thị Log Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "docs" in message and message["docs"]:
            with st.expander("📄 Xem nguồn trích xuất"):
                for i, d in enumerate(message["docs"]):
                    st.write(f"**Trích đoạn {i+1}:** {d.page_content}")

# Input Box cho User
if st.session_state.retriever:
    if query := st.chat_input("Hãy đặt câu hỏi về tài liệu của bạn..."):
        # Hiển thị câu hỏi của User
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Xử lý và hiển thị câu trả lời của AI
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và sinh câu trả lời..."):
                try:
                    response_text, docs = generate_answer(query, st.session_state.retriever)
                    st.markdown(response_text)
                    with st.expander("📄 Xem nguồn trích xuất"):
                        for i, d in enumerate(docs):
                            st.write(f"**Trích đoạn {i+1}:** {d.page_content}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text, 
                        "docs": docs
                    })
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
else:
    st.info("👈 Vui lòng tải lên và xử lý tài liệu ở thanh bên trái (Sidebar) để bắt đầu.")

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