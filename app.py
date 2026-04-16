import logging
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

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

#===========================GIAO DIEN=============================================


# st.title("📄 SmartDoc AI - Document Processor")

# uploaded_file = st.file_uploader("Tải lên tài liệu của bạn (PDF, DOCX)", type=["pdf", "docx"])

# if uploaded_file is not None:
#     with st.spinner("Đang xử lý tài liệu..."):
#         document_chunks = process_document(uploaded_file)
        
#         if document_chunks:
#             st.success(f"Đã xử lý thành công! Chia thành {len(document_chunks)} đoạn (chunks).")
            
#             # Hiển thị thử nội dung của chunk đầu tiên để kiểm tra
#             with st.expander("Xem trước Chunk đầu tiên"):
#                 st.write(document_chunks[0].page_content)

#             with st.expander("Xem trước Chunk thứ hai"):
#                 st.write(document_chunks[1].page_content)
            
#             with st.expander("Xem trước Chunk thứ ba"):
#                 st.write(document_chunks[2].page_content)


# ------------------ CẤU HÌNH GIAO DIỆN ------------------
st.set_page_config(
    page_title="SmartDoc AI",
    # page_icon="📄",
    layout="wide"
)

# Custom CSS theo yêu cầu từ file (Section 5.1.1)
st.markdown("""
<style>
    /* Primary Color: #007BFF */
    .stButton > button {
        background-color: #007BFF;
        color: white;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #2C2F33;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }
    /* Main background */
    .main {
        background-color: #F8F9FA;
    }
    /* Text color */
    body, .stMarkdown, .stTextInput > label {
        color: #212529;
    }
    /* Upload button màu amber */
    .upload-btn {
        background-color: #FFC107;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
with st.sidebar:
    
    # Logo Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 12px 0 20px 0; border-bottom: 1px solid #4a4e54; margin-bottom: 20px;">
        <div style="background-color: #007BFF; width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 20px;">🧠</span>
        </div>
        <div>
            <h1 style="font-size: 18px; font-weight: 700; margin: 0;">SmartDoc AI</h1>
            <p style="font-size: 15px; color: #adb5bd; margin: 2px 0 0 0;">Hệ thống xử lý tài liệu AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # st.title("Hướng dẫn sử dụng")

    # st.markdown("""
    # 1. Tải lên tài liệu **PDF** hoặc **DOCX**
    # 2. Hệ thống tự động xử lý & chia chunks
    # 3. Nhập câu hỏi liên quan đến nội dung
    # 4. Nhận câu trả lời chính xác
    # """)

    # st.markdown("---")

    # st.subheader("Cấu hình mô hình")

    # st.info("""
    # - **Embedding**: paraphrase-multilingual-mpnet-base-v2  
    # - **Vector DB**: FAISS  
    # - **LLM**: Qwen2.5:7b (Ollama)  
    # - **Chunk size**: 1000, overlap: 100
    # """)
    
    # Hướng dẫn sử dụng
    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #adb5bd; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
        Hướng dẫn sử dụng
    </h2>
    """, unsafe_allow_html=True)
    
    # Các bước hướng dẫn
    steps = [
        ("1", "Tải lên file PDF cần phân tích"),
        ("2", "Chờ hệ thống xử lý và phân tích"),
        ("3", "Đặt câu hỏi về nội dung tài liệu"),
        ("4", "Nhận câu trả lời thông minh từ AI")
    ]
    
    for num, text in steps:
        st.markdown(f"""
        <div style="display: flex; gap: 12px; padding: 10px; background-color: #3a3e44; border-radius: 8px; margin-bottom: 8px; border: 1px solid #4a4e54;">
            <div style="flex-shrink: 0; width: 24px; height: 24px; background-color: #007BFF; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                <span style="font-size: 11px; font-weight: bold; color: white;">{num}</span>
            </div>
            <p style="font-size: 13px; color: #e9ecef; margin: 0;">{text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cấu hình hệ thống
    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #adb5bd; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
        Cấu hình hệ thống
    </h2>
    """, unsafe_allow_html=True)
    
    configs = [
        ("Model", "Qwen2.5:7b"),
        ("Context Length", "128K tokens"),
        ("Temperature", "0.7"),
        ("Max File Size", "50 MB"),
        ("Chunk Size", "1000"),
        ("Overlap", "100")
    ]
    
    for label, value in configs:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 10px; background-color: #35393e; border-radius: 8px; margin-bottom: 8px;">
            <span style="color: #adb5bd; font-size: 13px;">{label}</span>
            <span style="color: #FFC107; font-family: monospace; font-size: 13px;">{value}</span>
        </div>
        """, unsafe_allow_html=True)
    
    

# ------------------ MAIN AREA ------------------
st.title("SmartDoc AI - Intelligent Document Q&A System")
st.markdown("### Hỏi đáp thông minh với tài liệu của bạn")

# Khởi tạo session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# ------------------ HÀM XỬ LÝ DOCUMENT ------------------
# def process_document(uploaded_file):

#     """Xử lý file PDF hoặc DOCX -> chunks"""
    
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
#         tmp_file.write(uploaded_file.getbuffer())
#         tmp_path = tmp_file.name

#     # Load document
#     if uploaded_file.name.endswith(".pdf"):
#         loader = PDFPlumberLoader(tmp_path)
#     elif uploaded_file.name.endswith(".docx"):
#         loader = Docx2txtLoader(tmp_path)
#     else:
#         st.error("Định dạng không hỗ trợ")
#         return None

#     docs = loader.load()
    
#     # Split text
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=100
#     )
#     chunks = text_splitter.split_documents(docs)
    
#     # Clean temp file
#     os.unlink(tmp_path)
    
#     return chunks

# ------------------ TẠO VECTOR STORE ------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# ------------------ UPLOAD FILE ------------------
uploaded_file = st.file_uploader(
    "📂 Tải lên tài liệu của bạn (PDF, DOCX)",
    type=["pdf", "docx"],
    help="Hỗ trợ PDF và DOCX, tối đa 50MB"
)

if uploaded_file is not None:
    with st.spinner("⏳ Đang xử lý tài liệu..."):
        document_chunks = process_document(uploaded_file)
        
        if document_chunks:
            st.session_state.chunks = document_chunks
            st.success(f"✅ Đã xử lý thành công! Chia thành {len(document_chunks)} đoạn (chunks).")
            
            # Hiển thị xem trước chunks (theo yêu cầu)
            with st.expander("📖 Xem trước Chunk đầu tiên"):
                st.write(document_chunks[0].page_content[:500] + "...")
            
            if len(document_chunks) > 1:
                with st.expander("📖 Xem trước Chunk thứ hai"):
                    st.write(document_chunks[1].page_content[:500] + "...")
            
            if len(document_chunks) > 2:
                with st.expander("📖 Xem trước Chunk thứ ba"):
                    st.write(document_chunks[2].page_content[:500] + "...")
            
            # Tạo vector store
            with st.spinner("🔄 Đang tạo embedding và vector store..."):
                st.session_state.vector_store = create_vector_store(document_chunks)
                st.success("🧠 Vector store sẵn sàng! Bạn có thể đặt câu hỏi bên dưới.")
        else:
            st.error("❌ Không thể xử lý tài liệu. Vui lòng thử lại.")

# ------------------ PHẦN ĐẶT CÂU HỎI ------------------
if st.session_state.vector_store is not None:
    st.markdown("---")
    st.subheader("💬 Đặt câu hỏi về tài liệu")
    
    # Tạo retriever
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Kết nối LLM (Ollama)
    try:
        llm = Ollama(model="qwen2.5:7b", temperature=0.7)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        user_question = st.text_input("🔍 Nhập câu hỏi của bạn:", placeholder="Ví dụ: Nội dung chính của tài liệu này là gì?")
        
        if user_question:
            with st.spinner("🤖 Đang suy nghĩ..."):
                response = qa_chain.invoke({"query": user_question})
                answer = response['result']
                
                # Hiển thị câu trả lời
                st.markdown("### 📝 Câu trả lời:")
                st.success(answer)
                
                # Tùy chọn xem source chunks
                with st.expander("🔗 Xem nguồn tham khảo (chunks liên quan)"):
                    for i, doc in enumerate(response['source_documents']):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content[:400] + "...")
                        st.markdown("---")
    
    except Exception as e:
        st.error(f"⚠️ Lỗi kết nối LLM: {e}\nVui lòng đảm bảo Ollama đang chạy và đã pull model qwen2.5:7b")
else:
    st.info("📌 Vui lòng tải lên tài liệu trước khi đặt câu hỏi.")


#=========================================================================

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