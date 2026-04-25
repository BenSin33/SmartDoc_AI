from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker# thêm thư viện cho Re-ranking
from langchain_community.cross_encoders import HuggingFaceCrossEncoder # thêm thư viện cho Re-ranking
import datetime
import time
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import logging
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from langchain_community.llms import Ollama
from langchain_classic.prompts import PromptTemplate

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

#===========================FUNCTIONS=========================================

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

#===========================GIAO DIEN=============================================

st.set_page_config(
    page_title="SmartDoc AI",
    # page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
/* ===== 1. GLOBAL ===== */
    .stApp {
        background-color: #F8F9FA !important;
    }

    /* Fix toàn bộ text trong MAIN */
    [data-testid="stMain"],
    [data-testid="stMain"] * {
        color: #212529 !important;
    }

    /* ===== 2. SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #2C2F33 !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* ===== 3. FILE UPLOADER ===== */
    [data-testid='stFileUploader'],
    [data-testid='stFileUploader'] * {
        color: #212529 !important;
    }

    [data-testid="stFileUploaderFileName"] {
        font-weight: bold !important;
    }

    /* ===== 4. BUTTON (FIX TOÀN BỘ) ===== */

    /* Button thường */
    button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    /* Button trong form (Hỏi) */
    div[data-testid="stForm"] button {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
    }

    /* Hover */
    button:hover {
        background-color: #0056b3 !important;
        color: #FFFFFF !important;
    }

    /* ===== 5. INPUT ===== */
    input, textarea, [data-baseweb="input"] input {
        color: #FFFFFF !important;
        background-color: #3a3e44 !important;
        border-color: #555555 !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: #999999 !important;
    }
    
    [data-baseweb="input"] {
        background-color: #3a3e44 !important;
    }

    /* ===== 5.1 SELECTBOX ===== */
    [data-baseweb="select"] {
        background-color: #3a3e44 !important;
    }

    [data-baseweb="select"] button {
        background-color: #3a3e44 !important;
        color: #FFFFFF !important;
        border-color: #555555 !important;
    }

    [data-baseweb="select"] button span {
        color: #FFFFFF !important;
    }

    [data-baseweb="popover"] {
        background-color: #3a3e44 !important;
    }

    [data-baseweb="menu"] {
        background-color: #3a3e44 !important;
    }

    [data-baseweb="menu"] li {
        color: #FFFFFF !important;
    }

    [data-baseweb="menu"] li:hover {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
    }

    [role="option"] {
        background-color: #3a3e44 !important;
        color: #FFFFFF !important;
    }

    [role="option"]:hover {
        background-color: #007BFF !important;
        color: #FFFFFF !important;
    }

    /* ===== 6. MARK ===== */
    mark {
        background-color: #fff3cd;
        color: #212529 !important;
        padding: 2px 4px;
        border-radius: 4px;
    }
    /* Fix vùng drag & drop */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #CED4DA !important;
    }

    /* Text bên trong dropzone */
    [data-testid="stFileUploaderDropzone"] * {
        color: #212529 !important;
    }

    /* Hover cho đẹp */
    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #F1F3F5 !important;
    }
</style>
""", unsafe_allow_html=True)

# CÁC HÀM XỬ LÝ XÓA (DIALOG) 

@st.dialog("Xác nhận xóa lịch sử")
def clear_history_dialog():
    st.write("Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện không?")
    if st.button("Xác nhận xóa", type="primary"):
        st.session_state.chat_history = [] # Xóa UI history
        if "memory" in st.session_state:
            st.session_state.memory.clear() # Xóa memory của LLM
        st.rerun() # Refresh lại giao diện ngay lập tức

@st.dialog("Xác nhận xóa tài liệu")
def clear_vector_store_dialog():
    st.write("Dữ liệu vector và các đoạn văn bản (chunks) sẽ bị xóa hoàn toàn. Bạn sẽ cần upload lại tài liệu mới.")
    if st.button("Xác nhận xóa", type="primary"):
        st.session_state.vector_store = None
        st.session_state.chunks = None
        if "user_question" in st.session_state:
            st.session_state.user_question = ""

# ------------------ SIDEBAR ------------------
with st.sidebar:
    
    # Logo Header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; padding: 12px 0 20px 0; border-bottom: 1px solid #4a4e54; margin-bottom: 20px;">
        <div style="background-color: #007BFF; width: 36px; height: 36px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 20px;">🧠</span>
        </div>
        <div>
            <h1 style="font-size: 18px; font-weight: 700; margin: 0; color: #FFFFFF;">SmartDoc AI</h1>
            <p style="font-size: 15px; color: #FFFFFF; margin: 2px 0 0 0; opacity: 0.8;">Hệ thống xử lý tài liệu AI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    # Hướng dẫn sử dụng
    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #FFFFFF; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
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
                <span style="font-size: 11px; font-weight: bold; color: #FFFFFF;">{num}</span>
            </div>
            <p style="font-size: 13px; color: #FFFFFF; margin: 0;">{text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Cấu hình hệ thống
    # --- CHUNK CONFIG ---
    st.markdown("### ⚙️ Tùy chỉnh Chunk")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        help="Kích thước mỗi đoạn văn bản"
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50,
        help="Số ký tự chồng lấp giữa các chunk"
    )

    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap


    st.markdown(f"""
    <div style="color: #FFC107; font-size: 13px;">
    Chunk Size: {chunk_size}<br>
    Overlap: {chunk_overlap}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #FFFFFF; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
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
            <span style="color: #FFFFFF; font-size: 13px; opacity: 0.9;">{label}</span>
            <span style="color: #FFC107; font-family: monospace; font-size: 13px; font-weight: bold;">{value}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #FFFFFF; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
        Performance Metrics
    </h2>
    """, unsafe_allow_html=True)

    if "metrics" in st.session_state:
        st.markdown(f"""
        <div style="padding: 10px; background-color: #35393e; border-radius: 8px; margin-bottom: 8px;">
            <span style="color: #FFFFFF; font-size: 13px; opacity: 0.9;">Doc Processing:</span>
            <span style="color: #FFC107; font-family: monospace; font-size: 13px; font-weight: bold; float: right;">
                {st.session_state.metrics.get('doc_processing_time', 0):.2f} s
            </span>
        </div>
        <div style="padding: 10px; background-color: #35393e; border-radius: 8px; margin-bottom: 8px;">
            <span style="color: #FFFFFF; font-size: 13px; opacity: 0.9;">Embedding:</span>
            <span style="color: #FFC107; font-family: monospace; font-size: 13px; font-weight: bold; float: right;">
                {st.session_state.metrics.get('embedding_time', 0):.2f} s
            </span>
        </div>
        <div style="padding: 10px; background-color: #35393e; border-radius: 8px; margin-bottom: 8px;">
            <span style="color: #FFFFFF; font-size: 13px; opacity: 0.9;">Q&A Time:</span>
            <span style="color: #FFC107; font-family: monospace; font-size: 13px; font-weight: bold; float: right;">
                {st.session_state.metrics.get('qa_time', 0):.2f} s
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Chưa có dữ liệu metrics.")

    # --- RAG vs CoRAG COMPARISON ---
    st.markdown("---")
    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #FFFFFF; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
        📊 RAG vs CoRAG
    </h2>
    """, unsafe_allow_html=True)
    
    # Khởi tạo nếu chưa có (fallback an toàn)
    if "rag_corag_metrics" not in st.session_state:
        st.session_state.rag_corag_metrics = {
            "rag": {"qa_time": [], "retrieval_count": [], "relevance_scores": []},
            "corag": {"qa_time": [], "retrieval_count": [], "relevance_scores": []}
        }
    
    rag_metrics = st.session_state.rag_corag_metrics["rag"]
    corag_metrics = st.session_state.rag_corag_metrics["corag"]
    
    if len(rag_metrics["qa_time"]) > 0 or len(corag_metrics["qa_time"]) > 0:
        # RAG Stats
        if len(rag_metrics["qa_time"]) > 0:
            avg_rag_time = sum(rag_metrics["qa_time"]) / len(rag_metrics["qa_time"])
            avg_rag_relevance = sum(rag_metrics["relevance_scores"]) / len(rag_metrics["relevance_scores"])
            
            st.markdown(f"""
            <div style="background-color: #1e3a5f; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #007BFF;">
                <b style="color: #007BFF;">📈 RAG</b><br>
                <span style="color: #FFFFFF; font-size: 12px;">Avg Time: <b>{avg_rag_time:.2f}s</b> | Queries: <b>{len(rag_metrics["qa_time"])}</b> | Relevance: <b>{avg_rag_relevance*100:.1f}%</b></span>
            </div>
            """, unsafe_allow_html=True)
        
        # CoRAG Stats
        if len(corag_metrics["qa_time"]) > 0:
            avg_corag_time = sum(corag_metrics["qa_time"]) / len(corag_metrics["qa_time"])
            avg_corag_relevance = sum(corag_metrics["relevance_scores"]) / len(corag_metrics["relevance_scores"])
            avg_retrieval_count = sum(corag_metrics["retrieval_count"]) / len(corag_metrics["retrieval_count"])
            
            st.markdown(f"""
            <div style="background-color: #1f3a1f; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #28a745;">
                <b style="color: #28a745;">🔄 CoRAG</b><br>
                <span style="color: #FFFFFF; font-size: 12px;">Avg Time: <b>{avg_corag_time:.2f}s</b> | Queries: <b>{len(corag_metrics["qa_time"])}</b> | Relevance: <b>{avg_corag_relevance*100:.1f}%</b> | Retries: <b>{avg_retrieval_count:.1f}</b></span>
            </div>
            """, unsafe_allow_html=True)
        
        # Comparison
        if len(rag_metrics["qa_time"]) > 0 and len(corag_metrics["qa_time"]) > 0:
            time_diff = ((avg_corag_time - avg_rag_time) / avg_rag_time) * 100
            relevance_diff = ((avg_corag_relevance - avg_rag_relevance) / max(avg_rag_relevance, 0.01)) * 100
            
            st.markdown(f"""
            <div style="background-color: #3a3a3a; padding: 12px; border-radius: 8px; border-left: 4px solid #FFC107;">
                <b style="color: #FFC107;">⚖️ So sánh</b><br>
                <span style="color: #FFFFFF; font-size: 12px;">Time Diff: <b>{time_diff:+.1f}%</b> | Relevance Diff: <b>{relevance_diff:+.1f}%</b></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color: #aaaaaa; font-size: 12px; text-align: center; padding: 10px; background-color: #3a3e44; border-radius: 8px; border: 1px dashed #4a4e54;">
            🔄 Chạy cả 2 model để xem so sánh
        </div>
        """, unsafe_allow_html=True)

    #UI chat history
    st.markdown("---")
    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #FFFFFF; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
        Lịch sử trò chuyện
    </h2>
    """, unsafe_allow_html=True)

    # Kiểm tra xem có lịch sử chưa
    if "chat_history" in st.session_state and st.session_state.chat_history:
        # Dùng reversed() để hiển thị câu hỏi mới nhất lên trên cùng
        for chat in reversed(st.session_state.chat_history):
            # Cắt ngắn câu hỏi làm tiêu đề expander nếu nó quá dài
            short_q = chat['question'][:25] + "..." if len(chat['question']) > 25 else chat['question']
            
            with st.expander(f"Q: {short_q}"):
                st.markdown(f"**Bạn:** {chat['question']}")
                st.markdown(f"**AI:** {chat['answer']}")
    else:
        st.markdown("""
        <div style="color: #aaaaaa; font-size: 13px; text-align: center; padding: 10px; background-color: #3a3e44; border-radius: 8px; border: 1px dashed #4a4e54;">
            Chưa có đoạn hội thoại nào.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear History", use_container_width=True, help="Xóa lịch sử chat trong session"):
            clear_history_dialog()
            
    with col2:
        if st.button("Clear Vector", use_container_width=True, help="Xóa tài liệu hiện tại"):
            clear_vector_store_dialog()

    st.markdown("---")
    st.markdown("""
    <h2 style="font-size: 20px; font-weight: 600; letter-spacing: 1px; color: #FFFFFF; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
        Quản lý Index
    </h2>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Save Index", use_container_width=True, help="Lưu vector index ra đĩa để dùng lại sau"):
            if st.session_state.vector_store:
                with st.spinner("Đang lưu index..."):
                    vector_engine = VectorEngine()
                    vector_engine.vector_store = st.session_state.vector_store
                    vector_engine.save_local_index()
                    st.success("Đã lưu index thành công!")
            else:
                st.warning("Chưa có vector store để lưu. Vui lòng upload tài liệu trước.")

    with col4:
        if st.button("Load Index", use_container_width=True, help="Tải vector index đã lưu từ đĩa"):
            with st.spinner("Đang tải index..."):
                vector_engine = VectorEngine()
                retriever = vector_engine.load_local_index()
                if retriever:
                    st.session_state.vector_store = vector_engine.vector_store
                    # Cần khôi phục lại chunks để lọc metadata, đây là một giới hạn cần cải thiện
                    # st.session_state.chunks = ... 
                    st.success("Đã tải index thành công!")
                else:
                    st.error("Không tìm thấy index đã lưu.")
    

# ------------------ MAIN AREA ------------------
# st.title("SmartDoc AI - Intelligent Document Q&A System")

st.markdown('<h1 style="color: #212529;">SmartDoc AI - Intelligent Document Q&A System</h1>', 
            unsafe_allow_html=True)


# st.markdown("### Hỏi đáp thông minh với tài liệu của bạn")
st.markdown('<h3 style="color: #212529;">Hỏi đáp thông minh với tài liệu của bạn</h3>', 
            unsafe_allow_html=True)

# Khởi tạo session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
#Lưu lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Lưu trữ performance metrics
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "doc_processing_time": 0,
        "embedding_time": 0,
        "qa_time": 0
    }

# Lưu trữ metrics so sánh RAG vs CoRAG
if "rag_corag_metrics" not in st.session_state:
    st.session_state.rag_corag_metrics = {
        "rag": {
            "qa_time": [],
            "retrieval_count": [],
            "relevance_scores": []
        },
        "corag": {
            "qa_time": [],
            "retrieval_count": [],
            "relevance_scores": []
        }
    }

if "model_selection" not in st.session_state:
    st.session_state.model_selection = "RAG"

# Thêm: Khởi tạo Memory lưu ngữ cảnh cho LangChain
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# ------------------ TẠO VECTOR STORE ------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


@st.cache_resource
def get_cross_encoder_compressor():
    cross_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=cross_model, top_n=3)


def is_follow_up_question(question: str) -> bool:
    lowered_question = question.lower().strip()
    follow_up_markers = [
        "phần đó",
        "mục đó",
        "đoạn đó",
        "ý đó",
        "nội dung của phần đó",
        "phần này",
        "mục này",
        "đoạn này",
        "ý này",
        "ở trên",
        "bên trên",
        "tiếp theo",
        "cái đó",
        "điều đó",
        "phần vừa nêu",
        "mục vừa nêu"
    ]
    return any(marker in lowered_question for marker in follow_up_markers)


def build_recent_chat_context(chat_history, max_turns: int = 2) -> str:
    recent_turns = chat_history[-max_turns:]
    context_lines = []
    for idx, turn in enumerate(recent_turns, start=1):
        context_lines.append(f"Lượt {idx} - Câu hỏi: {turn['question']}")
        context_lines.append(f"Lượt {idx} - Trả lời: {turn['answer']}")
    return "\n".join(context_lines)


def rewrite_follow_up_question(llm, question: str, chat_history):
    if not chat_history or not is_follow_up_question(question):
        return question

    recent_context = build_recent_chat_context(chat_history)
    rewrite_prompt = f"""
    Bạn có nhiệm vụ biến một câu hỏi follow-up thành câu hỏi độc lập, rõ nghĩa hơn để hệ thống truy xuất tài liệu chính xác hơn.

    Lịch sử hội thoại gần nhất:
    {recent_context}

    Câu hỏi follow-up hiện tại:
    {question}

    Yêu cầu:
    - Nếu câu hỏi hiện tại có từ tham chiếu như "phần đó", "mục đó", "ý trên", hãy thay bằng đối tượng cụ thể được nhắc gần nhất.
    - Giữ nguyên ý định gốc của người dùng.
    - Chỉ trả về câu hỏi đã viết lại.
    - Nếu lịch sử không đủ rõ để suy ra đối tượng, trả lại nguyên câu hỏi.
    """

    try:
        rewritten_question = llm.invoke(rewrite_prompt).strip()
        return rewritten_question or question
    except Exception:
        return question

# ===== CoRAG (CORRECTIVE RAG) =====
class CoRAGRetriever:
    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm
        self.retrieval_count = 0
        self.relevance_scores = []
    
    def retrieve_and_validate(self, question: str, max_retries: int = 2):
        """
        Retrieve documents và validate relevance.
        Nếu không relevant, tự động viết lại câu hỏi và retrieval lại.
        """
        self.retrieval_count = 0
        self.relevance_scores = []
        current_question = question
        
        for attempt in range(max_retries):
            self.retrieval_count += 1
            
            # Bước 1: Retrieve documents
            docs = self.base_retriever.get_relevant_documents(current_question)
            
            # Bước 2: Validate relevance của documents
            validation_prompt = f"""
            Đánh giá xem các documents sau có liên quan đến câu hỏi không?
            Câu hỏi: {current_question}
            
            Documents: {' '.join([d.page_content[:200] for d in docs[:3]])}
            
            Trả lời chỉ với: RELEVANT hoặc NOT_RELEVANT
            """
            
            validation_result = self.llm.invoke(validation_prompt).strip().upper()
            
            # Lưu relevance score (1.0 nếu RELEVANT, 0.0 nếu NOT_RELEVANT)
            relevance_score = 1.0 if "RELEVANT" in validation_result else 0.0
            self.relevance_scores.append(relevance_score)
            
            if "RELEVANT" in validation_result or attempt == max_retries - 1:
                return docs
            
            # Bước 3: Nếu không relevant, viết lại câu hỏi
            rewrite_prompt = f"""
            Viết lại câu hỏi này một cách khác để tìm kiếm tài liệu tốt hơn.
            CHỈ TRẢ VỀ CÂU HỎI, không thêm lời giải thích.
            
            Câu hỏi gốc: {current_question}
            """
            current_question = self.llm.invoke(rewrite_prompt).strip()
        
        return docs

# ------------------ UPLOAD FILE ------------------

st.markdown('<p class="custom-upload-label">📂 Tải lên tài liệu của bạn (PDF, DOCX)</p>', 
            unsafe_allow_html=True)

danh_sach_tai_len = st.file_uploader(
    "",
    type=["pdf", "docx"],
    accept_multiple_files=True, # Đã bật tính năng cho phép tải lên nhiều file
    help="Hỗ trợ PDF và DOCX, có thể chọn nhiều file cùng lúc"
)

# Khởi tạo dict lưu trữ byte của file PDF để hiển thị (nếu chưa có)
if "pdf_bytes_dict" not in st.session_state:
    st.session_state.pdf_bytes_dict = {}

if danh_sach_tai_len:
    # Lưu byte của các file PDF để tí nữa hiển thị iframe
    for file in danh_sach_tai_len:
        if file.name.endswith('.pdf'):
            st.session_state.pdf_bytes_dict[file.name] = file.getvalue()

    all_chunks = []
    with st.spinner("Đang xử lý các tài liệu..."):
            # --- METRICS: Bắt đầu đếm giờ xử lý tài liệu ---
            start_time_processing = time.time()

            # Vòng lặp gom chunks của tất cả các file lại
            for file in danh_sach_tai_len:
                chunks = process_document(
                    file,
                    st.session_state.chunk_size,
                    st.session_state.chunk_overlap
                )
                if chunks:
                    all_chunks.extend(chunks)
            
            # --- METRICS: Dừng đếm giờ và lưu kết quả ---
            st.session_state.metrics["doc_processing_time"] = time.time() - start_time_processing

            if all_chunks:
                st.session_state.chunks = all_chunks
                
                # Thay thế st.success bằng custom div
                st.markdown(f"""
                <div style="
                    color: #212529;
                    background-color: #d4edda;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    margin: 10px 0;">
                    ✅ Đã xử lý thành công {len(danh_sach_tai_len)} tài liệu! Tổng cộng {len(all_chunks)} đoạn (chunks). 
                    (Thời gian: {st.session_state.metrics['doc_processing_time']:.2f}s)
                </div>
                """, unsafe_allow_html=True)
                
                # Custom expander cho Chunk đầu tiên để test
                st.markdown("""
                <details style="margin: 10px 0; border: 1px solid #dee2e6; border-radius: 8px; padding: 8px;">
                    <summary style="color: #212529; font-weight: 600; cursor: pointer; padding: 8px;">
                        📄 Xem trước Chunk đầu tiên
                    </summary>
                    <div style="color: #212529; padding: 12px; background-color: #F8F9FA; border-radius: 4px; margin-top: 8px;">
                """ + all_chunks[0].page_content[:500] + "..."
                """
                    </div>
                </details>
                """, unsafe_allow_html=True)
                
                # Custom spinner và success message
                with st.spinner(""):
                    st.markdown('<p style="color: #212529;">⏳ Đang tạo embedding và vector store...</p>', 
                            unsafe_allow_html=True)
                    
                    # --- METRICS: Bắt đầu đếm giờ embedding ---
                    start_time_embedding = time.time()

                    st.session_state.vector_store = create_vector_store(all_chunks)

                    # --- METRICS: Dừng đếm giờ và lưu kết quả ---
                    st.session_state.metrics["embedding_time"] = time.time() - start_time_embedding
                    
                    st.markdown(f"""
                    <div style="
                        color: #212529;
                        background-color: #d4edda;
                        padding: 12px;
                        border-radius: 8px;
                        border-left: 4px solid #28a745;
                        margin: 10px 0;">
                        ✅ Vector store sẵn sàng! (Thời gian: {st.session_state.metrics['embedding_time']:.2f}s). Bạn có thể đặt câu hỏi bên dưới.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    color: #212529;
                padding: 12px;
                border-radius: 8px;
                border-left: 4px solid #dc3545;
                margin: 10px 0;">
                ❌ Không thể xử lý các tài liệu này. Vui lòng thử lại.
            </div>
            """, unsafe_allow_html=True)

# ------------------ PHẦN ĐẶT CÂU HỎI ------------------
if st.session_state.vector_store is not None:
    st.markdown("---")
    st.markdown('<h3 style="color: #212529;">Đặt câu hỏi về tài liệu</h3>', 
            unsafe_allow_html=True)
    
    # --- MODEL SELECTION ---
    model_mode = st.radio(
        "Chọn model:",
        ["RAG", "CoRAG (Corrective RAG)"],
        horizontal=True,
        help="RAG: Standard retrieval | CoRAG: Tự validate và cải thiện retrieval"
    )
    st.session_state.model_selection = model_mode
    
    search_mode = st.radio(
        "Chọn chế độ truy xuất (để so sánh):",
        ["Hybrid (Vector + từ khoá)", "Chỉ Vector Search (Pure Semantic)"],
        horizontal=True
    )

    danh_sach_file = list(set([chunk.metadata.get("source_file") 
                               for chunk in st.session_state.chunks 
                               if chunk.metadata.get("source_file")]))

    file_can_loc = st.selectbox("Lọc tìm kiếm theo tài liệu (Tuý chọn): ", ["Toàn bộ tài liệu"] + danh_sach_file)

    # Câu 9 & 10 Tính năng nâng cao:

    st.markdown("---")
    st.markdown("<b style='color: #212529;'>Tính năng nâng cao:</b>", unsafe_allow_html=True)
    col_adv1, col_adv2 = st.columns(2)
    with col_adv1:
        # câu 9:
        use_reranker = st.checkbox("Bật Re-ranking (MMR) để tăng tính đáng tin và đa dạng của kết quả.", help = "Sử dụng Cross-Encoder để sắp xếp lại tài liệu")
    with col_adv2:
        #câu 10:
        use_self_rag = st.checkbox("Bật Sè=lf-RAG để LLM tự đánh giá và chọn lọc tài liệu phù hợp với câu hỏi.", help="tự đông tối ưu câu hỏi và tự đánh giá (Confidence Score)")
    
    # Tạo retriever với similarity
    # retriever = st.session_state.vector_store.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 3}
    # )

    # Lưu ý mmr sẽ có tốc độ chậm hơn

    # Tạo retriever với chế độ MMR để tăng tính đa dạng thông tin
    # 1. FAISS Retriever (Semantic)

    faiss_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
    if file_can_loc != "Toàn bộ tài liệu":
        faiss_kwargs["filter"] = {"source_file": file_can_loc}

    faiss_retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=faiss_kwargs
    )
    
    # 2. BM25 Retriever (Keyword)
    bm25_retriever = BM25Retriever.from_documents(st.session_state.chunks)
    bm25_retriever.k = 5
    
    # 3. Kết hợp Hybrid
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]
    )

    # Điều kiện chọn retriever theo chế độ người dùng chọn
    if search_mode == "Hybrid (Vector + từ khoá)":
        active_retriever = hybrid_retriever
    else:
        active_retriever = faiss_retriever

    # Logic Re-ranking cho câu 9:
    if use_reranker:
        with st.spinner("Đang tải mô hình Cross-Encoder (Lần đầu sẽ tốn chút thời gian)..."):
            compressor = get_cross_encoder_compressor() # chỉ lấy top 3 chunk tốt nhất sau khi re-rank

            # Gói retriever hiện tại vào trong bộ nén (compressor) để tự động re-rank mỗi khi truy xuất
            #compressor : nhận vào một list các document trả về từ retriever, 
            # đánh giá lại chúng dựa trên sự phù hợp với câu hỏi, 
            # và chỉ giữ lại top N tài liệu tốt nhất để đưa vào LLM. 
            # Điều này giúp tăng chất lượng câu trả lời bằng cách đảm bảo rằng LLM chỉ phải xử lý những thông tin liên quan nhất.
            active_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=active_retriever
            )
    
    # --- CoRAG INITIALIZATION ---
    corag_retriever = None
    if st.session_state.model_selection == "CoRAG (Corrective RAG)":
        llm_temp = Ollama(model="qwen2.5:7b", temperature=0.2)
        corag_retriever = CoRAGRetriever(active_retriever, llm_temp)
            
    
    # Kết nối LLM (Ollama)
    try:
        llm = Ollama(model="qwen2.5:7b", temperature=0.2)
        
        # 1. ĐỊNH NGHĨA PROMPT TEMPLATE TỐI ƯU TIẾNG VIỆT
        # ====== [CÂU 10]: TÍCH HỢP CONFIDENCE SCORING VÀO PROMPT ======
        prompt_template = """
            [INSTRUCTION]
            Bạn là hệ thống RAG thông minh. Nhiệm vụ của bạn là trả lời câu hỏi CHỈ dựa trên Context được cung cấp.

            [CONSTRAINT]
            1. CHỈ sử dụng thông tin trong Context. Không bổ sung kiến thức bên ngoài.
            2. Trả lời bằng Tiếng Việt 100%, diễn giải rõ ràng.
            3. [QUAN TRỌNG]: Ở cuối câu trả lời, bạn BẮT BUỘC phải tự đánh giá độ tự tin (Confidence Score) về câu trả lời của mình dựa trên ngữ cảnh được cung cấp. Định dạng: "🎯 Độ tự tin (Confidence Score): X/100" và giải thích ngắn gọn tại sao bạn chấm điểm đó.

            [CONTEXT]
            {context}

            [QUESTION]
            {question}

            [ANSWER]
            """
        
        # 2. KHỞI TẠO PROMPT LANGCHAIN
        QA_PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        # 3. ĐƯA PROMPT VÀO CHAIN
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=active_retriever,
            memory=st.session_state.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT} 
        )
        
        
        with st.form("qa_form"):
            user_question = st.text_input(
                "Nhập câu hỏi của bạn:",
                key="user_question",
                placeholder="Ví dụ: Nội dung chính của tài liệu này là gì?"
            )
    
            submitted = st.form_submit_button("Hỏi")    
        
        def highlight_text(context, answer):
            words = answer.split()[:15]
            highlighted = context
            for w in words:
                if len(w) > 4:
                    highlighted = highlighted.replace(w, f"<mark>{w}</mark>")
            return highlighted

        if submitted and user_question:

            final_question = rewrite_follow_up_question(
                llm,
                user_question,
                st.session_state.chat_history
            )

            if final_question != user_question:
                st.info(f"**Câu hỏi follow-up đã được làm rõ:** {final_question}")

            if use_self_rag:
                with st.spinner("Self-RAG đang phân tích và tối ưu hoá câu hỏi (Query rewriting)..."):
                    # yêu cầu LLM tự đánh giá và viết lại câu hỏi tốt hơn
                    rewrite_prompt = f"""Nhiệm vụ của bạn là viết lại câu hỏi sau một cách rõ ràng, chi tiết hơn để hệ thống tìm kiếm tài liệu có thể hiểu dễ nhất. 
                    CHỈ TRẢ VỀ CÂU HỎI ĐÃ VIẾT LẠI, không thêm bất kỳ lời giải thích nào.
                    Câu hỏi gốc: "{user_question}"
                    """
                    final_question = llm.invoke(rewrite_prompt)

                    st.info(f"**Câu hỏi đã được tối ưu lại:** {final_question}")

            with st.spinner("Đang suy nghĩ..."):
                # --- METRICS: Bắt đầu đếm giờ trả lời câu hỏi ---
                start_time_qa = time.time()

                # --- SELECT RETRIEVER: RAG or CoRAG ---
                if st.session_state.model_selection == "CoRAG (Corrective RAG)" and corag_retriever:
                    # CoRAG: Retrieve with validation
                    docs = corag_retriever.retrieve_and_validate(final_question)
                    response = qa_chain.invoke({"question": final_question})
                    answer = response['answer']
                    
                    # Lưu CoRAG metrics
                    qa_time = time.time() - start_time_qa
                    st.session_state.rag_corag_metrics["corag"]["qa_time"].append(qa_time)
                    st.session_state.rag_corag_metrics["corag"]["retrieval_count"].append(corag_retriever.retrieval_count)
                    avg_relevance = sum(corag_retriever.relevance_scores) / len(corag_retriever.relevance_scores) if corag_retriever.relevance_scores else 0
                    st.session_state.rag_corag_metrics["corag"]["relevance_scores"].append(avg_relevance)
                else:
                    # Standard RAG
                    response = qa_chain.invoke({"question": final_question})
                    answer = response['answer']
                    
                    # Lưu RAG metrics
                    qa_time = time.time() - start_time_qa
                    st.session_state.rag_corag_metrics["rag"]["qa_time"].append(qa_time)
                    st.session_state.rag_corag_metrics["rag"]["retrieval_count"].append(1)
                    st.session_state.rag_corag_metrics["rag"]["relevance_scores"].append(1.0)

                # --- METRICS: Dừng đếm giờ và lưu kết quả ---
                st.session_state.metrics["qa_time"] = time.time() - start_time_qa

                # ===== ANSWER =====
                st.markdown('<h3 style="color: #212529;">Câu trả lời:</h3>', unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                    color: #212529;
                    background-color: #d4edda;
                    padding: 16px;
                    border-radius: 8px;
                    border-left: 5px solid #28a745;
                    margin: 10px 0;
                    font-size: 16px;
                    line-height: 1.6;">
                    {answer}
                </div>
                """, unsafe_allow_html=True)

                # --- METRICS: Hiển thị thời gian QA ---
                if st.session_state.model_selection == "CoRAG (Corrective RAG)" and corag_retriever:
                    st.info(f"⏱️ CoRAG - Thời gian: {st.session_state.metrics['qa_time']:.2f}s | Retrieval attempts: {corag_retriever.retrieval_count} | Avg Relevance: {sum(corag_retriever.relevance_scores)/len(corag_retriever.relevance_scores)*100:.1f}%")
                else:
                    st.info(f"⏱️ RAG - Thời gian xử lý câu hỏi: {st.session_state.metrics['qa_time']:.2f} giây")

                # ===== SOURCE =====
                with st.expander("📚 Nguồn tham khảo"):
                    for i, doc in enumerate(response['source_documents']):
                        page = doc.metadata.get("page", "Không rõ")
                        # --- THÊM 2 DÒNG LẤY METADATA NÀY ---
                        source_file = doc.metadata.get("source_file", "Không rõ")
                        uploaded_date = doc.metadata.get("uploaded_date", "Không rõ")
                        
                        highlighted = highlight_text(doc.page_content, answer)

                        # Cập nhật lại chuỗi in đậm hiển thị tên file
                        st.markdown(f"""
                        <div style="padding: 12px; background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; margin-bottom: 12px;">
                            <b>📄 File: {source_file} | Trang: {page}</b> (Tải lên: {uploaded_date})<br><br>
                            <div style="line-height:1.6;">
                                {highlighted[:500]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Fix lại nút button để tránh trùng key Streamlit khi có nhiều file
                        if st.button(f"Xem chi tiết Chunk {i+1} ({source_file})", key=f"view_{i}_{source_file}"):
                            st.session_state["selected_chunk"] = doc.page_content

                # ===== FULL CONTENT =====
                if "selected_chunk" in st.session_state:
                    st.markdown('<h3 style="color:#212529;">📖 Nội dung đầy đủ</h3>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="line-height:1.6;">
                        {st.session_state["selected_chunk"]}
                    </div>
                    """, unsafe_allow_html=True)

                # ===== SAVE HISTORY =====
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer
                })

                
    except Exception as e:
        st.markdown(f"""
        <div style="
            color: #212529;
            background-color: #f8d7da;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #dc3545;
            margin: 10px 0;">
            ❌ Lỗi kết nối LLM: {e}<br>
            Vui lòng đảm bảo Ollama đang chạy và đã pull model qwen2.5:7b
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="
        background-color: #e3f2fd; 
        padding: 12px; 
        border-radius: 8px; 
        border-left: 4px solid #007BFF;
        color: #212529;
        font-weight: 500;">
        Vui lòng tải lên tài liệu trước khi đặt câu hỏi.
    </div>
    """, unsafe_allow_html=True)

