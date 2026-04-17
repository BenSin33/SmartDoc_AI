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
#===========================GIAO DIEN=============================================

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
    .stApp {
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
    
    /* Đổi màu đen cho tất cả text trong main area */
    .main, .stAlert, .streamlit-expanderHeader, .stSpinner, .stFileUploader {
        color: #000000 !important;
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
# st.title("SmartDoc AI - Intelligent Document Q&A System")

st.markdown('<h1 style="color: #000000;">SmartDoc AI - Intelligent Document Q&A System</h1>', 
            unsafe_allow_html=True)


# st.markdown("### Hỏi đáp thông minh với tài liệu của bạn")
st.markdown('<h3 style="color: #000000;">Hỏi đáp thông minh với tài liệu của bạn</h3>', 
            unsafe_allow_html=True)

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

st.markdown("""
<style>
    /* Custom label cho file uploader */
    .custom-upload-label {
        color: #000000;
        font-weight: 500;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="custom-upload-label">📂 Tải lên tài liệu của bạn (PDF, DOCX)</p>', 
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["pdf", "docx"],
    help="Hỗ trợ PDF và DOCX, tối đa 200MB mỗi file"
)

if uploaded_file is not None:
    with st.spinner("Đang xử lý tài liệu..."):
        document_chunks = process_document(uploaded_file)
        
        # if document_chunks:
        #     st.session_state.chunks = document_chunks
        #     st.success(f"Đã xử lý thành công! Chia thành {len(document_chunks)} đoạn (chunks).")
            
        #     # Hiển thị xem trước chunks (theo yêu cầu)
        #     with st.expander("Xem trước Chunk đầu tiên"):
        #         st.write(document_chunks[0].page_content[:500] + "...")
            
        #     if len(document_chunks) > 1:
        #         with st.expander("Xem trước Chunk thứ hai"):
        #             st.write(document_chunks[1].page_content[:500] + "...")
            
        #     if len(document_chunks) > 2:
        #         with st.expander("Xem trước Chunk thứ ba"):
        #             st.write(document_chunks[2].page_content[:500] + "...")
            
        #     # Tạo vector store
        #     with st.spinner("Đang tạo embedding và vector store..."):
        #         st.session_state.vector_store = create_vector_store(document_chunks)
        #         st.success("Vector store sẵn sàng! Bạn có thể đặt câu hỏi bên dưới.")
        # else:
        #     st.error("Không thể xử lý tài liệu. Vui lòng thử lại.")

        if document_chunks:
            st.session_state.chunks = document_chunks
            
            # Thay thế st.success bằng custom div
            st.markdown(f"""
            <div style="
                color: #000000;
                background-color: #d4edda;
                padding: 12px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
                margin: 10px 0;">
                ✅ Đã xử lý thành công! Chia thành {len(document_chunks)} đoạn (chunks).
            </div>
            """, unsafe_allow_html=True)
            
            # Custom expander cho Chunk đầu tiên
            st.markdown("""
            <details style="margin: 10px 0; border: 1px solid #dee2e6; border-radius: 8px; padding: 8px;">
                <summary style="color: #000000; font-weight: 600; cursor: pointer; padding: 8px;">
                    📄 Xem trước Chunk đầu tiên
                </summary>
                <div style="color: #000000; padding: 12px; background-color: #f8f9fa; border-radius: 4px; margin-top: 8px;">
            """ + document_chunks[0].page_content[:500] + "..."
            """
                </div>
            </details>
            """, unsafe_allow_html=True)
            
            # Custom expander cho Chunk thứ hai
            if len(document_chunks) > 1:
                st.markdown("""
                <details style="margin: 10px 0; border: 1px solid #dee2e6; border-radius: 8px; padding: 8px;">
                    <summary style="color: #000000; font-weight: 600; cursor: pointer; padding: 8px;">
                        📄 Xem trước Chunk thứ hai
                    </summary>
                    <div style="color: #000000; padding: 12px; background-color: #f8f9fa; border-radius: 4px; margin-top: 8px;">
                """ + document_chunks[1].page_content[:500] + "..."
                """
                    </div>
                </details>
                """, unsafe_allow_html=True)
            
            # Custom expander cho Chunk thứ ba
            if len(document_chunks) > 2:
                st.markdown("""
                <details style="margin: 10px 0; border: 1px solid #dee2e6; border-radius: 8px; padding: 8px;">
                    <summary style="color: #000000; font-weight: 600; cursor: pointer; padding: 8px;">
                        📄 Xem trước Chunk thứ ba
                    </summary>
                    <div style="color: #000000; padding: 12px; background-color: #f8f9fa; border-radius: 4px; margin-top: 8px;">
                """ + document_chunks[2].page_content[:500] + "..."
                """
                    </div>
                </details>
                """, unsafe_allow_html=True)
            
            # Custom spinner và success message
            with st.spinner(""):
                st.markdown('<p style="color: #000000;">⏳ Đang tạo embedding và vector store...</p>', 
                        unsafe_allow_html=True)
                
                st.session_state.vector_store = create_vector_store(document_chunks)
                
                st.markdown("""
                <div style="
                    color: #000000;
                    background-color: #d4edda;
                    padding: 12px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    margin: 10px 0;">
                    ✅ Vector store sẵn sàng! Bạn có thể đặt câu hỏi bên dưới.
                </div>
                """, unsafe_allow_html=True)
        else:
            # Thay thế st.error
            st.markdown("""
            <div style="
                color: #000000;
                background-color: #f8d7da;
                padding: 12px;
                border-radius: 8px;
                border-left: 4px solid #dc3545;
                margin: 10px 0;">
                ❌ Không thể xử lý tài liệu. Vui lòng thử lại.
            </div>
            """, unsafe_allow_html=True)

# ------------------ PHẦN ĐẶT CÂU HỎI ------------------
if st.session_state.vector_store is not None:
    st.markdown("---")
    # st.subheader("Đặt câu hỏi về tài liệu")
    st.markdown('<h3 style="color: #000000;">Đặt câu hỏi về tài liệu</h3>', 
            unsafe_allow_html=True)
    
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
        
        user_question = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Nội dung chính của tài liệu này là gì?")
        
        if user_question:
            with st.spinner("Đang suy nghĩ..."):
                response = qa_chain.invoke({"query": user_question})
                answer = response['result']
                
                # Hiển thị câu trả lời
                st.markdown("### Câu trả lời:")
                st.success(answer)
                
                # Tùy chọn xem source chunks
                with st.expander("Xem nguồn tham khảo (chunks liên quan)"):
                    for i, doc in enumerate(response['source_documents']):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content[:400] + "...")
                        st.markdown("---")
    
    # except Exception as e:
    #     st.error(f"Lỗi kết nối LLM: {e}\nVui lòng đảm bảo Ollama đang chạy và đã pull model qwen2.5:7b")
    except Exception as e:
        st.markdown(f"""
        <div style="
            color: #000000;
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
    # st.info("Vui lòng tải lên tài liệu trước khi đặt câu hỏi.")
    st.markdown("""
    <div style="
        background-color: #e3f2fd; 
        padding: 12px; 
        border-radius: 8px; 
        border-left: 4px solid #007BFF;
        color: #000000;
        font-weight: 500;">
        Vui lòng tải lên tài liệu trước khi đặt câu hỏi.
    </div>
    """, unsafe_allow_html=True)



#=========================================================================

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
