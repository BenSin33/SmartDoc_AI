import logging
import os
import tempfile
import streamlit as st
import base64
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

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
    input {
        color: #212529 !important;
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

# ------------------ UPLOAD FILE ------------------

st.markdown('<p class="custom-upload-label">📂 Tải lên tài liệu của bạn (PDF, DOCX)</p>', 
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["pdf", "docx"],
    help="Hỗ trợ PDF và DOCX, tối đa 200MB mỗi file"
)

if uploaded_file is not None:
    st.session_state["pdf_bytes"] = uploaded_file.getvalue()

if uploaded_file is not None:
    with st.spinner("Đang xử lý tài liệu..."):
        document_chunks = process_document(
            uploaded_file,
            st.session_state.chunk_size,
            st.session_state.chunk_overlap
        )
        
        if document_chunks:
            st.session_state.chunks = document_chunks
            
            # Thay thế st.success bằng custom div
            st.markdown(f"""
            <div style="
                color: #212529;
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
                <summary style="color: #212529; font-weight: 600; cursor: pointer; padding: 8px;">
                    📄 Xem trước Chunk đầu tiên
                </summary>
                <div style="color: #212529; padding: 12px; background-color: #F8F9FA; border-radius: 4px; margin-top: 8px;">
            """ + document_chunks[0].page_content[:500] + "..."
            """
                </div>
            </details>
            """, unsafe_allow_html=True)
            
            # Custom expander cho Chunk thứ hai
            if len(document_chunks) > 1:
                st.markdown("""
                <details style="margin: 10px 0; border: 1px solid #dee2e6; border-radius: 8px; padding: 8px;">
                    <summary style="color: #212529; font-weight: 600; cursor: pointer; padding: 8px;">
                        📄 Xem trước Chunk thứ hai
                    </summary>
                    <div style="color: #212529; padding: 12px; background-color: #F8F9FA; border-radius: 4px; margin-top: 8px;">
                """ + document_chunks[1].page_content[:500] + "..."
                """
                    </div>
                </details>
                """, unsafe_allow_html=True)
            
            # Custom expander cho Chunk thứ ba
            if len(document_chunks) > 2:
                st.markdown("""
                <details style="margin: 10px 0; border: 1px solid #dee2e6; border-radius: 8px; padding: 8px;">
                    <summary style="color: #212529; font-weight: 600; cursor: pointer; padding: 8px;">
                        📄 Xem trước Chunk thứ ba
                    </summary>
                    <div style="color: #212529; padding: 12px; background-color: #F8F9FA; border-radius: 4px; margin-top: 8px;">
                """ + document_chunks[2].page_content[:500] + "..."
                """
                    </div>
                </details>
                """, unsafe_allow_html=True)
            
            # Custom spinner và success message
            with st.spinner(""):
                st.markdown('<p style="color: #212529;">⏳ Đang tạo embedding và vector store...</p>', 
                        unsafe_allow_html=True)
                
                st.session_state.vector_store = create_vector_store(document_chunks)
                
                st.markdown("""
                <div style="
                    color: #212529;
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
                color: #212529;
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
    st.markdown('<h3 style="color: #212529;">Đặt câu hỏi về tài liệu</h3>', 
            unsafe_allow_html=True)
    
    # Tạo retriever với similarity
    # retriever = st.session_state.vector_store.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k": 3}
    # )

    # Lưu ý mmr sẽ có tốc độ chậm hơn

    # Tạo retriever với chế độ MMR để tăng tính đa dạng thông tin
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,                # Lấy ra 3 đoạn cuối cùng cho AI
            "fetch_k": 20,         # Quét trước 20 đoạn tiềm năng
            "lambda_mult": 0.7     # Độ đa dạng (0.5 - 0.7 là mức ổn định)
        }
    )
    
    # Kết nối LLM (Ollama)
    try:
        llm = Ollama(model="qwen2.5:7b", temperature=0.2)
        
        # 1. ĐỊNH NGHĨA PROMPT TEMPLATE TỐI ƯU TIẾNG VIỆT
        prompt_template = """
            [INSTRUCTION]
            Bạn là hệ thống hỏi đáp tài liệu (RAG). Nhiệm vụ của bạn là trả lời câu hỏi CHỈ dựa trên Context được cung cấp.

            [CONSTRAINT - BẮT BUỘC TUÂN THỦ]
            1. CHỈ sử dụng thông tin trong Context. Không suy diễn, không bổ sung kiến thức bên ngoài.
            2. Nếu Context KHÔNG chứa thông tin để trả lời:
            -> CHỈ được trả lời đúng 1 câu duy nhất:
            "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp."
            -> KHÔNG được viết thêm bất kỳ nội dung nào khác.
            3. Trả lời bằng Tiếng Việt 100%.
            4. Trình bày:
            - Ngắn gọn
            - Rõ ràng
            - Dùng bullet points nếu có nhiều ý
            5. Không lặp lại nguyên văn Context, phải diễn giải lại.
            6. Không thêm giải thích ngoài câu hỏi.

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
            retriever=retriever,
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
            with st.spinner("Đang suy nghĩ..."):
                response = qa_chain.invoke({"question": user_question})
                answer = response['answer']

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

                # ===== SOURCE =====
                with st.expander("📚 Nguồn tham khảo"):
                    for i, doc in enumerate(response['source_documents']):
                        page = doc.metadata.get("page", "Không rõ")
                        highlighted = highlight_text(doc.page_content, answer)

                        st.markdown(f"""
                        <div style="
                            padding: 12px;
                            background-color: #ffffff;
                            border: 1px solid #dee2e6;
                            border-radius: 8px;
                            margin-bottom: 12px;
                        ">
                            <b>📄 Trang: {page}</b><br><br>
                            <div style="line-height:1.6;">
                                {highlighted[:500]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # BUTTON
                        if st.button(f"Xem chi tiết Chunk {i+1}", key=f"view_{i}"):
                            st.session_state["selected_chunk"] = doc.page_content

                        # PDF VIEWER
                        if "pdf_bytes" in st.session_state:
                            import base64
                            base64_pdf = base64.b64encode(st.session_state["pdf_bytes"]).decode('utf-8')

                            pdf_display = f"""
                            <iframe src="data:application/pdf;base64,{base64_pdf}#page={page}"
                            width="100%" height="500" type="application/pdf"
                            style="border:1px solid #ccc;"></iframe>
                            """

                            st.markdown(pdf_display, unsafe_allow_html=True)

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