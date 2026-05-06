import time
import logging
import streamlit as st
# Các module của Langchain & Third-party
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from style import CSS_STYLE
from document_processor import process_document
from vector_engine import VectorEngine, create_vector_store
from rag_core import (
    get_cross_encoder_compressor, 
    rewrite_follow_up_question, 
    CoRAGRetriever
)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================== GIAO DIỆN & CONFIG ===============================
st.set_page_config(page_title="SmartDoc AI", layout="wide")
st.markdown(CSS_STYLE, unsafe_allow_html=True)

# Các hàm phụ trợ UI
def highlight_text(context, answer):
    if not answer: return context
    words = answer.split()[:15]
    highlighted = context
    for w in words:
        if len(w) > 4:
            highlighted = highlighted.replace(w, f"<mark>{w}</mark>")
    return highlighted

@st.dialog("Xác nhận xóa lịch sử")
def clear_history_dialog():
    st.write("Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện không?")
    if st.button("Xác nhận xóa", type="primary"):
        st.session_state.chat_history = [] 
        if "memory" in st.session_state:
            st.session_state.memory.clear() 
        st.rerun() 

@st.dialog("Xác nhận xóa tài liệu")
def clear_vector_store_dialog():
    st.write("Dữ liệu vector và các đoạn văn bản (chunks) sẽ bị xóa hoàn toàn. Bạn sẽ cần upload lại tài liệu mới.")
    if st.button("Xác nhận xóa", type="primary"):
        st.session_state.vector_store = None
        st.session_state.chunks = None
        st.session_state.processed_files = [] # Reset danh sách file đã xử lý
        if "user_question" in st.session_state:
            st.session_state.user_question = ""
        if "pdf_bytes_dict" in st.session_state:
            st.session_state.pdf_bytes_dict = {}
        if "selected_chunk" in st.session_state:
            del st.session_state["selected_chunk"]
        st.session_state.uploader_key_version = st.session_state.get("uploader_key_version", 0) + 1
        st.rerun()

# =========================== KHỞI TẠO STATE ===============================
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "chunks" not in st.session_state: st.session_state.chunks = None
if "processed_files" not in st.session_state: st.session_state.processed_files = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {"doc_processing_time": 0, "embedding_time": 0, "qa_time": 0}
if "rag_corag_metrics" not in st.session_state:
    st.session_state.rag_corag_metrics = {
        "rag": {"qa_time": [], "retrieval_count": [], "relevance_scores": []},
        "corag": {"qa_time": [], "retrieval_count": [], "relevance_scores": []}
    }
if "model_selection" not in st.session_state: st.session_state.model_selection = "RAG"
if "uploader_key_version" not in st.session_state: st.session_state.uploader_key_version = 0
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

# =========================== SIDEBAR ===============================
with st.sidebar:
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

    with st.expander("📘 Hướng dẫn sử dụng", expanded=False):
        steps = [
            ("1", "Tải lên file PDF cần phân tích"), ("2", "Chờ hệ thống xử lý và phân tích"),
            ("3", "Đặt câu hỏi về nội dung tài liệu"), ("4", "Nhận câu trả lời thông minh từ AI")
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

    st.markdown("### ⚙️ Tùy chỉnh Chunk")
    chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap ", 0, 500, 100, 50)
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap

    st.markdown("### 📊 Performance Metrics")
    st.metric("Doc Processing", f"{st.session_state.metrics.get('doc_processing_time', 0):.2f} s")
    st.metric("Embedding", f"{st.session_state.metrics.get('embedding_time', 0):.2f} s")
    st.metric("Q&A Time", f"{st.session_state.metrics.get('qa_time', 0):.2f} s")

    st.markdown("---")
    st.markdown("""<h2 style="font-size: 20px; font-weight: 600; color: #FFFFFF; margin-bottom: 16px;">📊 RAG vs CoRAG</h2>""", unsafe_allow_html=True)
    
    rag_metrics = st.session_state.rag_corag_metrics["rag"]
    corag_metrics = st.session_state.rag_corag_metrics["corag"]
    if len(rag_metrics["qa_time"]) > 0 or len(corag_metrics["qa_time"]) > 0:
        if len(rag_metrics["qa_time"]) > 0:
            avg_rag_time = sum(rag_metrics["qa_time"]) / len(rag_metrics["qa_time"])
            avg_rag_relevance = sum(rag_metrics["relevance_scores"]) / len(rag_metrics["relevance_scores"])
            st.markdown(f"""<div style="background-color: #1e3a5f; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #007BFF;"><b style="color: #007BFF;">📈 RAG</b><br><span style="color: #FFFFFF; font-size: 12px;">Avg Time: <b>{avg_rag_time:.2f}s</b> | Relevance: <b>{avg_rag_relevance*100:.1f}%</b></span></div>""", unsafe_allow_html=True)
        if len(corag_metrics["qa_time"]) > 0:
            avg_corag_time = sum(corag_metrics["qa_time"]) / len(corag_metrics["qa_time"])
            avg_corag_relevance = sum(corag_metrics["relevance_scores"]) / len(corag_metrics["relevance_scores"])
            st.markdown(f"""<div style="background-color: #1f3a1f; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid #28a745;"><b style="color: #28a745;">🔄 CoRAG</b><br><span style="color: #FFFFFF; font-size: 12px;">Avg Time: <b>{avg_corag_time:.2f}s</b> | Relevance: <b>{avg_corag_relevance*100:.1f}%</b></span></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="color: #aaaaaa; font-size: 12px; text-align: center; padding: 10px; background-color: #3a3e44; border-radius: 8px; border: 1px dashed #4a4e54;">🔄 Chạy cả 2 model để xem so sánh</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<h2 style="font-size: 20px; font-weight: 600; color: #FFFFFF; margin-bottom: 16px;">Lịch sử trò chuyện</h2>""", unsafe_allow_html=True)
    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            short_q = chat['question'][:25] + "..." if len(chat['question']) > 25 else chat['question']
            with st.expander(f"Q: {short_q}"):
                st.markdown(f"**Bạn:** {chat['question']}")
                st.markdown(f"**AI:** {chat['answer']}")
    else:
        st.markdown("""<div style="color: #aaaaaa; font-size: 13px; text-align: center; padding: 10px; background-color: #3a3e44; border-radius: 8px; border: 1px dashed #4a4e54;">Chưa có đoạn hội thoại nào.</div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear History", use_container_width=True): clear_history_dialog()
    with col2:
        if st.button("Clear Vector", use_container_width=True): clear_vector_store_dialog()

    st.markdown("---")
    st.markdown("""<h2 style="font-size: 20px; font-weight: 600; color: #FFFFFF; margin-bottom: 16px;">Quản lý Index</h2>""", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Save Index", use_container_width=True):
            if st.session_state.vector_store:
                with st.spinner("Đang lưu index..."):
                    vector_engine = VectorEngine()
                    vector_engine.vector_store = st.session_state.vector_store
                    vector_engine.save_local_index()
                    st.success("Đã lưu index thành công!")
            else: st.warning("Chưa có vector store để lưu.")
    with col4:
        if st.button("Load Index", use_container_width=True):
            with st.spinner("Đang tải index..."):
                vector_engine = VectorEngine()
                retriever = vector_engine.load_local_index()
                if retriever:
                    st.session_state.vector_store = vector_engine.vector_store
                    st.session_state.chunks = vector_engine.chunks
                    st.success("Đã tải index thành công!")
                else: st.error("Không tìm thấy index đã lưu.")

# =========================== MAIN AREA ===============================
st.markdown('<h1 style="color: #212529;">SmartDoc AI - Intelligent Document Q&A System</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="color: #212529;">Hỏi đáp thông minh với tài liệu của bạn</h3>', unsafe_allow_html=True)

st.markdown('<p class="custom-upload-label">📂 Tải lên tài liệu của bạn (PDF, DOCX)</p>', unsafe_allow_html=True)
danh_sach_tai_len = st.file_uploader("", type=["pdf", "docx"], accept_multiple_files=True, key=f"uploaded_documents_{st.session_state.uploader_key_version}")

if "pdf_bytes_dict" not in st.session_state: st.session_state.pdf_bytes_dict = {}

# LOGIC XỬ LÝ FILE (TRÁNH LẶP LẠI)
if danh_sach_tai_len:
    current_names = [f.name for f in danh_sach_tai_len]
    if st.session_state.processed_files != current_names:
        with st.spinner("⏳ Hệ thống đang xử lý tài liệu mới..."):
            try:
                start_proc = time.time()
                all_chunks = []
                for file in danh_sach_tai_len:
                    if file.name.endswith('.pdf'):
                        st.session_state.pdf_bytes_dict[file.name] = file.getvalue()
                    chunks = process_document(file, st.session_state.chunk_size, st.session_state.chunk_overlap)
                    if chunks: all_chunks.extend(chunks)
                
                st.session_state.metrics["doc_processing_time"] = time.time() - start_proc

                if all_chunks:
                    st.session_state.chunks = all_chunks
                    st.toast(f"✅ Đã xử lý {len(all_chunks)} chunks", icon="📄")
                    
                    start_embed = time.time()
                    st.session_state.vector_store = create_vector_store(all_chunks)
                    st.session_state.metrics["embedding_time"] = time.time() - start_embed
                    
                    st.session_state.processed_files = current_names # Lưu trạng thái đã xử lý
                    st.success("✅ Thành công! Vector store đã sẵn sàng.")
                else:
                    st.error("❌ Không thể trích xuất nội dung.")
            except Exception as e:
                st.error(f"❌ Lỗi xử lý: {str(e)}")

# =========================== PHẦN ĐẶT CÂU HỎI ===============================
if st.session_state.vector_store is not None:
    st.markdown("---")
    st.markdown('<h3 style="color: #212529;">Đặt câu hỏi về tài liệu</h3>', unsafe_allow_html=True)
    
    model_mode = st.radio("Chọn model:", ["RAG", "CoRAG (Corrective RAG)"], horizontal=True)
    st.session_state.model_selection = model_mode
    search_mode = st.radio("Chọn chế độ truy xuất:", ["Hybrid (Vector + từ khoá)", "Chỉ Vector Search (Pure Semantic)"], horizontal=True)

    if st.session_state.chunks:
        danh_sach_file = list(set([
            chunk.metadata.get("source_file") 
            for chunk in st.session_state.chunks 
            if chunk.metadata.get("source_file")
        ]))
    else:
        danh_sach_file = []
    file_can_loc = st.selectbox("Lọc tìm kiếm theo tài liệu:", ["Toàn bộ tài liệu"] + danh_sach_file)

    st.markdown("<b style='color: #212529;'>Tính năng nâng cao:</b>", unsafe_allow_html=True)
    col_adv1, col_adv2 = st.columns(2)
    with col_adv1: use_reranker = st.checkbox("Bật Re-ranking (Cross-Encoder)")
    with col_adv2: use_self_rag = st.checkbox("Bật Self-RAG")
    
    faiss_kwargs={"k": 5}
    if file_can_loc != "Toàn bộ tài liệu":
        faiss_kwargs["filter"] = {"source_file": file_can_loc}

    faiss_retriever = st.session_state.vector_store.as_retriever(search_kwargs=faiss_kwargs)
    
    # Khởi tạo BM25 cho Hybrid
    if st.session_state.chunks:
        bm25_retriever = BM25Retriever.from_documents(st.session_state.chunks)
        bm25_retriever.k = 5
    else:
        bm25_retriever = None
    if bm25_retriever:
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7]
        )
        active_retriever = hybrid_retriever if search_mode == "Hybrid (Vector + từ khoá)" else faiss_retriever
    else:
        active_retriever = faiss_retriever

    if use_reranker:
        compressor = get_cross_encoder_compressor()
        if compressor:
            active_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=active_retriever)
    
    try:
        llm = Ollama(model="qwen2.5:7b", temperature=0.1)
        prompt_template = """Chỉ sử dụng Context sau để trả lời. Trả lời bằng Tiếng Việt 100%.
        Context: {context}
        Câu hỏi: {question}
        [ANSWER]:"""
        
        QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=active_retriever, memory=st.session_state.memory,
            return_source_documents=True, combine_docs_chain_kwargs={"prompt": QA_PROMPT} 
        )
        
        with st.form("qa_form"):
            user_question = st.text_input("Nhập câu hỏi của bạn:", key="user_question_input")
            submitted = st.form_submit_button("Hỏi")    
        
        if submitted and user_question:
            # 1. Xử lý câu hỏi follow-up
            final_question = rewrite_follow_up_question(llm, user_question, st.session_state.chat_history)
            if final_question != user_question:
                st.info(f"**Câu hỏi follow-up đã được làm rõ:** {final_question}")

            if use_self_rag:
                with st.spinner("Self-RAG đang tối ưu..."):
                    final_question = llm.invoke(f"Viết lại câu hỏi này để tìm kiếm tài liệu tốt hơn: {final_question}")

            with st.spinner("Đang suy nghĩ..."):
                start_time_qa = time.time()
                
                if st.session_state.model_selection == "CoRAG (Corrective RAG)":
                    corag = CoRAGRetriever(active_retriever, llm)
                    docs = corag.retrieve_and_validate(final_question)
                    response = qa_chain.invoke({"question": final_question})
                    
                    # Update metrics
                    st.session_state.rag_corag_metrics["corag"]["qa_time"].append(time.time() - start_time_qa)
                    st.session_state.rag_corag_metrics["corag"]["relevance_scores"].append(sum(corag.relevance_scores)/len(corag.relevance_scores) if corag.relevance_scores else 0)
                else:
                    response = qa_chain.invoke({"question": final_question})
                    st.session_state.rag_corag_metrics["rag"]["qa_time"].append(time.time() - start_time_qa)
                    st.session_state.rag_corag_metrics["rag"]["relevance_scores"].append(1.0)

                answer = response['answer']
                st.session_state.metrics["qa_time"] = time.time() - start_time_qa
                
                st.markdown('<h3 style="color: #212529;">Câu trả lời:</h3>', unsafe_allow_html=True)
                st.markdown(f"""<div style="color: #212529; background-color: #d4edda; padding: 16px; border-radius: 8px; border-left: 5px solid #28a745;">{answer}</div>""", unsafe_allow_html=True)

                with st.expander("📚 Nguồn tham khảo"):
                    for i, doc in enumerate(response['source_documents']):
                        page = doc.metadata.get("page", "N/A")
                        source_file = doc.metadata.get("source_file", "N/A")
                        highlighted = highlight_text(doc.page_content, answer)
                        st.markdown(f"""<div style="padding: 12px; background-color: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; margin-bottom: 12px;"><b>📄 File: {source_file} | Trang: {page}</b><br><br>{highlighted[:500]}...</div>""", unsafe_allow_html=True)
                        if st.button(f"Xem chi tiết Chunk {i+1}", key=f"view_{i}_{source_file}"):
                            st.session_state["selected_chunk"] = doc.page_content

                if "selected_chunk" in st.session_state:
                    st.markdown('<h3 style="color:#212529;">📖 Nội dung đầy đủ</h3>', unsafe_allow_html=True)
                    st.info(st.session_state["selected_chunk"])

                st.session_state.chat_history.append({"question": user_question, "answer": answer})
                
    except Exception as e:
        st.error(f"❌ Lỗi kết nối: {e}")
else:
    st.markdown("""<div style="background-color: #e3f2fd; padding: 12px; border-radius: 8px; border-left: 4px solid #007BFF; color: #212529;">Vui lòng tải lên tài liệu trước khi đặt câu hỏi.</div>""", unsafe_allow_html=True)