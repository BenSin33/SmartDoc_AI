import streamlit as st
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

@st.cache_resource
def get_cross_encoder_compressor():
    cross_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=cross_model, top_n=3)

def is_follow_up_question(question: str) -> bool:
    lowered_question = question.lower().strip()
    follow_up_markers = [
        "phần đó", "mục đó", "đoạn đó", "ý đó",
        "nội dung của phần đó", "phần này", "mục này",
        "đoạn này", "ý này", "ở trên", "bên trên",
        "tiếp theo", "cái đó", "điều đó", "phần vừa nêu", "mục vừa nêu"
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

class CoRAGRetriever:
    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm
        self.retrieval_count = 0
        self.relevance_scores = []
    
    def retrieve_and_validate(self, question: str, max_retries: int = 2):
        self.retrieval_count = 0
        self.relevance_scores = []
        current_question = question
        
        for attempt in range(max_retries):
            self.retrieval_count += 1
            docs = self.base_retriever.get_relevant_documents(current_question)
            
            validation_prompt = f"""
            Đánh giá xem các documents sau có liên quan đến câu hỏi không?
            Câu hỏi: {current_question}
            
            Documents: {' '.join([d.page_content[:200] for d in docs[:3]])}
            
            Trả lời chỉ với: RELEVANT hoặc NOT_RELEVANT
            """
            
            validation_result = self.llm.invoke(validation_prompt).strip().upper()
            relevance_score = 1.0 if "RELEVANT" in validation_result else 0.0
            self.relevance_scores.append(relevance_score)
            
            if "RELEVANT" in validation_result or attempt == max_retries - 1:
                return docs
            
            rewrite_prompt = f"""
            Viết lại câu hỏi này một cách khác để tìm kiếm tài liệu tốt hơn.
            CHỈ TRẢ VỀ CÂU HỎI, không thêm lời giải thích.
            
            Câu hỏi gốc: {current_question}
            """
            current_question = self.llm.invoke(rewrite_prompt).strip()
        
        return docs