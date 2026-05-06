import streamlit as st
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import logging

logger = logging.getLogger(__name__)

# =========================
# 🔹 CROSS ENCODER (GIỮ API CŨ)
# =========================
@st.cache_resource
def get_cross_encoder():
    try:
        return HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    except Exception as e:
        logger.error(f"Lỗi load CrossEncoder: {e}")
        return None


@st.cache_resource
def get_cross_encoder_compressor():
    """
    ⚠️ GIỮ TÊN HÀM CŨ để app.py không lỗi import
    """
    try:
        cross_model = get_cross_encoder()
        if not cross_model:
            return None

        return CrossEncoderReranker(
            model=cross_model,
            top_n=2  # giảm nhẹ để tăng tốc
        )
    except Exception as e:
        logger.error(f"Lỗi compressor: {e}")
        return None


# =========================
# 🔹 FOLLOW-UP (GIỮ NGUYÊN API)
# =========================
def is_follow_up_question(question: str) -> bool:
    lowered_question = question.lower().strip()

    markers = [
        "phần đó", "mục đó", "đoạn đó", "ý đó",
        "phần này", "mục này", "đoạn này",
        "ở trên", "bên trên",
        "tiếp theo", "cái đó", "điều đó",
        "phần vừa nêu", "mục vừa nêu"
    ]

    return any(marker in lowered_question for marker in markers)


def build_recent_chat_context(chat_history, max_turns: int = 2):
    recent_turns = chat_history[-max_turns:]

    context_lines = []
    for idx, turn in enumerate(recent_turns, start=1):
        context_lines.append(f"Lượt {idx} - Câu hỏi: {turn['question']}")
        context_lines.append(f"Lượt {idx} - Trả lời: {turn['answer']}")

    return "\n".join(context_lines)


def rewrite_follow_up_question(llm, question: str, chat_history):
    if not chat_history or not is_follow_up_question(question):
        return question

    # ⚡ giảm gọi LLM nếu câu đã đủ rõ
    if len(question.split()) > 8:
        return question

    recent_context = build_recent_chat_context(chat_history)

    rewrite_prompt = f"""
    Viết lại câu hỏi follow-up thành câu rõ nghĩa.

    Context:
    {recent_context}

    Question:
    {question}

    Chỉ trả về câu hỏi.
    """

    try:
        rewritten = llm.invoke(rewrite_prompt).strip()
        return rewritten if rewritten else question
    except Exception:
        return question


# =========================
# 🔥 CoRAG (GIỮ API CŨ - OPTIMIZED)
# =========================
class CoRAGRetriever:
    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm
        self.retrieval_count = 0
        self.relevance_scores = []

    # ⚡ heuristic thay LLM validation
    def _is_relevant(self, docs):
        if not docs:
            return False

        # kiểm tra độ dài nội dung
        total_length = sum(len(d.page_content) for d in docs[:3])

        return total_length > 300  # threshold nhẹ

    def retrieve_and_validate(self, question: str, max_retries: int = 2):
        self.retrieval_count = 0
        self.relevance_scores = []

        current_question = question
        best_docs = []

        for attempt in range(max_retries):
            self.retrieval_count += 1

            docs = self.base_retriever.invoke(current_question)

            if docs:
                best_docs = docs

            # ⚡ heuristic score (nhanh)
            is_relevant = self._is_relevant(docs)
            score = 1.0 if is_relevant else 0.0
            self.relevance_scores.append(score)

            if is_relevant or attempt == max_retries - 1:
                return docs

            # 🔁 chỉ rewrite khi fail
            try:
                rewrite_prompt = f"""
                Viết lại câu hỏi để tìm tài liệu tốt hơn:
                {current_question}
                """
                current_question = self.llm.invoke(rewrite_prompt).strip()
            except:
                break

        return best_docs