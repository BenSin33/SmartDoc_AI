import logging
import streamlit as st

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logger = logging.getLogger(__name__)

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
    try:
        cross_model = get_cross_encoder()

        if not cross_model:
            return None

        return CrossEncoderReranker(
            model=cross_model,
            top_n=3
        )

    except Exception as e:
        logger.error(f"Lỗi compressor: {e}")
        return None


def is_follow_up_question(question: str) -> bool:
    lowered_question = question.lower().strip()

    markers = [
        "phần đó",
        "mục đó",
        "đoạn đó",
        "ý đó",
        "phần này",
        "mục này",
        "đoạn này",
        "ở trên",
        "bên trên",
        "tiếp theo",
        "cái đó",
        "điều đó",
        "phần vừa nêu",
        "mục vừa nêu"
    ]

    return any(marker in lowered_question for marker in markers)


def build_recent_chat_context(chat_history, max_turns: int = 2):
    recent_turns = chat_history[-max_turns:]

    context_lines = []

    for idx, turn in enumerate(recent_turns, start=1):
        context_lines.append(
            f"Lượt {idx} - Câu hỏi: {turn['question']}"
        )
        context_lines.append(
            f"Lượt {idx} - Trả lời: {turn['answer']}"
        )

    return "\n".join(context_lines)


def rewrite_follow_up_question(llm, question: str, chat_history):

    if not chat_history:
        return question

    if not is_follow_up_question(question):
        return question

    # câu dài thường đã đủ nghĩa
    if len(question.split()) > 12:
        return question

    recent_context = build_recent_chat_context(chat_history)

    rewrite_prompt = f"""
    Bạn là hệ thống rewrite câu hỏi cho RAG.

    Nhiệm vụ:
    - Viết lại câu hỏi follow-up thành câu hoàn chỉnh.
    - Giữ nguyên ý nghĩa.
    - Không thêm thông tin mới.
    - Chỉ trả về câu hỏi.

    Context:
    {recent_context}

    Follow-up question:
    {question}
    """

    try:
        rewritten = llm.invoke(rewrite_prompt).strip()

        if rewritten:
            logger.info(f"Rewrite question: {rewritten}")
            return rewritten

        return question

    except Exception as e:
        logger.error(f"Lỗi rewrite follow-up: {e}")
        return question


class CoRAGRetriever:

    def __init__(
        self,
        base_retriever,
        llm,
        relevance_threshold: float = 0.45,
        min_docs: int = 1
    ):

        self.base_retriever = base_retriever
        self.llm = llm
        self.cross_model = get_cross_encoder()

        self.relevance_threshold = relevance_threshold
        self.min_docs = min_docs

        self.retrieval_count = 0
        self.relevance_scores = []

    def _calculate_relevance_score(self, question, docs):

        if not docs:
            return 0.0

        if not self.cross_model:
            return 0.0

        try:

            pairs = [
                (question, doc.page_content[:1500])
                for doc in docs[:3]
            ]

            scores = self.cross_model.score(pairs)

            if not scores:
                return 0.0

            avg_score = sum(scores) / len(scores)

            return float(avg_score)

        except Exception as e:
            logger.error(f"Lỗi relevance scoring: {e}")
            return 0.0

    def _is_relevant(self, score, docs):

        if not docs:
            return False

        if len(docs) < self.min_docs:
            return False

        return score >= self.relevance_threshold

    def _rewrite_query(self, question):

        rewrite_prompt = f"""
        Bạn là hệ thống tối ưu câu hỏi retrieval cho RAG.

        Nhiệm vụ:
        - Viết lại câu hỏi để tìm tài liệu tốt hơn.
        - Giữ nguyên ý nghĩa.
        - Làm câu hỏi rõ ràng hơn.
        - Không trả lời câu hỏi.
        - Chỉ trả về câu hỏi mới.

        Original question:
        {question}
        """

        try:

            rewritten = self.llm.invoke(rewrite_prompt).strip()

            if rewritten:
                logger.info(f"Rewrite retrieval query: {rewritten}")
                return rewritten

            return question

        except Exception as e:
            logger.error(f"Lỗi rewrite retrieval: {e}")
            return question

    def retrieve_and_validate(
        self,
        question: str,
        max_retries: int = 2
    ):

        self.retrieval_count = 0
        self.relevance_scores = []

        current_question = question

        best_docs = []
        best_score = 0.0

        for attempt in range(max_retries):

            logger.info(f"Retrieval attempt: {attempt + 1}")

            self.retrieval_count += 1

            docs = self.base_retriever.invoke(current_question)

            score = self._calculate_relevance_score(
                current_question,
                docs
            )

            self.relevance_scores.append(score)

            logger.info(f"Relevance score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_docs = docs

            if self._is_relevant(score, docs):

                logger.info(
                    f"Relevant docs found with score: {score:.4f}"
                )

                return {
                    "documents": docs,
                    "score": score,
                    "is_relevant": True,
                    "retrieval_count": self.retrieval_count
                }


            if attempt < max_retries - 1:
                current_question = self._rewrite_query(
                    current_question
                )

        logger.warning(
            f"No highly relevant docs found. "
            f"Best score: {best_score:.4f}"
        )

        return {
            "documents": best_docs,
            "score": best_score,
            "is_relevant": best_score >= 0.30,
            "retrieval_count": self.retrieval_count
        }

def has_enough_context(result, minimum_score=0.35):

    if not result:
        return False

    return (
        result["is_relevant"]
        and result["score"] >= minimum_score
        and len(result["documents"]) > 0
    )
