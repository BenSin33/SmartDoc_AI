import os
import pickle
import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

class VectorEngine:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        logger.info(f"Initializing embeddings with model: {model_name}...")
        
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vector_store = None
        self.chunks = None   # thêm

    # ================= CREATE =================
    def create(self, documents):
        logger.info(f"Creating FAISS index with {len(documents)} chunks...")
        
        self.chunks = documents   # lưu chunks
        self.vector_store = FAISS.from_documents(documents, self.embedder)

        return self.vector_store.as_retriever(search_kwargs={"k": 3})

    # ================= SAVE =================
    def save_local_index(self, folder_path: str = "faiss_index"):
        if not self.vector_store:
            logger.warning("No vector store to save.")
            return

        os.makedirs(folder_path, exist_ok=True)

        # 1. Save FAISS
        self.vector_store.save_local(folder_path)

        # 2. Save chunks
        with open(os.path.join(folder_path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info(f"✅ Saved FAISS + chunks at {folder_path}")

    # ================= LOAD =================
    def load_local_index(self, folder_path: str = "faiss_index"):
        try:
            # 1. Load FAISS
            self.vector_store = FAISS.load_local(
                folder_path,
                self.embedder,
                allow_dangerous_deserialization=True
            )

            # 2. Load chunks
            chunk_path = os.path.join(folder_path, "chunks.pkl")
            if os.path.exists(chunk_path):
                with open(chunk_path, "rb") as f:
                    self.chunks = pickle.load(f)
            else:
                logger.warning("⚠️ chunks.pkl not found!")

            logger.info("✅ Loaded FAISS + chunks successfully")

            return self.vector_store.as_retriever(search_kwargs={"k": 3})

        except Exception as e:
            logger.error(f"❌ Error loading index: {e}")
            return None


# ================= HELPER =================
def create_vector_store(chunks):
    engine = VectorEngine()
    engine.create(chunks)
    return engine.vector_store