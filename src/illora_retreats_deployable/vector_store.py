import os
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from logger import setup_logger

logger = setup_logger("VectorStoreService")

# Global variable to hold the loaded brain
_VECTOR_STORE = None

def create_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    csv_path = os.path.join(data_dir, "qa_pairs.csv")
    index_path = os.path.join(data_dir, "vector_store")

    # 2. Hardcode the Model (Crucial for consistency)
    print(f"[DEBUG] Loading Embedding Model: all-MiniLM-L6-v2 ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Try to Load Existing Brain
    if os.path.exists(index_path):
        try:
            print(f"[DEBUG] Loading existing brain from: {index_path}")
            _VECTOR_STORE = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            return _VECTOR_STORE
        except Exception as e:
            print(f"[WARN] Could not load existing brain: {e}")

    # 4. Fallback: Build from CSV on the fly
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found at {csv_path}. Returning empty brain.")
        return FAISS.from_texts([""], embeddings)

    print(f"[DEBUG] Building brain from CSV: {csv_path}")
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    docs = loader.load()
    
    _VECTOR_STORE = FAISS.from_documents(docs, embeddings)
    return _VECTOR_STORE