import os
import shutil
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings # Changed from OpenAI
from langchain_community.vectorstores import FAISS
from config import Config

# 1. Define Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
csv_path = os.path.join(data_dir, "qa_pairs.csv")
index_path = os.path.join(data_dir, "vector_store")

# 2. Check if CSV exists
if not os.path.exists(csv_path):
    print(f"❌ ERROR: File not found at: {csv_path}")
    exit()

print(f"✅ Found CSV at: {csv_path}")

# 3. Clean old brain
if os.path.exists(index_path):
    shutil.rmtree(index_path)
    print("🗑️  Deleted old brain (cache).")

# 4. Load Data
try:
    print("⏳ Reading CSV...")
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} questions.")
except Exception as e:
    print(f"❌ CSV Error: {e}")
    exit()

# 5. Build New Brain
try:
    print("🧠 Building new brain with Local Embeddings...")
    # Use the same free local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(index_path)
    print(f"✅ SUCCESS! Brain saved to: {index_path}")
except Exception as e:
    print(f"❌ Build Error: {e}")