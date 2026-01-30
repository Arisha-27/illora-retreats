import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Central configuration for AI Chieftain
    """

    # ------------------------
    # Google Sheets
    # ------------------------
    GSHEET_WEBAPP_URL = os.getenv("GSHEET_WEBAPP_URL")
    GSHEET_QNA_SHEET = "QnA_Manager"
    GSHEET_DOS_SHEET = "Dos and Donts"
    GSHEET_CAMPAIGN_SHEET = "Campaigns_Manager"

    # ------------------------
    # LLM Provider
    # ------------------------
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

    # ------------------------
    # OpenAI
    # ------------------------
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
    OPENAI_EMBEDDING_MODEL = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # ------------------------
    # Groq (fallback)
    # ------------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_API_BASE = os.getenv(
        "GROQ_API_BASE", "https://api.groq.com/openai/v1"
    )

    # ------------------------
    # Stripe
    # ------------------------
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

    # ------------------------
    # Paths
    # ------------------------
    CSV_DATA_PATH = os.getenv("CSV_DATA_PATH", "data/qa_pairs.csv")
    RAW_DOCS_DIR = os.getenv("RAW_DOCS_DIR", "data/raw_docs")
    SUMMARY_OUTPUT_PATH = os.getenv(
        "SUMMARY_OUTPUT_PATH", "data/combined_summary.txt"
    )
    QA_OUTPUT_CSV = os.getenv("QA_OUTPUT_CSV", "data/qa_pairs.csv")
    UPLOAD_TEMP_DIR = os.getenv("UPLOAD_TEMP_DIR", "Hotel_docs")

    # ------------------------
    # QnA generation
    # ------------------------
    MAX_SUMMARY_TOKENS = int(os.getenv("MAX_SUMMARY_TOKENS", "500"))
    QA_PAIR_COUNT = int(os.getenv("QA_PAIR_COUNT", "100"))

    # ------------------------
    # GitHub Models (optional)
    # ------------------------
    GITHUB_ENDPOINT = "https://models.github.ai/inference"
    GITHUB_MODEL = "openai/gpt-5"
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

    # ------------------------
    # Property Images (safe to commit)
    # ------------------------
    PROPERTY_IMAGES = [
        "https://ilora-retreats.com/mara/wp-content/uploads/2024/07/Luxury-Masai-Mara-Camps-in-the-Iconic-National-Reserve.jpg",
        "https://cf.bstatic.com/xdata/images/hotel/max200/701775390.jpg",
        "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/2c/eb/aa/b7/caption.jpg",
    ]
