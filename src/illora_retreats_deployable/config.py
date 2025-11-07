import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Central configuration for AI Chieftain
    """

    GSHEET_WEBAPP_URL = "https://script.google.com/macros/s/AKfycbxQjqqC_KM-zKlXAf2fs6B3jUjBBvuIES0a2VA4guZP0rZMR7A8JJGxDIUEzmcSZWFJ/exec"
    GSHEET_QNA_SHEET = "QnA_Manager"
    GSHEET_DOS_SHEET = "Dos and Donts"
    GSHEET_CAMPAIGN_SHEET = "Campaigns_Manager"

    # ------------------------
    # LLM Provider (switch between "openai" and "groq")
    # ------------------------
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

    # ------------------------
    # OpenAI (GPT models & embeddings)
    # ------------------------
    OPENAI_API_KEY = "REMOVED_GROQ_KEY"
    OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")  
    OPENAI_EMBEDDING_MODEL = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

    # ------------------------
    # Groq (fallback)
    # ------------------------
    GROQ_API_KEY   = "REMOVED_GROQ_KEY"
    GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_API_BASE  = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

    # ------------------------
    # Stripe Payments
    # ------------------------
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

    # ------------------------
    # Data paths
    # ------------------------
    CSV_DATA_PATH = os.getenv("CSV_DATA_PATH", "data\\qa_pairs.csv")

    
    # ------------------------
    # QNA generation
    # ------------------------
    
    RAW_DOCS_DIR = "data\\raw_docs"
    SUMMARY_OUTPUT_PATH =  "data\\combined_summary.txt"
    QA_OUTPUT_CSV =  "data\\qa_pairs.csv"
    UPLOAD_TEMP_DIR = "Hotel_docs"

    # Model / API config
    MAX_SUMMARY_TOKENS = int(os.getenv("MAX_SUMMARY_TOKENS", "500"))
    QA_PAIR_COUNT = int(os.getenv("QA_PAIR_COUNT", "100"))

    # --------------------------
    # Github Token for AI use
    # --------------------------

    endpoint = "https://models.github.ai/inference"
    model = "openai/gpt-5"
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

    # ------------------------
    # Property Images
    # ------------------------
    PROPERTY_IMAGES = [
        "https://ilora-retreats.com/mara/wp-content/uploads/2024/07/Luxury-Masai-Mara-Camps-in-the-Iconic-National-Reserve.jpg",
        "https://cf.bstatic.com/xdata/images/hotel/max200/701775390.jpg?k=7b58d29883aebb2a08b48977a2de7189a5ce69d84aee141b2842190f288d0a06&o=&hp=1",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWIA4IOIX3x2cbfhiK9GgkECyi77Lj682ipQ&s",
        "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/2c/eb/aa/b7/caption.jpg?w=900&h=500&s=1",
        "https://cf.bstatic.com/xdata/images/hotel/max1024x768/604728244.jpg?k=699c8a171b7fa1ce928c65228f9be210bb14b86b576894495822a0e376d258a2&o=&hp=1",
        "https://cf.bstatic.com/xdata/images/hotel/max1024x768/701805371.jpg?k=8e679865b1ee8ebd9e495c0ed1ac0d069db7dbc37073f8d246b5f7857d208871&o=&hp=1",
        "https://media-cdn.tripadvisor.com/media/photo-m/1280/2e/1f/c1/86/caption.jpg",
        "https://cf.bstatic.com/xdata/images/hotel/max1024x768/604728250.jpg?k=2b78cde6f5024be8c32d599bc7b2fc88214df23965d287d478679e7109825f9b&o=&hp=1",
    ]








