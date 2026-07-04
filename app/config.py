import os 
from dotenv import load_dotenv 

load_dotenv()

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"

    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_URL = os.getenv("QDRANT_URL", QDRANT_CLUSTER_ENDPOINT)
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "KUB_RAG_Assistant")

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


settings = Settings()
