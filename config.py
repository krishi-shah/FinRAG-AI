"""
Configuration — all settings loaded from environment variables.
Copy `.env.example` to `.env` and fill in your keys.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# SEC EDGAR
SEC_BASE_URL = "https://www.sec.gov/Archives/"

# RAG Pipeline Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Retrieval mode: "dense", "sparse", "hybrid"
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))

# Reranking
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# Index persistence
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
