"""
Central config
"""
from pathlib import Path

# Chroma settings
CHROMA_DIR = Path(".chroma_persistent_db")
COLLECTION = "my_document_collection"

# HuggingFace embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# chunking
CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50
LOCAL_LLM = "microsoft/Phi-3-mini-4k-instruct"

# num of chunks to return each time (for RAG)
TOP_K = 4 

# Streamlit
APP_TITLE = "Forensics Q&A (RAG)"

