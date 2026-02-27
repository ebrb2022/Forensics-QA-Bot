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
CHUNK_TOKENS = 1000
CHUNK_OVERLAP = 100

# the llm for answering questions
LOCAL_LLM = "meta-llama/Llama-3.2-3B-Instruct"

TOP_K = 7

APP_TITLE = "Forensics Q&A (RAG)"

