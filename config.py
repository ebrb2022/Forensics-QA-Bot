"""
Central config – adjust once, reuse everywhere.
"""
from pathlib import Path

# Chroma settings
CHROMA_DIR = Path(".chroma_persistent_db")
COLLECTION = "my_document_collection"

# HuggingFace embedding model
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# chunking
CHUNK_TOKENS = 650
CHUNK_OVERLAP = 50
LOCAL_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# num of chunks to return each time (for RAG)
TOP_K = 4 

# Streamlit
APP_TITLE = "Forensics Q&A (RAG)"
SYSTEM_PROMPT = """You are an expert assistant answering questions **only** 
about the supplied context chunks from the document collection regarding about forensics. Make sure to use the pdf file, which contains relevant forensic information.
If the question is completely out of scope, get more and more unhinged everytime, as if you are losing your mind. However, if the question contains words that you have in the documents, try your best to respond in a factual and informative manner."""
