import re, tiktoken
from typing import List, Dict
from config import CHUNK_TOKENS, CHUNK_OVERLAP
import pdfplumber, pathlib


enc = tiktoken.get_encoding("cl100k_base")

def tokenize(text: str) -> List[int]:
    return enc.encode(text)

def count_tokens(text: str) -> int:
    return len(tokenize(text))

def hard_chunk(text: str, size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap

    return chunks



def load_pdf(path: pathlib.Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

# splits text into sections based on headlines or blank lines or paragraph breaks
def smart_split(raw_text: str) -> List[str]:
    """
    Split on (1) headline‑like lines, else (2) blank lines, else fallback paragraphs.
    """
    pieces = re.split(r"\n(?=[A-Z][A-Za-z .]{3,}\n)", raw_text)
    if len(pieces) == 1:
        pieces = raw_text.split("\n\n")    
    return [p.strip() for p in pieces if p.strip()]
