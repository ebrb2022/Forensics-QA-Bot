import re
from typing import List
from config import CHUNK_TOKENS, CHUNK_OVERLAP
import pdfplumber, pathlib


def hard_chunk(text: str, size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP):
    """
    just split text into chunks with some overlap
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap

    return chunks



def load_pdf(path: pathlib.Path) -> str:
    """
    load text from a PDF using pdfplumber
    """
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def load_text(path: pathlib.Path) -> str:
    """
    load text from a txt file
    """
    return path.read_text(encoding="utf-8")

# splits text into sections based on headlines or blank lines or paragraph breaks
def smart_split(raw_text: str) -> List[str]:
    """
    split on headline‑like lines, else blank lines, else fallback paragraphs
    """
    pieces = re.split(r"\n(?=[A-Z][A-Za-z .]{3,}\n)", raw_text)
    if len(pieces) == 1:
        pieces = raw_text.split("\n\n")    
    return [p.strip() for p in pieces if p.strip()]
