from pathlib import Path
import os
from dotenv import load_dotenv
import chromadb
import chromadb.utils.embedding_functions as ef
from utils import load_pdf, load_text, smart_split, hard_chunk
from config import CHROMA_DIR, COLLECTION, EMBED_MODEL

# as usual, load .env file
load_dotenv()

hf = os.getenv("HF_TOKEN")

# base folder is where this file lives
BASE = Path(__file__).parent

# all the sources to be ingested
DOCS = [
    (BASE / "sources/Digital_Forensic_Models_Exam.pdf", "paper"),
    (BASE / "sources/Forensic_Science_Intro.pdf", "paper"),
    (BASE / "sources/Science_Criminal_Law.pdf", "textbook"),
    (BASE / "sources/Sequencing_Tech.pdf", "paper"),
    (BASE / "sources/Social_Process.pdf", "chapter"),
    (BASE / "sources/Digital_Forensic_Evidence.txt", "article")
]

# loads a PDF or text file
def read_text(path: Path) -> str:
    return load_pdf(path) if path.suffix.lower() == ".pdf" else load_text(path)

# reads txt files, splits into sections, creates chunks


def build_chunks():
    """ read each doc, split into sections, create overlapping chunks, return text metadata dicts """
    chunks = []
    for path, kind in DOCS:
        raw = read_text(path)
        for i, section in enumerate(smart_split(raw)):
            meta = {"source": path.name, "kind": kind, "section": i}
            for chunk in hard_chunk(section, size=500, overlap=50):
                chunks.append({
                    "text": chunk,
                    "metadata": meta
                })
    return chunks


# make a Chroma client + collection
def main() -> None:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    hf_ef = ef.SentenceTransformerEmbeddingFunction(
        model_name= EMBED_MODEL
    )

    try:
        chroma_client.delete_collection(COLLECTION)
        print("Deleted old collection.")
    except:
        pass
    collection = chroma_client.create_collection(
        COLLECTION, embedding_function=hf_ef
    )

    print("Created new collection.")

    data = build_chunks()
    print(f"Ingesting {len(data)} chunks …")
    collection.add(
        ids=[f"c{i}" for i in range(len(data))],
        documents=[d["text"]    for d in data],
        metadatas=[d["metadata"] for d in data],
    )
    print("Now you must run:  streamlit run app.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
