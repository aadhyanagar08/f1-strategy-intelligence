"""
Ingest FIA regulation PDFs into ChromaDB.

Usage:
    pip install pypdf
    python ingest.py                        # ingests all PDFs from ./regulations/
    python ingest.py --dir /path/to/pdfs    # custom directory
"""

import argparse
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def _chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c]


def ingest(regulations_dir: str) -> None:
    from pypdf import PdfReader

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_or_create_collection(
        name="fia_regulations", embedding_function=DefaultEmbeddingFunction()
    )

    pdf_files = list(Path(regulations_dir).glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {regulations_dir}")
        return

    for pdf_path in pdf_files:
        print(f"Ingesting: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))
        doc_ids, texts, metas = [], [], []

        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for i, chunk in enumerate(_chunk_text(page_text)):
                chunk_id = f"{pdf_path.stem}_p{page_num}_c{i}"
                doc_ids.append(chunk_id)
                texts.append(chunk)
                metas.append({"source": pdf_path.name, "page": page_num})

        if texts:
            col.upsert(ids=doc_ids, documents=texts, metadatas=metas)
            print(f"  → {len(texts)} chunks indexed")

    print(f"\nDone. Collection '{col.name}' now has {col.count()} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest FIA PDFs into ChromaDB")
    parser.add_argument("--dir", default="./regulations", help="Directory containing PDFs")
    args = parser.parse_args()
    ingest(args.dir)
