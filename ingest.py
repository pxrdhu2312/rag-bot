"""
ingest.py
---------
Reads company_faq.txt, splits it into overlapping chunks, generates local
embeddings using a HuggingFace sentence-transformer model, and persists
them into a local FAISS vector store on disk.

Run this once (and again any time company_faq.txt changes):
    python ingest.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAQ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company_faq.txt")
VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():
    if not os.path.exists(FAQ_PATH):
        raise FileNotFoundError(f"FAQ document not found at {FAQ_PATH}")

    print("Loading FAQ document...")
    loader = TextLoader(FAQ_PATH, encoding="utf-8")
    documents = loader.load()

    print("Chunking document...")
    # Chunk size / overlap chosen so each chunk roughly covers one FAQ
    # sub-section (headings act as natural chunk boundaries) while overlap
    # prevents an answer from being split across two chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME} (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Vector store saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
