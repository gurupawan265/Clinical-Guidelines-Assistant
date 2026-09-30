"""
Chroma initialization, indexing, and retrieval logic.
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.rag.document_loader import ingest_documents

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

_embeddings_model = None
_vectorstore = None

def get_embeddings_model():
    """Returns the cached HuggingFace embeddings model."""
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_model

def get_vector_store() -> Chroma:
    """Returns the cached Chroma vector store instance."""
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
            raise FileNotFoundError(
                "Chroma database not found. Run:\npython src/rag/vector_store.py"
            )
        embeddings = get_embeddings_model()
        _vectorstore = Chroma(
            collection_name="clinical_guidelines",
            embedding_function=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
    return _vectorstore

import hashlib

def generate_chunk_id(doc: Document, idx: int) -> str:
    url = doc.metadata.get("url", "")
    topic = doc.metadata.get("topic", "")
    start_index = doc.metadata.get("start_index", idx)
    content = doc.page_content
    raw_key = f"{topic}::{url}::{start_index}::{content}"
    return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

def build_vector_store() -> Chroma:
    """
    Ingests documents, chunks them, and stores them in a local Chroma vector database using deterministic chunk IDs.
    """
    print("Fetching and ingesting raw documents...")
    raw_docs = ingest_documents()
    
    print(f"\nSplitting {len(raw_docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True  # Helpful for debugging where the chunk came from
    )
    
    chunked_docs = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunked_docs)} chunks from {len(raw_docs)} raw documents.")
    
    print("Initializing embedding model and Chroma DB...")
    embeddings = get_embeddings_model()
    
    chunk_ids = [generate_chunk_id(doc, i) for i, doc in enumerate(chunked_docs)]
    
    # Initialize DB (it loads the existing one if present)
    vectorstore = Chroma(
        collection_name="clinical_guidelines",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # Delete the collection to clear out old/stale chunks
    vectorstore.delete_collection()
    
    # Reinitialize a fresh collection
    vectorstore = Chroma(
        collection_name="clinical_guidelines",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    vectorstore.add_documents(documents=chunked_docs, ids=chunk_ids)
    print("Chunks successfully embedded and stored in Chroma with deterministic IDs.")
    
    return vectorstore

def get_retriever():
    """Returns a retriever interface for the vector store."""
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        raise FileNotFoundError(
            "Chroma database not found. Run:\npython src/rag/vector_store.py"
        )
    embeddings = get_embeddings_model()
    vectorstore = Chroma(
        collection_name="clinical_guidelines",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    # Return the top 4 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 4})

if __name__ == "__main__":
    # Running this file directly builds the database
    build_vector_store()
