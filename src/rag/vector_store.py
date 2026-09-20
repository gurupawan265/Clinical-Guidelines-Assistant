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

def get_embeddings_model():
    """Returns the HuggingFace embeddings model."""
    # Using all-MiniLM-L6-v2 as requested for fast, local, CPU-friendly embeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store() -> Chroma:
    """
    Ingests documents, chunks them, and stores them in a local Chroma vector database.
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
    
    # Initialize and populate the DB
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name="clinical_guidelines"
    )
    print("Chunks successfully embedded and stored in Chroma.")
    
    return vectorstore

def get_retriever():
    """Returns a retriever interface for the vector store."""
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
