"""
Logic for fetching, parsing, and tagging medical guideline documents.
"""

import os
import requests
from collections import Counter
from typing import List
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document

SOURCES = [
    {
        "topic": "Diabetes",
        "url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "type": "html",
        "source_name": "WHO"
    },
    {
        "topic": "Hypertension",
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "type": "html",
        "source_name": "WHO"
    },
    {
        "topic": "Asthma",
        "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "type": "pdf",
        "source_name": "NIH/CDC"
    }
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def download_pdf(url: str, filename: str) -> str:
    """Downloads a PDF from a URL to the data directory."""
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    return filepath

def ingest_documents() -> List[Document]:
    """
    Fetches documents from defined sources, loads them into LangChain Document objects,
    and tags them with metadata.
    """
    all_docs = []
    failures = []
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for source in SOURCES:
        try:
            if source["type"] == "html":
                # Load HTML directly from URL
                loader = WebBaseLoader(source["url"])
                docs = loader.load()
            elif source["type"] == "pdf":
                # Download PDF first, then load
                filename = f"{source['topic'].lower()}_guideline.pdf"
                filepath = download_pdf(source["url"], filename)
                loader = PyPDFLoader(filepath)
                docs = loader.load()
            else:
                raise ValueError(f"Unknown source type: {source['type']}")
                
            # Tag metadata
            for doc in docs:
                doc.metadata["topic"] = source["topic"]
                doc.metadata["url"] = source["url"]
                doc.metadata["source_name"] = source["source_name"]
                doc.metadata["content_type"] = source["type"]
                
            all_docs.extend(docs)
            print(f"Successfully loaded {source['topic']} ({len(docs)} documents/pages)")
            
        except Exception as e:
            print(f"Failed to load {source['topic']}: {e}")
            failures.append({"topic": source["topic"], "error": str(e)})

    # Print Summary
    print("\n--- Ingestion Summary ---")
    print(f"Total Documents Loaded: {len(all_docs)}")
    
    topic_counts = Counter(doc.metadata.get("topic", "Unknown") for doc in all_docs)
    print("Topic Breakdown:")
    for topic, count in topic_counts.items():
        print(f"  - {topic}: {count} docs/pages")
        
    if failures:
        print("Failures:")
        for f in failures:
            print(f"  - {f['topic']}: {f['error']}")
    else:
        print("Failures: 0")
        
    return all_docs

if __name__ == "__main__":
    ingest_documents()
