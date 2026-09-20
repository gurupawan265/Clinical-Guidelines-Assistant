"""
A small script to manually sanity-check retrieval quality from the Chroma DB.
"""
import sys
from src.rag.vector_store import get_retriever

def main():
    print("\n--- RAG Retrieval Sanity Check ---")
    query = input("Enter a health question to check retrieval (or 'q' to quit): ")
    if query.lower() in ['q', 'quit', 'exit']:
        sys.exit(0)
        
    print(f"\nSearching for: '{query}'...\n")
    retriever = get_retriever()
    results = retriever.invoke(query)
    
    print(f"--- Top {len(results)} Chunks Retrieved ---\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Topic: {doc.metadata.get('topic', 'Unknown')} | Source: {doc.metadata.get('source_name', 'Unknown')} ({doc.metadata.get('content_type', 'Unknown')})")
        print(f"URL: {doc.metadata.get('url', 'Unknown')}")
        # Print a snippet of the content
        snippet = doc.page_content[:300].replace('\n', ' ')
        print(f"Content snippet: {snippet}...")
        print("-" * 50)
        
if __name__ == "__main__":
    # To run this script:
    # set PYTHONPATH=. && python src/rag/retrieval_check.py
    while True:
        main()
