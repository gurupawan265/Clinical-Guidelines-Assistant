import sys
import os
from src.rag.vector_store import get_retriever, build_vector_store

if not os.path.exists('chroma_db'):
    build_vector_store()

queries = [
    'What are common symptoms of diabetes?',
    'What is hypertension?',
    'What are common symptoms of asthma?',
    'How is type 2 diabetes diagnosed?',
    'What can trigger asthma symptoms?'
]

retriever = get_retriever()
for query in queries:
    print(f"\n{'='*50}\nQUERY: {query}\n{'='*50}")
    results = retriever.invoke(query)
    for i, doc in enumerate(results, 1):
        print(f"Result {i}: Topic: {doc.metadata.get('topic')} | URL: {doc.metadata.get('url')}")
        snippet = doc.page_content[:200].replace('\n', ' ')
        print(f"Snippet: {snippet}...\n")
