import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.vector_store import get_embeddings_model
from langchain_chroma import Chroma

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
embeddings = get_embeddings_model()
vectorstore = Chroma(
    collection_name="clinical_guidelines",
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_DIR
)

queries = [
    'What are common symptoms of diabetes?',
    'What is hypertension?',
    'What are common symptoms of asthma?',
    'How is type 2 diabetes diagnosed?',
    'What is the recommended treatment for a broken tibia?'
]

for q in queries:
    print(f"\nQUERY: {q}")
    docs_and_scores = vectorstore.similarity_search_with_score(q, k=4)
    for doc, score in docs_and_scores:
        print(f"Score: {score:.4f} | Topic: {doc.metadata.get('topic')} | Content: {doc.page_content[:50].replace('\n', ' ')}")
