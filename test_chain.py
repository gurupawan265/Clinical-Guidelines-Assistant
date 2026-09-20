import sys
from src.rag.chain import get_rag_chain

queries = [
    'What are common symptoms of diabetes?',
    'What is hypertension?',
    'What are common symptoms of asthma?',
    'How is type 2 diabetes diagnosed?',
    'What is the recommended treatment for a broken tibia?'
]

try:
    chain = get_rag_chain()
    for q in queries:
        print(f"\nQUERY: {q}")
        print("Answer:", chain.invoke(q))
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
