import sys
import os
import time
import json
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')

# Ensure the root of the project is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

load_dotenv()

from src.rag.chain import conditional_generate, retrieve_and_gate

# The 10 evaluation questions
QUESTIONS = [
    # Direct factual
    "What are the common symptoms of diabetes?",
    "How is dengue virus transmitted?",
    "What is the definition of hypertension according to WHO?",
    
    # Requiring info from different sections / multiple chunks
    "How can asthma be managed and what are its common triggers?",
    "What are the risk factors for COPD and how can it be prevented?",
    "How does tuberculosis spread and what are its main symptoms?",
    
    # Multi-disease (might retrieve mixed chunks)
    "What are common noncommunicable diseases mentioned in the guidelines?",
    "Are there any vaccines mentioned for these respiratory diseases?",
    
    # Out of scope (knowledge base does NOT contain enough information)
    "What is the recommended treatment for malaria?",
    "How should a broken femur be treated?"
]

def run_evaluation():
    print("=========================================")
    print("          RAG BASELINE EVALUATION        ")
    print("=========================================\n")
    
    results = []
    
    for i, q in enumerate(QUESTIONS, 1):
        print(f"--- Question {i}/10 ---")
        print(f"Q: {q}")
        
        start_time = time.time()
        
        # We manually call retrieve_and_gate just to log the chunks and scores before generation
        # (conditional_generate calls it internally again, which is slightly inefficient for a test but keeps the test decoupled from internal refactors)
        retrieval = retrieve_and_gate(q)
        
        if retrieval["context"] == "GATE_REJECTED":
            print("[Retrieval] Gate rejected the query (insufficient relevance).")
            # We still call conditional_generate to get the final output logic
            
        gen_start = time.time()
        final_res = conditional_generate({"question": q})
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"\n[Answer] (Latency: {latency:.2f}s)")
        print(final_res["response"])
        
        print("\n[Extracted Sources]")
        print(json.dumps(final_res["sources"], indent=2))
        print("\n" + "="*40 + "\n")
        
        results.append({
            "question": q,
            "latency": latency,
            "retrieved_chunks_count": len(retrieval["docs"]) if retrieval["context"] != "GATE_REJECTED" else 0,
            "response": final_res["response"],
            "sources": final_res["sources"]
        })

if __name__ == "__main__":
    run_evaluation()
