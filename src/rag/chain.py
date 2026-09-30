"""
Basic retrieval-augmented generation (RAG) chain.
"""
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.rag.vector_store import get_retriever, get_embeddings_model, get_vector_store, CHROMA_DB_DIR
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1,
    max_tokens=512,
)

# 2. Define the strict prompt template with instructions for citations
PROMPT_TEMPLATE = """<|im_start|>system
You are a clinical guidelines assistant. 
Your primary task is to answer the user's question based strictly on the provided Context.

RULES:
1. Use ONLY the provided Context. Do not rely on your general knowledge.
2. If the Context does not contain enough information to fully answer the question, say "I don't know".
3. Never fabricate or hallucinate medical facts, treatments, or statistics.
4. Be concise and direct. Do not get cut off.
5. If the Context is completely irrelevant, you MUST output exactly "I don't know." and nothing else.
<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}<|im_end|>
<|im_start|>assistant
"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

# 3. Retrieval Gate Component
RELEVANCE_THRESHOLD = 1.2  # L2 distance: lower is better.
DISCLAIMER = "\n\n**Medical information disclaimer:** This response is for general informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment."

def retrieve_and_gate(query: str) -> dict:
    """
    Performs similarity search with scores.
    If the best score > RELEVANCE_THRESHOLD, returns a rejection flag.
    Otherwise, formats and returns the context.
    """
    vectorstore = get_vector_store()
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=4)
    
    print(f"\n[Gate] Scores for '{query}':")
    for doc, score in docs_and_scores:
        print(f"  -> Score: {score:.4f} | {doc.page_content[:40].replace(chr(10), ' ')}...")
        
    best_score = docs_and_scores[0][1] if docs_and_scores else float('inf')
    if best_score > RELEVANCE_THRESHOLD:
        print(f"[Gate] Best score {best_score:.4f} exceeds threshold {RELEVANCE_THRESHOLD}. Rejecting.")
        return {"context": "GATE_REJECTED", "docs": []}
        
    formatted_texts = []
    for i, (doc, _) in enumerate(docs_and_scores, 1):
        content = doc.page_content.strip()
        formatted_texts.append(content)
    return {"context": "\n\n".join(formatted_texts), "docs": [d for d, s in docs_and_scores]}

def output_guardrail(text: str) -> str:
    dosage_pattern = r'\d+\.?\d*\s*(mg|mcg|ml|g|units|iu)\b'
    personalized_pattern = r'\b(you should take|your treatment plan|I recommend that you|prescribe)\b'
    
    if re.search(dosage_pattern, text, re.IGNORECASE) or re.search(personalized_pattern, text, re.IGNORECASE):
        return "This response has been blocked because it contains specific medication dosages or personalized treatment recommendations, which this assistant is not authorized to provide."
    return text

def extract_sources_list(docs) -> list:
    if not docs: 
        return []
    sources_dict = {}
    for doc in docs:
        title = doc.metadata.get("title", doc.metadata.get("topic", "Unknown Source"))
        url = doc.metadata.get("url", "")
        # Use URL as key to deduplicate
        key = url if url else title
        if key not in sources_dict:
            sources_dict[key] = {"title": title, "url": url}
    
    return list(sources_dict.values())

def conditional_generate(inputs: dict) -> dict:
    """
    Checks if the gate rejected the query. If so, returns fallback.
    Otherwise, formats the prompt, calls the LLM, checks output guardrails, and adds disclaimer + sources.
    Returns a dictionary with 'response' and 'sources' list.
    """
    query = inputs["question"]
    retrieval = retrieve_and_gate(query)
    
    if retrieval["context"] == "GATE_REJECTED":
        return {
            "response": "I don't have enough relevant information in my clinical guideline sources to answer that reliably. Please consult a qualified healthcare professional for guidance specific to your situation.",
            "sources": []
        }
    
    prompt_val = prompt.invoke({"context": retrieval["context"], "question": query})
    raw_response = llm.invoke(prompt_val).content
    
    safe_response = output_guardrail(raw_response).strip()
    
    if "blocked because it contains specific medication dosages" in safe_response:
        return {
            "response": safe_response,
            "sources": []
        }
        
    if not safe_response or "i don't know" in safe_response.lower():
        return {
            "response": "I don't have enough relevant information in my clinical guideline sources to answer that reliably. Please consult a qualified healthcare professional for guidance specific to your situation.",
            "sources": []
        }
    
    final_response = safe_response + DISCLAIMER
    return {
        "response": final_response,
        "sources": extract_sources_list(retrieval["docs"])
    }

def get_rag_chain():
    """
    Builds the simple chain with a relevance gate.
    """
    rag_chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(conditional_generate)
    )
    return rag_chain
