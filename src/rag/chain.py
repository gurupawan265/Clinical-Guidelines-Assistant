"""
Basic retrieval-augmented generation (RAG) chain.
"""
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src.rag.vector_store import get_retriever, get_embeddings_model, CHROMA_DB_DIR
from langchain_chroma import Chroma

# 1. Define the LLM (Using Hugging Face local pipeline)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.1, # Low temperature for strict factual answers
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# 2. Define the strict prompt template with instructions for citations
PROMPT_TEMPLATE = """<|im_start|>system
You are a clinical assistant. 
Use ONLY the provided context. If the answer is not present in the context, say 'I don't know'. Do not hallucinate.<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}<|im_end|>
<|im_start|>assistant
"""

prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

# 3. Retrieval Gate Component
vectorstore = Chroma(
    collection_name="clinical_guidelines",
    embedding_function=get_embeddings_model(),
    persist_directory=CHROMA_DB_DIR
)

RELEVANCE_THRESHOLD = 1.2  # L2 distance: lower is better.
DISCLAIMER = "\n\n**Medical information disclaimer:** This response is for general informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment."

def retrieve_and_gate(query: str) -> dict:
    """
    Performs similarity search with scores.
    If the best score > RELEVANCE_THRESHOLD, returns a rejection flag.
    Otherwise, formats and returns the context.
    """
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=4)
    
    print(f"\n[Gate] Scores for '{query}':")
    for doc, score in docs_and_scores:
        print(f"  -> Score: {score:.4f} | {doc.page_content[:40].replace('\n', ' ')}...")
        
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

def format_sources(docs) -> str:
    if not docs: 
        return ""
    sources = []
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown Source")
        url = doc.metadata.get("url", "")
        sources.append(f"* {source_name} — {url}")
    
    # Deduplicate
    unique_sources = list(dict.fromkeys(sources))
    return "\n\n**Sources**\n" + "\n".join(unique_sources)

def conditional_generate(inputs: dict) -> str:
    """
    Checks if the gate rejected the query. If so, returns fallback.
    Otherwise, formats the prompt, calls the LLM, checks output guardrails, and adds disclaimer + sources.
    """
    query = inputs["question"]
    retrieval = retrieve_and_gate(query)
    
    if retrieval["context"] == "GATE_REJECTED":
        return "I don't have enough relevant information in my clinical guideline sources to answer that reliably. Please consult a qualified healthcare professional for guidance specific to your situation."
    
    prompt_val = prompt.invoke({"context": retrieval["context"], "question": query})
    raw_response = llm.invoke(prompt_val)
    
    safe_response = output_guardrail(raw_response)
    
    if "blocked because it contains specific medication dosages" in safe_response:
        return safe_response
    
    final_response = safe_response.strip() + DISCLAIMER + format_sources(retrieval["docs"])
    return final_response

def get_rag_chain():
    """
    Builds the simple chain with a relevance gate.
    """
    rag_chain = (
        {"question": RunnablePassthrough()}
        | RunnableLambda(conditional_generate)
    )
    return rag_chain
