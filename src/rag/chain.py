"""
Basic retrieval-augmented generation (RAG) chain.
"""

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
Use ONLY the provided context. If the answer is not present in the context, say 'I don't know'. Do not hallucinate.

When you use information from the context, you MUST include an inline citation to the source using the bracketed number provided, for example [1] or [2].<|im_end|>
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

def retrieve_and_gate(query: str) -> str:
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
        return "GATE_REJECTED"
        
    formatted_texts = []
    for i, (doc, _) in enumerate(docs_and_scores, 1):
        source = doc.metadata.get("url", "Unknown Source")
        content = doc.page_content.strip()
        formatted_texts.append(f"Source [{i}] ({source}):\n{content}\n")
    return "\n".join(formatted_texts)

def conditional_generate(inputs: dict) -> str:
    """
    Checks if the gate rejected the query. If so, returns 'I don't know.'
    Otherwise, formats the prompt and calls the LLM.
    """
    if inputs["context"] == "GATE_REJECTED":
        return "I don't know."
    
    prompt_val = prompt.invoke(inputs)
    return llm.invoke(prompt_val)

def get_rag_chain():
    """
    Builds the simple chain with a relevance gate.
    """
    rag_chain = (
        {"context": RunnableLambda(retrieve_and_gate), "question": RunnablePassthrough()}
        | RunnableLambda(conditional_generate)
    )
    return rag_chain

if __name__ == "__main__":
    chain = get_rag_chain()
    # Simple interactive test loop
    print("\n--- Basic RAG Chain ---")
    while True:
        query = input("Ask a medical question (or 'q' to quit): ")
        if query.lower() in ['q', 'quit']:
            break
        print("Thinking...")
        response = chain.invoke(query)
        print("\nAnswer:\n" + response)
        print("-" * 50)
