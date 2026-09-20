from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.chain import llm, get_rag_chain

ROUTER_TEMPLATE = """<|im_start|>system
You are a clinical routing assistant. Classify the user's input into EXACTLY ONE of the following three categories. Output ONLY the category name (either 'emergency', 'out-of-scope', or 'general-info').

Categories:
1. emergency - The user mentions chest pain, severe bleeding, inability to breathe, suicidal thoughts, or other immediate life-threatening situations.
2. out-of-scope - The user asks for personalized dosing, a specific medical diagnosis, or topics unrelated to clinical guidelines.
3. general-info - The user asks standard queries about disease mechanisms, symptoms, or guidelines.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

router_prompt = PromptTemplate.from_template(ROUTER_TEMPLATE)

def classify_query(query: str) -> str:
    # 1. Classification Call
    classifier_chain = router_prompt | llm | StrOutputParser()
    print(f"\n[Router] Classifying query...")
    category = classifier_chain.invoke({"question": query}).strip().lower()
    
    # Clean up output in case the 1.5B LLM gets chatty
    if "emergency" in category:
        category = "emergency"
    elif "out-of-scope" in category:
        category = "out-of-scope"
    else:
        category = "general-info"
        
    print(f"[Router] Route selected: {category}")
    return category

def agent_workflow(query: str) -> str:
    """
    Main entry point for the agent. 
    It routes the query BEFORE any retrieval happens.
    """
    category = classify_query(query)
    
    if category == "emergency":
        return ("\u26a0\ufe0f **MEDICAL EMERGENCY ALERT**\n"
                "It sounds like you may be experiencing a medical emergency. "
                "Please stop using this assistant and immediately call 911 (or your local emergency number) "
                "or go to the nearest emergency room. This assistant cannot provide medical diagnoses or emergency guidance.")
    
    elif category == "out-of-scope":
        return ("I cannot provide personalized medical advice, dosing, or diagnoses. "
                "I can only provide general clinical guidelines. Please consult a qualified healthcare provider.")
    
    else:
        # Proceed to RAG pipeline
        print("[Router] Proceeding to RAG chain...")
        rag_chain = get_rag_chain()
        return rag_chain.invoke(query)

if __name__ == "__main__":
    queries = [
        "My dad just collapsed and is clutching his chest, what do I do?",
        "I am 180lbs, how many units of Humalog should I take for a 50g carb meal?",
        "What are the common symptoms of asthma?"
    ]
    
    for q in queries:
        print(f"\n{'='*50}\nUSER: {q}")
        response = agent_workflow(q)
        print(f"\nAGENT:\n{response}")
