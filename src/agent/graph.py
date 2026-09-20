from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.chain import llm, get_rag_chain

# 1. Define State
class AgentState(TypedDict):
    query: str
    route: str
    response: str

# 2. Router Node
ROUTER_TEMPLATE = """<|im_start|>system
You are a clinical routing assistant. Classify the user's input into EXACTLY ONE of the following three categories. Output ONLY the category name (either 'emergency', 'out-of-scope', or 'general-info').

Categories:
1. emergency - The user mentions chest pain, severe bleeding, inability to breathe, suicidal thoughts, or other immediate life-threatening situations.
2. out-of-scope - The user asks for personalized dosing, a specific medical diagnosis, or topics unrelated to clinical guidelines.
3. general-info - The user asks standard queries about disease mechanisms, symptoms, or guidelines.

Examples:
- "My dad just collapsed and is clutching his chest, what do I do?" -> emergency
- "I cut my arm deeply and the bleeding won't stop, help!" -> emergency
- "I feel like I'm having an asthma attack and my inhaler is empty." -> emergency
- "I am 180lbs, how many units of Humalog should I take for a 50g carb meal?" -> out-of-scope
- "I have a red rash on my arm that feels warm, is it an infection?" -> out-of-scope
- "What is the best over-the-counter painkiller for a sprained ankle?" -> out-of-scope
- "What are the common symptoms of asthma?" -> general-info
- "How is hypertension diagnosed?" -> general-info
- "Can type 2 diabetes be managed with diet alone?" -> general-info
- "Should I take 10mg or 20mg of lisinopril for my blood pressure?" -> out-of-scope
- "My blood sugar is 400 and I am vomiting, what should I do?" -> emergency
- "What is the recommended target HbA1c for adults with diabetes?" -> general-info
<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
router_prompt = PromptTemplate.from_template(ROUTER_TEMPLATE)

def route_query(state: AgentState):
    classifier = router_prompt | llm | StrOutputParser()
    result = classifier.invoke({"query": state["query"]}).strip().lower()
    
    if "emergency" in result:
        route = "emergency"
    elif "out-of-scope" in result:
        route = "out-of-scope"
    else:
        route = "general-info"
    
    print(f"\n[Graph Router] Analyzed query, selecting route: {route}")
    return {"route": route}

# 3. Conditional Edge Logic
def decide_next_node(state: AgentState):
    return state["route"]

# 4. Leaf Nodes
def handle_emergency(state: AgentState):
    response = ("\u26a0\ufe0f This may be a medical emergency. Please stop using this assistant "
                "and contact your local emergency medical service or go to the nearest "
                "emergency department immediately. This assistant cannot assess or manage medical emergencies.")
    return {"response": response}

def handle_out_of_scope(state: AgentState):
    response = ("I cannot provide personalized medical advice, dosing, or diagnoses. "
                "I can only provide general clinical guidelines. Please consult a qualified healthcare provider.")
    return {"response": response}

def handle_general_info(state: AgentState):
    print("[Graph Node] Proceeding to RAG chain...")
    chain = get_rag_chain()
    response = chain.invoke(state["query"])
    return {"response": response}

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", route_query)
workflow.add_node("emergency", handle_emergency)
workflow.add_node("out-of-scope", handle_out_of_scope)
workflow.add_node("general-info", handle_general_info)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "emergency": "emergency",
        "out-of-scope": "out-of-scope",
        "general-info": "general-info"
    }
)

workflow.add_edge("emergency", END)
workflow.add_edge("out-of-scope", END)
workflow.add_edge("general-info", END)

app = workflow.compile()

if __name__ == "__main__":
    queries = [
        # Emergency
        "My dad just collapsed and is clutching his chest, what do I do?",
        "I cut my arm deeply and the bleeding won't stop, help!",
        "I feel like I'm having an asthma attack and my inhaler is empty.",
        # Out-of-scope
        "I am 180lbs, how many units of Humalog should I take for a 50g carb meal?",
        "I have a red rash on my arm that feels warm, is it an infection?",
        "What is the best over-the-counter painkiller for a sprained ankle?",
        # General-info
        "What are the common symptoms of asthma?",
        "How is hypertension diagnosed?",
        "Can type 2 diabetes be managed with diet alone?"
    ]
    
    for q in queries:
        print(f"\n{'='*60}\nUSER: {q}")
        output = app.invoke({"query": q})
        print(f"\nAGENT:\n{output['response']}")
