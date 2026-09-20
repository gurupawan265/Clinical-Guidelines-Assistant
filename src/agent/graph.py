from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.chain import llm, get_rag_chain

# 1. Define State
class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    route: str
    reformulated_query: str
    response: str

def format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    formatted = ""
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted

# 2. Router Node
ROUTER_TEMPLATE = """<|im_start|>system
You are a clinical routing assistant. Classify the user's latest input into EXACTLY ONE of the following three categories. Output ONLY the category name (either 'emergency', 'out-of-scope', or 'general-info').

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
{chat_history}<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
router_prompt = PromptTemplate.from_template(ROUTER_TEMPLATE)

def route_query(state: AgentState):
    history_str = format_history(state.get("chat_history", []))
    classifier = router_prompt | llm | StrOutputParser()
    result = classifier.invoke({
        "chat_history": history_str, 
        "query": state["query"]
    }).strip().lower()
    
    if "emergency" in result:
        route = "emergency"
    elif "out-of-scope" in result:
        route = "out-of-scope"
    else:
        route = "general-info"
    
    return {"route": route}

# 3. Conditional Edge Logic
def decide_next_node(state: AgentState):
    if state["route"] == "general-info":
        return "reformulate"
    return state["route"]

# 4. Nodes
def handle_emergency(state: AgentState):
    response = ("\u26a0\ufe0f This may be a medical emergency. Please stop using this assistant "
                "and contact your local emergency medical service or go to the nearest "
                "emergency department immediately. This assistant cannot assess or manage medical emergencies.")
    return {"response": response}

def handle_out_of_scope(state: AgentState):
    response = ("I cannot provide personalized medical advice, dosing, or diagnoses. "
                "I can only provide general clinical guidelines. Please consult a qualified healthcare provider.")
    return {"response": response}

REFORMULATE_TEMPLATE = """<|im_start|>system
You are a medical query reformulator. Given a conversation history and a new user query, rewrite the new user query to be a standalone query that can be used by a search engine to retrieve clinical guidelines.
If the new query is already standalone, just repeat it.

RULES:
- Output exactly ONE standalone search query.
- Do NOT answer the question.
- Do NOT provide medical facts.
- Do NOT provide citations.
- Do NOT add explanations.
- Preserve the user's intent; only resolve missing context from conversation history.

EXAMPLES:

Conversation:
User: What are common symptoms of asthma?
Assistant: Common asthma symptoms include cough, wheezing, shortness of breath, and chest tightness.
User: What about children?

Output:
What are common symptoms of asthma in children?

Conversation:
User: How is hypertension diagnosed?
Assistant: Hypertension is diagnosed using a sphygmomanometer over multiple readings.
User: What are the common triggers?

Output:
What are the common triggers of hypertension?

Conversation:
User: What is the target HbA1c for adults with diabetes?
Assistant: The target HbA1c is typically less than 7%.
User: How is type 2 diabetes diagnosed?

Output:
How is type 2 diabetes diagnosed?
<|im_end|>
{chat_history}<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
reformulate_prompt = PromptTemplate.from_template(REFORMULATE_TEMPLATE)

def reformulate_query(state: AgentState):
    history = state.get("chat_history", [])
    if not history:
        return {"reformulated_query": state["query"]}
    
    history_str = format_history(history)
    chain = reformulate_prompt | llm | StrOutputParser()
    rewritten = chain.invoke({
        "chat_history": history_str, 
        "query": state["query"]
    }).strip()
    return {"reformulated_query": rewritten}

def handle_general_info(state: AgentState):
    chain = get_rag_chain()
    req_query = state.get("reformulated_query", state["query"])
    response = chain.invoke(req_query)
    return {"response": response}

def update_history(state: AgentState):
    history = state.get("chat_history", [])
    # Only append if we actually have a valid query/response this turn
    if state.get("query") and state.get("response"):
        # Ensure we don't mutate the existing list object, create a new one to be safe with state
        new_history = list(history)
        new_history.append({"role": "user", "content": state["query"]})
        new_history.append({"role": "assistant", "content": state["response"]})
        
        # Keep last 4 turns (8 messages)
        if len(new_history) > 8:
            new_history = new_history[-8:]
        return {"chat_history": new_history}
    return {}

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", route_query)
workflow.add_node("emergency", handle_emergency)
workflow.add_node("out-of-scope", handle_out_of_scope)
workflow.add_node("reformulate", reformulate_query)
workflow.add_node("general-info", handle_general_info)
workflow.add_node("update_history", update_history)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "emergency": "emergency",
        "out-of-scope": "out-of-scope",
        "reformulate": "reformulate"
    }
)

workflow.add_edge("emergency", "update_history")
workflow.add_edge("out-of-scope", "update_history")
workflow.add_edge("reformulate", "general-info")
workflow.add_edge("general-info", "update_history")
workflow.add_edge("update_history", END)

app = workflow.compile()
