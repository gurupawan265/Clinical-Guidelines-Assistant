import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agent.graph import app

queries = [
    "What are common symptoms of asthma?",
    "What about children?",
    "What are the common triggers?",
    "What about severe difficulty breathing right now?"
]

print("=== Phase 6: Multi-turn Conversation Support ===\n")
state = {"chat_history": []}

for i, q in enumerate(queries, 1):
    print(f"--- Turn {i} ---")
    print(f"Original query: {q}")
    state["query"] = q
    state = app.invoke(state)
    print(f"Route: {state['route']}")
    if state['route'] == 'general-info' and 'reformulated_query' in state:
        print(f"Reformulated query: {state['reformulated_query']}")
    print(f"Final response:\n{state['response']}\n")
