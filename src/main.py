import os
from dotenv import load_dotenv, find_dotenv

# Ensure environment variables are loaded securely from the root .env before importing agent logic
load_dotenv(find_dotenv())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

from src.agent.graph import app as workflow_app

app = FastAPI(title="Clinical Guidelines Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://0.0.0.0:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dictionary for conversation history
# conversation_id -> list of chat messages
conversations: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    message: str
    conversation_id: str

class Source(BaseModel):
    title: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    route: str
    sources: List[Source]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message or not req.conversation_id:
        raise HTTPException(status_code=400, detail="message and conversation_id are required")
    
    # Retrieve existing history
    chat_history = conversations.get(req.conversation_id, [])
    
    # Invoke LangGraph
    try:
        result = workflow_app.invoke({
            "query": req.message,
            "chat_history": chat_history
        })
    except RuntimeError as e:
        if "temporarily unavailable" in str(e):
            return ChatResponse(
                answer=str(e),
                route="general-info",
                sources=[]
            )
        print(f"Workflow error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing the request.")
    except Exception as e:
        # Avoid leaking stack traces to the frontend
        print(f"Workflow error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred while processing the request.")
    
    # Extract values from state
    answer = result.get("response", "No response generated.")
    route = result.get("route", "general-info")
    sources = result.get("sources", [])
    
    # Update our in-memory store using the updated history from graph
    conversations[req.conversation_id] = result.get("chat_history", chat_history)
    
    return ChatResponse(
        answer=answer,
        route=route,
        sources=[Source(title=s.get("title", ""), url=s.get("url", "")) for s in sources]
    )

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
