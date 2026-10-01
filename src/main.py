import os
from dotenv import load_dotenv, find_dotenv

# Set threading and HuggingFace environment limits before importing heavy ML libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure environment variables are loaded securely from the root .env before importing agent logic
load_dotenv(find_dotenv())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import traceback

from src.agent.graph import app as workflow_app

app = FastAPI(title="Clinical Guidelines Assistant API")

# Configure CORS
# Allow setting CORS_ORIGINS from environment (e.g. "https://my-frontend.vercel.app,http://localhost:5173" or "*")
cors_origins_env = os.environ.get("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    allow_origins = ["*"]
    allow_credentials = False
else:
    allow_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
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

@app.get("/test-groq")
def test_groq():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        return {"status": "error", "message": "GROQ_API_KEY environment variable is missing on server"}
    try:
        from src.rag.chain import llm
        res = llm.invoke("Hi").content
        return {"status": "ok", "response": res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-chroma")
def test_chroma():
    try:
        from src.rag.vector_store import get_vector_store
        vs = get_vector_store()
        docs = vs.similarity_search("asthma", k=1)
        return {"status": "ok", "docs_count": len(docs)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
        print(f"[ERROR] Workflow RuntimeError: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    except Exception as e:
        print(f"[ERROR] Workflow Exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
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
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)

