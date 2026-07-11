"""
span  -> 1 unit of execution time 

trace -> a collection of spans that share the same trace_id, representing a single request or operation

waterfall -> a collection of traces that share the same waterfall_id, representing a single user session or workflow

"""

import logfire
import os
from dotenv import load_dotenv

load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response
from app.agents.graph import rag_agent

from pydantic import BaseModel
from typing import Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API")


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


def _state_summary(state: dict) -> dict:
    """Return compact state details for debug logs without dumping full documents."""

    return {
        "current_query": state.get("current_query"),
        "messages_count": len(state.get("messages", [])),
        "documents_count": len(state.get("documents", [])),
        "plan": state.get("plan", []),
        "status": state.get("status"),
        "final_answer": state.get("final_answer"),
    }
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": "",
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
            logfire.info(f"FastAPI Initial State: {_state_summary(initial_state)}")
            final_output = rag_agent.invoke(initial_state, config=config)
            logfire.info(f"FastAPI Final Graph Output: {_state_summary(final_output)}")

            return {
                "question": q,
                "answer": final_output.get("final_answer"),
                "thought_process": final_output.get("plan"),
                "status": final_output.get("status"),
                "sources": final_output.get("documents", [])
            }

    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": []
        }
