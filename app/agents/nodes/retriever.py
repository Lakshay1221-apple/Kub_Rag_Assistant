import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents

def _state_summary(state: AgentState) -> dict:
    """Return compact state details for debug logs without dumping full documents."""

    return {
        "current_query": state.get("current_query"),
        "messages_count": len(state.get("messages", [])),
        "documents_count": len(state.get("documents", [])),
        "plan": state.get("plan", []),
        "status": state.get("status"),
        "has_final_answer": bool(state.get("final_answer")),
    }


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    logfire.info(f"Retriever Node Input State: {_state_summary(state)}")
    
    
    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info(f"Searching Qdrant for: {query}")
        raw_results = search_enterprise_knowledge(query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} candidates from Vector DB")
        
        doc_contents = [doc['content'] for doc in raw_results]
        
        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")
            
        formatted_docs = [f"CONTENT: {doc}" for doc in reranked_contents]
    
    output = {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
    logfire.info(f"Retriever Node Output State: {_state_summary({**state, **output})}")
    return output
