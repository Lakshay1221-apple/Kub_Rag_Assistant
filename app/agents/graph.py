import logfire
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


def _state_summary(state: AgentState) -> dict:
    """Return compact state details for graph routing logs."""

    return {
        "current_query": state.get("current_query"),
        "messages_count": len(state.get("messages", [])),
        "documents_count": len(state.get("documents", [])),
        "plan": state.get("plan", []),
        "status": state.get("status"),
        "has_final_answer": bool(state.get("final_answer")),
    }


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    logfire.info(f"LangGraph Routing State: {_state_summary(state)}")
    if state["current_query"] == "CONVERSATIONAL":
        logfire.info("LangGraph Routing Decision: responder")
        return "responder"
    logfire.info("LangGraph Routing Decision: retriever")
    return "retriever"

workflow.set_entry_point("planner")


# Conditional Edge: Planner -> Router -> (Retriever OR Responder)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder"
    }
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)


# --- MEMORY UPGRADE ---
# MemorySaver allows the agent to remember conversations based on 'thread_id'
checkpointer = MemorySaver()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
