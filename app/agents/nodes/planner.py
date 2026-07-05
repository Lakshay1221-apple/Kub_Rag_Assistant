from app.agents.state import AgentState 
from app.config import settings 
from langchain_groq import ChatGroq 
import logfire 

llm = ChatGroq(api_key = settings.GROQ_API_KEY, model = settings.GROQ_MODEL)

def planner_node(state: AgentState):

    '''
    The planner node determines if a 
    search is needed based on the Entire conversation.
    '''

    history = ""

    for msg in state['messages']:
        role = "User" if msg['role'] == 'user' else "Assistant"
        history += f"{role}: {msg['content']}\n"

    
    user_message = state['messages'][-1]['content'] if state['messages'] else ""

    prompt = f"""
        You are an intelligent query planner for a RAG system.

        Your task is to determine whether the user's latest message can be answered using the existing conversation context or if external retrieval/search is required.

        CONVERSATION HISTORY:
        {history}

        LATEST USER MESSAGE:
        "{user_message}"

        Instructions:

        1. Return "CONVERSATIONAL" if:
        - The message is a greeting (e.g., hi, hello, hey).
         - The answer can be derived entirely from the conversation history.
         - The user is referring to previous messages, context, or personal information already present in the conversation.
         - The question does not require external knowledge retrieval.

        2. Return a concise, optimized search query if:
         - The user is asking about Kubernetes, Intel, Networking, Cloud, DevOps, or other technical topics.
         - The answer requires documentation, technical references, specifications, troubleshooting information, or knowledge not present in the conversation history.
         - Fresh or authoritative information would improve the answer.

            Rules:
        - Do not explain your reasoning.
        - Do not output JSON.
        - Do not output anything except:
        - "CONVERSATIONAL"
        - OR a refined search query suitable for retrieval.

        Examples:

        User: "Hi"
        Output:
        CONVERSATIONAL

        User: "What is my name?"
        Output:
        CONVERSATIONAL

        User: "Explain Kubernetes Horizontal Pod Autoscaler"
        Output:
        Kubernetes Horizontal Pod Autoscaler overview and working

        User: "How does Intel TDX work?"
        Output:
        Intel Trust Domain Extensions TDX architecture and implementation

        Now determine the correct output.
        """
    
    with logfire.span("Planner Node"):
        decision = llm.invoke(prompt).content.strip()
        logfire.info(f"Planner Node Decision: {decision}")

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversational response without external search.",
            "plan" : ['Intent: Conversational Response', 'Action: Generate response based on conversation history'],
        }
    
    return {
        "current_query": decision,
        "status": "Search query generated for external retrieval.",
        "plan" : ['Intent: External Search Required', f'Action: Execute search with query "{decision}"'],
    }
    







