import logfire 
from app.agents.state import AgentState 
from app.config import settings 
from langchain_groq import ChatGroq 

llm = ChatGroq(api_key = settings.GROQ_API_KEY, model = settings.GROQ_MODEL, temperature = 0.0, max_tokens = 100)

def generate_node(state : AgentState):

    '''
    Synthesizes a response based on the conversation history and the latest user message.
    '''

    query = state['current_query']

    history_str = ""

    for msg in state['messages'][:-1]:
        role = "User" if msg['role'] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state['messages'][-1]['content'] if state['messages'] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating response based on conversation history.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI assistant.
        Answer the user's latest message using the CONVERSATION HISTROY below. 

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """

    else:
        logfire.info(f"Generating response based on external retrieval for query: {query}")
        max_content_chars = 25000
        full_content = ""

        for doc in state['documents']:
            if len(full_content) + len(doc) < max_content_chars:
                full_content += doc + "\n"
            else:
                logfire.warning("Maximum context length reached. Some documents may be truncated.")
                break
        
        prompt = f"""

        You are a Senior Technical Enterprise AI architect.
        Answer the question using the  TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_content}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"  
        """  

        with logfire.span("Generating Response", query = query, user_message = user_msg):
            try:
                content  = llm.invoke(prompt).content
                logfire.info("Response generated successfully.")

                return {
                    "final_answer": content,
                    "status" : "Response Generated",
                    "plan" : state["plan"],
                    "message" : [{"role" : "assistant", "content" : content}]
                }
            except Exception as e:
                logfire.error(f"Error generating response: {e}")
                return {
                    "final_answer": "",
                    "status" : "Error Generating Response",
                    "plan" : state["plan"],
                    "message" : [{"role" : "assistant", "content" : ""}]
                }
                







    




