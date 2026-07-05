import logfire 
from app.agents.state import AgentState 
from app.services.retrieval.retriever import search_enterprise_knowledge
from app.services.retrieval.ranking_services import rerank_documents 

def retrieve_node(state: AgentState):
    """
    Retrieves relevant documents based on the current query in the state.
    """

    query = state['current_query']

    with logfire.span("Knowledge Retrieval"):
        logfire.info(f"Retrieving documents for query: {query}")
        raw_results = search_enterprise_knowledge(query=query, limit=15)
        logfire.info(f"Retrieved {len(raw_results)} documents from the knowledge base.")

        doc_contents = [doc['content'] for doc in raw_results]

        with logfire.span("Document Re-ranking"):
            try:
                content = llm.invoke(f"Re-rank the following documents based on their relevance to the query: '{query}'. Return the documents in order of relevance, from most relevant to least relevant. Only return the document contents, separated by new lines.\n\nDocuments:\n" + "\n".join(doc_contents)).content
                logfire.info("Documents re-ranked successfully.")

                return {
                    "final_answer" : content,
                    "status" : "Response generated",
                    "plan" : state["plan"],
                    "message" : [{"role" : "assistant", "content" : content}]
                }
            except Exception as e:
                logfire.error(f"Error re-ranking documents: {e}")
                return {
                    "final_answer" : "",
                    "status" : "Error generating response",
                    "plan" : state["plan"],
                    "message" : [{"role" : "assistant", "content" : ""}]
                }