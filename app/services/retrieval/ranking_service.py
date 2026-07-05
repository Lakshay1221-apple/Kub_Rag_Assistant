import time 
import logfire 

from flashrank import Ranker , RerankRequest

_ranker = None

def _get_ranker() -> Ranker:
    """Get or create a Ranker instance."""

    global _ranker
    if _ranker is None:
        logfire.info("Initializing Ranker instance...")
        try:
            _ranker = Ranker()
            logfire.info("Ranker instance initialized successfully.")
        except Exception as e:
            logfire.error(f"Failed to initialize Ranker instance: {e}")
            raise

    return _ranker

def rerank_documents(query: str, documents: list[str]) -> list[str]:
    """Rerank documents based on their relevance to the query using FlashRank."""

    ranker = _get_ranker()

    try:
        logfire.info(f"Reranking {len(documents)} documents for query: '{query}'")
        start_time = time.time()

        rerank_request = RerankRequest(query=query, documents=documents)
        reranked_documents = ranker.rerank(rerank_request)

        elapsed_time = time.time() - start_time        
        logfire.info(f"Reranking completed in {elapsed_time:.2f} seconds.")
        
        return reranked_documents
    

    except Exception as e:
        logfire.error(f"Error during document reranking: {e}")
        raise




