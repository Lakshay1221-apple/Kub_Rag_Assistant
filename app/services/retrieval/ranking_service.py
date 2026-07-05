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




