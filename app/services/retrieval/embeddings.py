''' Embedding service for text retrieval. '''

import time
from xml.parsers.expat import model 
import logfire 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from app.config import settings 

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768

_active_model = None 
_model_type : str | None = None



def _probe_gemini():
    ''' Probe the Gemini API to check if it is available and working. '''

    try:
        model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview",
            google_api_key = settings.GEMINI_API_KEY,

        )
        model.embed_query('probe')
        logfire.info("Gemini embedding model is available and working.")
        return model
    
    except Exception as e:
        logfire.error(f"Gemini embedding model probe failed: {e}")
        return None
    


def _load_fallback():
    ''' Load the fallback embedding model. '''

    from sentence_transformers import SentenceTransformer
    logfire.info("Loading fallback embedding model (sentence_tranformers).")
    return SentenceTransformer('all-mpnet-base-v2')   



def _init():    
    ''' Initialize the embedding model. '''

    global _active_model, _model_type

    if _active_model is not None:
        return 
    
    _active_model = _probe_gemini()
    if not _active_model:
        _active_model = _load_fallback()
        _model_type = "fallback"
    else:
        _model_type = "gemini"     



def get_embedding_dim() -> int:    
    ''' Get the dimension of the embedding model. '''

    _init()
    return _GEMINI_DIM if _model_type == "gemini"  else _FALLBACK_DIM



def _embed_batch(batch: list[str]) -> list[list[float]]:    
    ''' Embed a batch of texts. '''

    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in("429", "rate", 'quota', 'resources_exhausted'))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f'Gemini embedding rate limit hit, retrying in {wait} seconds (attempt {attempt + 1}/3).'
                        f"(attempt {attempt + 1}/3)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini embedding failed after 4 attempts.")
    else:
        return _active_model.encode(batch).tolist()
    


def embed_query(query: str) -> list[float]:
    ''' Embed a single query. '''

    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    return _active_model.encode([query])[0].tolist()



def embed_texts(texts: list[str]) -> list[list[float]]:
    ''' Embed a list of texts in batches. '''

    _init()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        with logfire.span("embedding_batch", model = _model_type, start = i, batch_size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))

    return all_embeddings
