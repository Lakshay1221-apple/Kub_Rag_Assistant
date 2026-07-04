''' Embedding service for text retrieval. '''

import time

import logfire 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer
from app.config import settings 

BATCH_SIZE = 50
_GEMINI_DIM = 3072
_FALLBACK_DIM = 768

_active_model = None 
_model_type : str | None = None
_embedding_dim: int | None = None


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True when an embedding failure looks like quota or rate limiting."""

    err = str(exc).lower()
    return any(token in err for token in ("429", "rate", "quota", "resources_exhausted"))


def _switch_to_fallback_model() -> None:
    """Replace the active model with the local fallback implementation."""

    global _active_model, _model_type, _embedding_dim

    _active_model = _load_fallback()
    _model_type = "fallback"
    if _embedding_dim is None:
        _embedding_dim = _FALLBACK_DIM
    logfire.warning("Switched embedding service to fallback model.")


def _normalize_vectors(vectors: list[list[float]], target_dim: int) -> list[list[float]]:
    """Pad or truncate vectors to the requested dimensionality."""

    normalized: list[list[float]] = []
    for vector in vectors:
        if len(vector) >= target_dim:
            normalized.append(vector[:target_dim])
        else:
            normalized.append(vector + [0.0] * (target_dim - len(vector)))
    return normalized


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

    global _active_model, _model_type, _embedding_dim

    if _active_model is not None:
        return 
    
    _active_model = _probe_gemini()
    if not _active_model:
        _active_model = _load_fallback()
        _model_type = "fallback"
        _embedding_dim = _FALLBACK_DIM
    else:
        _model_type = "gemini"     
        _embedding_dim = _GEMINI_DIM



def get_embedding_dim() -> int:    
    ''' Get the dimension of the embedding model. '''

    _init()
    return _embedding_dim or _FALLBACK_DIM



def _embed_batch(batch: list[str]) -> list[list[float]]:    
    ''' Embed a batch of texts. '''

    if _model_type == "gemini":
        for attempt in range(4):
            try:
                return _active_model.embed_documents(batch)
            except Exception as e:
                is_rate_limit = _is_rate_limit_error(e)
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f'Gemini embedding rate limit hit, retrying in {wait} seconds (attempt {attempt + 1}/3).'
                        f"(attempt {attempt + 1}/3)."
                    )
                    time.sleep(wait)
                elif is_rate_limit:
                    logfire.warning(
                        "Gemini quota exhausted after retries; falling back to local embeddings."
                    )
                    _switch_to_fallback_model()
                    fallback_vectors = _active_model.encode(batch).tolist()
                    return _normalize_vectors(fallback_vectors, _GEMINI_DIM)
                else:
                    logfire.error(f"Gemini embedding failed: {e}")
                    raise
        raise RuntimeError("Gemini embedding failed after 4 attempts.")
    else:
        target_dim = _embedding_dim or _FALLBACK_DIM
        return _normalize_vectors(_active_model.encode(batch).tolist(), target_dim)
    


def embed_query(query: str) -> list[float]:
    ''' Embed a single query. '''

    _init()
    if _model_type == "gemini":
        return _active_model.embed_query(query)
    target_dim = _embedding_dim or _FALLBACK_DIM
    return _normalize_vectors(_active_model.encode([query]).tolist(), target_dim)[0]



def embed_texts(texts: list[str]) -> list[list[float]]:
    ''' Embed a list of texts in batches. '''

    _init()
    if not texts:
        return []
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        with logfire.span("embedding_batch", model = _model_type, start = i, batch_size=len(batch)):
            all_embeddings.extend(_embed_batch(batch))

    return all_embeddings
