'''A module for processing text data for ingestion.'''

import os 
import uuid 
import json 
import logfire 

from qdrant_client import QdrantClient
from qdrant_client.http import models 

from app.config import settings
from app.services.retrieval.embeddings import embed_texts, get_embedding_dim
from app.ingestion.chucking.splitter import chunk_text
from app.ingestion.loaders.office import parse_office
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text

PROCESSED_DATA_DIR = "processed_data"

qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    api_key = settings.QDRANT_API_KEY,
)

if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(service_name="ingestion_processor", service_version="1.0.0")
else:
    logfire.configure(send_to_logfire=False)

def save_processed_locally(data: dict, source_type : str, filename : str) -> str:
    '''Save processed data to a local JSON file.'''

    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok = True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding = 'utf-8')  as f:
        json.dump(data, f, ensure_ascii = False, indent = 4)
    logfire.info(f"Processed data saved locally at {dest}")

    return dest


def process_file(file_path: str, filename: str, source_type: str) -> None:
    ''' Process a file based on its type and save the processed data locally.'''

    with logfire.span("Processing File", file = filename , source  = source_type) :

        try:
            ext = filename.lower().split(".", 1)[-1]
            if ext == "pdf":
                text = parse_pdf(file_path)
            elif ext in ["html", "htm"]:
                text = parse_html(file_path)
            elif ext in ["txt", "md", "csv", "json"]: 
                text = parse_text(file_path)
            elif ext in ["docx", "pptx"]:
                text = parse_office(file_path)
            else:
                logfire.error(f"Unsupported file type: {ext}")
                raise ValueError(f"Unsupported file type: {ext}")     
            
            # Chunking text 

            chunks = chunk_text(text)
            logfire.info(f"Text chunked into {len(chunks)} chunks for file {filename}")
            if not chunks:
                logfire.warning(f"No chunks created for file {filename}. Skipping.")
                raise ValueError(f"No chunks created for file {filename}")
            
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks
            }

            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Processed data for file {filename} saved at {local_path}")

            with logfire.span("Embedding Chunks", file = filename , source  = source_type) :

                embeddings = embed_texts(chunks)

                points = [
                    models.PointStruct(
                        id = str(uuid.uuid4()),
                        vector = embedding,
                        payload = {
                            "filename": filename,
                            "source_type": source_type,
                            "chunk_index": idx,
                            "text": chunk
                        }
                    )
                    for idx, (embedding, chunk) in enumerate(zip(embeddings, chunks))
                ]

                logfire.info(f"Embeddings generated for {len(embeddings)} chunks for file {filename}")

                
                qdrant_client.upsert(
                    collection_name = settings.QDRANT_COLLECTION_NAME,
                    points = points,
                )
                logfire.info(f"Upserted {len(points)} points to Qdrant for file {filename}")

        
        except Exception as e:
            logfire.error(f"Error processing file {filename}: {e}")
            raise


def process_directory(dir_path: str, source_type: str) -> None:
    """ Process all files in a directory based on their type and save the processed data locally. """

    with logfire.span('Scanning Directory', directory = dir_path, source = source_type):

        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)



def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False) -> None:
    ''' Run the universal ingestion process for a given base directory. '''

    with logfire.span("Universal Ingestion", base_directory = base_dir, explicit_source_type = explicit_source_type, wipe = wipe):

        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE
                ),
            )

            logfire.info(
                f"Created Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' with dimension {dim}"
                f"({dim}-dim, Cosine)."
            )

        subdirs  = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name 
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No subdirectories found. Processing base directory as source type '{source_type}'")
            process_directory(base_dir, source_type)

        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir 
                )
                process_directory(os.path.join(base_dir, subdir), source_type)

    
