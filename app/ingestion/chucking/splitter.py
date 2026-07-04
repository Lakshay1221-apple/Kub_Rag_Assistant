'''A module for splitting a list of items into chunks of a specified size.'''

from typing import List, list 
import logfire 

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    '''
    Simple sematic-ish chunking of text into chunks of a specified size.
    '''  

    with logfire.span("chunk_text", chunk_size=chunk_size):
        if not text.strip():
            return []
        
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 <= chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        logfire.info("Chunking completed: {} chunks created", len(valid_chunks))
        return valid_chunks



        
