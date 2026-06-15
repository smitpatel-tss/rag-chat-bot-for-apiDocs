import time
import logging
from typing import List
from langchain_core.embeddings import Embeddings
from src.ingestion.nodes import SmartChunk, EmbeddedChunk

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    def __init__(self, embedder: Embeddings, batch_size: int = 100, max_retries: int = 3):
        self.embedder = embedder
        self.batch_size = batch_size
        self.max_retries = max_retries

    def _format_for_embedding(self, chunk: SmartChunk) -> str:
        parts = []
        
        if chunk.title:
            parts.append(f"Document: {chunk.title}")
            
        if chunk.parent_header:
            parts.append(f"Section: {chunk.parent_header}")
            
        if chunk.attributes:
            endpoints = chunk.attributes.get("endpoint_paths", [])
            if endpoints:
                parts.append(f"Endpoints: {', '.join(endpoints)}")
            
            semantic_info = chunk.attributes.get("semantic_analysis", {})
            if semantic_info:
                description = semantic_info.get("functional_description", "")
                if description:
                    parts.append(f"Summary: {description}")
                
                mandatory_fields = semantic_info.get("mandatory_fields_extracted", [])
                if mandatory_fields:
                    parts.append(f"Mandatory fields: {', '.join(mandatory_fields)}")
                
                optional_fields = semantic_info.get("optional_fields_extracted", [])
                if optional_fields:
                    parts.append(f"Optional fields: {', '.join(optional_fields)}")
                
        parts.append(f"\nContent:\n{chunk.content}")
        
        return "\n".join(parts)

    def generate_embeddings(self, chunks: List[SmartChunk]) -> List[EmbeddedChunk]:

        if not chunks:
            return []

        for chunk in chunks:
            if not chunk.content or not chunk.content.strip():
                raise ValueError(f"Invalid empty chunk: {chunk.chunk_id}")

        embedded_chunks: List[EmbeddedChunk] = []
        total_chunks = len(chunks)

        logger.info(f"Starting vector embedding generation for {total_chunks} chunks...")

        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i: i + self.batch_size]
            
            texts_to_embed = [self._format_for_embedding(chunk) for chunk in batch]
            
            vectors = None

            for attempt in range(self.max_retries):
                try:
                    vectors = self.embedder.embed_documents(texts_to_embed)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.error(
                            f"CRITICAL: Failed batch {i}-{i + len(batch) - 1} "
                            f"after {self.max_retries} attempts. Error: {e}"
                        )
                        raise

                    wait_time = 2 * (attempt + 1)
                    logger.warning(
                        f"Transient error on batch {i}-{i + len(batch) - 1}. "
                        f"Retrying in {wait_time}s... (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)

            if not vectors or len(vectors) != len(batch):
                raise ValueError(
                    f"CRITICAL: Embedding vector count mismatch! "
                    f"Expected {len(batch)} vectors, received {len(vectors) if vectors else 0}."
                )

            for chunk, vector in zip(batch, vectors):
                if not vector:
                    raise ValueError(f"Empty embedding vector for chunk_id={chunk.chunk_id}")

                chunk_dict = chunk.model_dump()
                embedded_chunks.append(
                    EmbeddedChunk(**chunk_dict, embedding=vector)
                )

            logger.info(f"Successfully embedded chunks {i} to {i + len(batch) - 1}")

        logger.info(f"Embedding generation complete. Successfully processed {len(embedded_chunks)} chunks.")
        return embedded_chunks