from src.embeddings.engine import EmbeddingEngine
from src.database.base import BaseVectorStore
from src.database.pg_store import PGVectorStore
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.nodes import SmartChunk, EmbeddedChunk
from src.llm.factory import LLMFactory
from src.config.settings import settings


def run_pipeline(mdfile_path: str, frontend_url: str, document_id: str):
    extraction_pipeline: IngestionPipeline = IngestionPipeline(llm_provider=LLMFactory)

    raw_chunks: list[SmartChunk] = extraction_pipeline.process_file(file_path=mdfile_path, source_url=frontend_url, doc_id=document_id)

    embedding_model = LLMFactory.get_embedding_model()

    sample_embedding = embedding_model.embed_query("test")
    actual_dim = len(sample_embedding)

    if actual_dim != settings.EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding dimension mismatch. Expected {settings.EMBEDDING_DIMENSION}, got {actual_dim}")

    embedding_engine: EmbeddingEngine = EmbeddingEngine(embedding_model)
    embedded_chunks: list[EmbeddedChunk] = embedding_engine.generate_embeddings(raw_chunks)

    # return list(zip(raw_chunks, embedded_chunks))

    db_string: str = settings.VECTOR_DB_URL
    vector_store: BaseVectorStore = PGVectorStore(connection_string=db_string)

    collection_name = settings.VECTOR_DB_COLLECTION
    vector_dim = settings.EMBEDDING_DIMENSION

    try:
        vector_store.initialize_collection(table_name=collection_name, vector_dim=vector_dim)
        vector_store.upsert_chunks(table_name=collection_name, chunks=embedded_chunks)
    finally:
        vector_store.close()