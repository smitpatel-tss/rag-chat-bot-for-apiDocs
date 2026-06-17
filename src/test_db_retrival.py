from src.database.pg_store import PGVectorStore
from src.llm.factory import LLMFactory
from src.config.settings import settings

def test_pipeline(user_query: str):
    embedder = LLMFactory.get_embedding_model()
    vector_store = PGVectorStore(settings.VECTOR_DB_URL)
    vector_store.initialize_collection(settings.VECTOR_DB_COLLECTION, settings.EMBEDDING_DIMENSION)
    try:
        query_embedding = embedder.embed_query(user_query)
    except Exception as e:
        raise ValueError("Could not process the query text into a vector.")
    if not query_embedding:
        raise ValueError("Embedding generation returned an empty vector.")

    retrieved_chunks = vector_store.hybrid_search_RRF(
        table_name=settings.VECTOR_DB_COLLECTION,
        query_text=user_query,
        query_embedding=query_embedding
    )

    if not retrieved_chunks:
        print("FALIED!")

    for r in retrieved_chunks:
        print(r)

if __name__ == "__main__":
    test_pipeline("What are the mandatory fields?")