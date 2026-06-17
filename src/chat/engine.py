import json
import logging
from typing import List, Dict, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.prompts.factory import PromptFactory
from src.database.pg_store import PGVectorStore

logger = logging.getLogger(__name__)

class RAGChatEngine:
    def __init__(
        self, 
        vector_store: PGVectorStore, 
        llm: BaseChatModel, 
        embedder: Embeddings,
        table_name: str = "api_documentation_chunks"
    ):
        self.vector_store = vector_store
        self.llm = llm
        self.embedder = embedder
        self.table_name = table_name

        self.prompt_template = PromptFactory.get("api_assistant", "v1")

    def _format_context(self, chunks: List[dict]) -> str:

        if not chunks:
            return "No relevant documentation found."

        formatted_sections = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Untitled Section")
            content = chunk.get("content", "")
            attrs = chunk.get("attributes", {})
            
            endpoints = attrs.get("endpoint_paths", [])
            methods = attrs.get("http_methods", [])
            
            semantics = attrs.get("semantic_analysis", {})
            mandatory_fields = semantics.get("mandatory_fields_extracted", [])
            payload_purpose = semantics.get("payload_purpose", "GENERAL_INFO")

            section = f"-- CHUNK {i}, from document: {title} --\n"
            if endpoints:
                section += f"ENDPOINTS: {', '.join(endpoints)}\n"
            if methods:
                section += f"METHODS: {', '.join(methods)}\n"
            if payload_purpose != "GENERAL_INFO":
                section += f"PURPOSE: {payload_purpose}\n"
            if mandatory_fields:
                section += f"MANDATORY FIELDS: {', '.join(mandatory_fields)}\n"
            
            section += f"\nRAW CONTENT:\n{content}\n"
            formatted_sections.append(section)

        return "\n\n".join(formatted_sections)
    
    def generate_answer(self, user_query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        
        logger.info(f"Processing query: '{user_query}'")

        try:
            query_embedding = self.embedder.embed_query(user_query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise ValueError("Could not process the query text into a vector.")
        if not query_embedding:
            raise ValueError("Embedding generation returned an empty vector.")

        logger.info("Executing Hybrid Search...")
        retrieved_chunks = self.vector_store.hybrid_search_RRF(
            table_name=self.table_name,
            query_text=user_query,
            query_embedding=query_embedding,
            filters=filters
        )

        retrieved_chunks = self.vector_store.hybrid_search_Reranker(
            table_name=self.table_name,
            query_text=user_query,
            query_embedding=query_embedding,
            filters=filters
        )

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in the documentation to answer your query.",
                "sources": []
            }

        formatted_context = self._format_context(retrieved_chunks)

        logger.info("Generating response from LLM...")
        chain = self.prompt_template | self.llm

        try:
            response = chain.invoke({
                "formatted_context": formatted_context,
                "user_query": user_query
            })
        except Exception as e:
            logger.exception(f"Failed to generate LLM response for query: '{user_query}'")
            raise RuntimeError("Failed to generate response from the language model.") from e

        source_metadata = [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "hybrid_score": chunk.get("final_score", 0.0),
                "source_url": chunk.get("source_url", ""),
                "anchor": chunk.get("anchor", "")
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": response.content,
            "sources": source_metadata
        }