import cohere
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

from src.config.settings import settings

class Reranker:
    def __init__(self):
        self.reranker_provider = settings.RERANKER_PROVIDER
        
        if self.reranker_provider == "cohere":
            cohere_key = settings.COHERE_API_KEY.get_secret_value() if hasattr(settings.COHERE_API_KEY, "get_secret_value") else settings.COHERE_API_KEY
            self.co_client = cohere.Client(cohere_key)

        elif self.reranker_provider == "cross_encoder":
            model_name = getattr(settings, "RERANKING_MODEL_BAAI", "Alibaba-NLP/gte-reranker-modernbert-base")
            self.cross_encoder_model = CrossEncoder(model_name, max_length=8192)
        
        else:
            raise ValueError(f"Unsupported Reranker provider: {self.reranker_provider}")

    def _format_doc_for_reranking(self, doc: Dict[str, Any]) -> str:
       
        title = doc.get("title", "No Title")
        raw_content = doc.get("content", "")
        
        attributes = doc.get("attributes") or {}
        
        endpoints = attributes.get("endpoint_paths", [])
        methods = attributes.get("http_methods", [])
        
        endpoint_str = ", ".join(endpoints) if endpoints else "Unknown Endpoint"
        method_str = ", ".join(methods) if methods else "UNKNOWN"
        
        semantic_data = attributes.get("semantic_analysis", {})
        func_desc = semantic_data.get("functional_description", "")
        
        header_parts = [f"API Reference: {title}", f"Endpoint: {method_str} {endpoint_str}"]
        
        if func_desc:
            header_parts.append(f"Context: {func_desc}")
            
        dense_header = " | ".join(header_parts) + "\n---\n"
        
        return f"{dense_header}{raw_content}"

    def rerank(self, doc_list: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        if not doc_list:
            return []

        try:
            if self.reranker_provider == "cohere":
                return self._reranker_cohere(doc_list=doc_list, query=query, limit=limit)
            if self.reranker_provider == "cross_encoder":
                return self._reranker_crossencoder(doc_list=doc_list, query=query, limit=limit)
            
            return doc_list[:limit]
        
        except Exception as e:
            print(f"Reranker failed: {e}. Falling back to default retrieval order.")
            return [doc.copy() for doc in doc_list[:limit]]

    def _reranker_cohere(self, doc_list: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        documents_to_rerank = [self._format_doc_for_reranking(c) for c in doc_list]

        rerank_response = self.co_client.rerank(
            model=getattr(settings, "RERANKING_MODEL", "rerank-english-v3.0"),
            query=query,
            documents=documents_to_rerank,
            top_n=limit
        )

        final_result = []
        for result in rerank_response.results:
            matched_candidate = doc_list[result.index].copy()
            matched_candidate["final_score"] = result.relevance_score
            final_result.append(matched_candidate)

        return final_result
        
    def _reranker_crossencoder(self, doc_list: List[Dict[str, Any]], query: str, limit: int) -> List[Dict[str, Any]]:
        documents_to_rerank = [(query, self._format_doc_for_reranking(c)) for c in doc_list]

        scores = self.cross_encoder_model.predict(documents_to_rerank)
        ranked = sorted(zip(doc_list, scores), key=lambda x: x[1], reverse=True)[:limit]
        
        final_results = []
        for doc, score in ranked:
            doc_copy = doc.copy()
            doc_copy["final_score"] = float(score) 
            final_results.append(doc_copy)
        
        return final_results