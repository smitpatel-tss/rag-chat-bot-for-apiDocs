import os
from pathlib import Path
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    
    LLM_PROVIDER: Literal["openai", "ollama_cloud", "mock"] = "openai"
    LLM_API_KEY: SecretStr = Field(
        default=SecretStr(""), 
        description="Wrapped in SecretStr to prevent accidental logging leaks."
    )
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_BASE_URL: str = ""
    
    EMBEDDING_PROVIDER: Literal["openai", "local", "mock"] = "openai"
    EMBEDDING_API_KEY: SecretStr = Field(default=SecretStr(""))
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_BASE_URL: str = ""

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    VECTOR_DB_PROVIDER: Literal["qdrant", "pgvector", "mock"] = "mock"
    VECTOR_DB_URL: str = "http://localhost:6333"
    VECTOR_DB_API_KEY: SecretStr = Field(default=SecretStr(""))
    VECTOR_DB_COLLECTION: str = "api_docs"
    
    PAGE_BOOST_WEIGHT: float = Field(
        default=0.20, 
        description="The additive weight applied to the score of chunks matching current_page_path"
    )
    MAX_CONTEXT_CHUNKS: int = Field(
        default=5, 
        ge=1, 
        le=20, 
        description="The final Top N chunks selected to be sent to the LLM context window."
    )

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    PAGE_BOOST_WEIGHT: float = Field(default= 0.15, description="parameter to boost the same path weights")
    MAX_CONTEXT_CHUNKS: int = Field(default = 5, description="Max context chunk to give to llm for answering")

    VECTOR_SEARCH_TOP_K: int = Field(default=20, description="top k vector search results to take for scoring")
    KEYWORD_SEARCH_TOP_K: int = Field(default=20, description="top k keyword search results to take for scoring")

    VECTOR_WEIGHT: float = Field(default=1.0, description="Weight for score from vector results")
    KEYWORD_WEIGHT: float = Field(default=1.0, description="Weight for score from keyword results")

    COHERE_API_KEY: SecretStr = Field(
        default=SecretStr(""), 
        description="Wrapped in SecretStr to prevent accidental logging leaks."
    )
    RERANKER_PROVIDER: Literal["cohere", "cross_encoder", "mock"] = "cross_encoder"
    RERANKING_MODEL: str = "rerank-english-v3.0"
    RERANK_CANDIDATE_LIMIT: int = 30
    RERANKING_MODEL_BAAI: str ="BAAI/bge-reranker-v2-m3"

settings = Settings()