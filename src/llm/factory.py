from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models import BaseChatModel
from src.config.settings import settings
from langchain_core.embeddings import Embeddings

class LLMFactory:
    @staticmethod
    def get_model() -> BaseChatModel:

        provider = settings.LLM_PROVIDER.lower()
        api_key = settings.LLM_API_KEY.get_secret_value()

        if not api_key:
            raise ValueError(f"LLM_API_KEY must be set for provider: {provider}")
        
        if provider == "openai":
            return ChatOpenAI(
                model= settings.LLM_MODEL,
                temperature= settings.LLM_TEMPERATURE,
                api_key=api_key
            )
            
        elif provider == "ollama_cloud":
            if not settings.LLM_BASE_URL:
                raise ValueError(f"Base Url missing for: {provider}")
            return ChatOpenAI(
                model= settings.LLM_MODEL,
                temperature= settings.LLM_TEMPERATURE,
                api_key=api_key,
                base_url= settings.LLM_BASE_URL
            )
        
        raise ValueError(f"Unsupported LLM provider configured: {provider}")
    
    @staticmethod
    def get_embedding_model() -> Embeddings:

        provider = settings.EMBEDDING_PROVIDER.lower()
        api_key = settings.EMBEDDING_API_KEY.get_secret_value()

        if provider == "openai":
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=api_key
            )

        elif provider == "ollama_cloud":
            if not settings.EMBEDDING_BASE_URL:
                raise ValueError(f"Base Url missing for: {provider}")
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=api_key,
                base_url=settings.EMBEDDING_BASE_URL
            )

        raise ValueError(
            f"Unsupported embedding provider configured: {provider}"
        )