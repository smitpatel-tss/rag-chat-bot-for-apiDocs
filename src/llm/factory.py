from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from src.config.settings import settings

class LLMFactory:
    @staticmethod
    def get_model() -> BaseChatModel:

        provider = settings.LLM_PROVIDER.lower()
        api_key = settings.LLM_API_KEY.get_secret_value()

        if not api_key:
            raise ValueError(f"LLM_API_KEY must be set for provider: {provider}")
        
        if provider == "openai":
            return ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                api_key=api_key
            )
            
        elif provider == "ollama_cloud":
            return ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                api_key=api_key,
                base_url=settings.LLM_BASE_URL
            )
        
        raise ValueError(f"Unsupported LLM provider configured: {provider}")

