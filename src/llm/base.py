from typing import Protocol
from langchain_core.language_models import BaseChatModel

class LLMProvider(Protocol):
    @staticmethod
    def get_model() -> BaseChatModel:
        pass