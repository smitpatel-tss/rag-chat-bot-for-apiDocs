from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class ChatPromptSpec:
    name: str
    version: str
    template: ChatPromptTemplate
    description: str = ""