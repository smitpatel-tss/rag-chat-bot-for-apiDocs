import json
import logging
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.database.pg_store import PGVectorStore

# class RAGChatEngine: