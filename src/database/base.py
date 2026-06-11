from abc import ABC, abstractmethod
from typing import List

from src.ingestion.nodes import EmbeddedChunk

class BaseVectorStore(ABC):

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the vector database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close any open database connections."""
        pass

    @abstractmethod
    def initialize_collection(self, table_name: str, vector_dim: int) -> None:
        """
        Create or initialize the underlying collection/table
        and any required indexes.
        """
        pass

    @abstractmethod
    def upsert_chunks(self, table_name: str, chunks: List[EmbeddedChunk]) -> None:
        """
        Insert new chunks or update existing chunks.
        """
        pass