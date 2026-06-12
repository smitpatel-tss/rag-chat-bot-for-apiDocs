import logging
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, Json
from psycopg2.extensions import connection
from pgvector.psycopg2 import register_vector
from typing import List, Optional

from src.database.base import BaseVectorStore
from src.ingestion.nodes import EmbeddedChunk

logger = logging.getLogger(__name__)

class PGVectorStore(BaseVectorStore):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn: Optional[connection] = None

    def connect(self) -> None:
        try:
            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
            logger.info("Successfully connected to PostgreSQL/pgvector.")
        except Exception as e:
            logger.error(f"Failed to connect to DB: {e}")
            raise

    def close(self) -> None:

        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Database connection closed.")

    def initialize_collection(self, table_name: str, vector_dim: int) -> None:
        if vector_dim <= 0:
            raise ValueError(f"vector_dim must be positive, got {vector_dim}")

        if self.conn is None:
            self.connect()
            
        assert self.conn is not None
        
        create_extension_query = "CREATE EXTENSION IF NOT EXISTS vector;"
        
        create_table_query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            chunk_id VARCHAR(255) PRIMARY KEY,
            doc_id VARCHAR(255) NOT NULL,
            
            prev_chunk_id VARCHAR(255),
            next_chunk_id VARCHAR(255),
            token_count INTEGER,
            
            source_url TEXT,
            anchor TEXT,
            
            title TEXT,
            document_version VARCHAR(100),
            parent_header TEXT,
            section_path JSONB,
            
            chunk_type VARCHAR(50),
            content TEXT NOT NULL,
            attributes JSONB,
            
            embedding VECTOR({}),
            
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """).format(
            sql.Identifier(table_name),
            sql.SQL(str(vector_dim)) 
        )
        
        create_index_query = sql.SQL("""
        CREATE INDEX IF NOT EXISTS {} 
        ON {} USING hnsw (embedding vector_cosine_ops);
        """).format(
            sql.Identifier(f"{table_name}_embedding_idx"),
            sql.Identifier(table_name)
        )
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_extension_query)
                cur.execute(create_table_query)
                cur.execute(create_index_query)
                self.conn.commit()
            logger.info(f"Initialized pgvector table '{table_name}' with dimension {vector_dim}.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def upsert_chunks(self, table_name: str, chunks: List[EmbeddedChunk]) -> None:
        if self.conn is None:
            self.connect()

        assert self.conn is not None

        if not chunks:
            return

        insert_query = sql.SQL("""
            INSERT INTO {} (
                chunk_id, doc_id, prev_chunk_id, next_chunk_id, token_count,
                source_url, anchor, title, document_version, parent_header,
                section_path, chunk_type, content, attributes, embedding
            ) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                prev_chunk_id = EXCLUDED.prev_chunk_id,
                next_chunk_id = EXCLUDED.next_chunk_id,
                token_count = EXCLUDED.token_count,
                source_url = EXCLUDED.source_url,
                anchor = EXCLUDED.anchor,
                title = EXCLUDED.title,
                document_version = EXCLUDED.document_version,
                parent_header = EXCLUDED.parent_header,
                section_path = EXCLUDED.section_path,
                chunk_type = EXCLUDED.chunk_type,
                content = EXCLUDED.content,
                attributes = EXCLUDED.attributes,
                embedding = EXCLUDED.embedding,
                updated_at = NOW();
        """).format(sql.Identifier(table_name))

        values = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.prev_chunk_id,
                chunk.next_chunk_id,
                chunk.token_count,
                chunk.source_url,
                chunk.anchor,
                chunk.title,
                chunk.document_version,
                chunk.parent_header,
                Json(chunk.section_path), 
                chunk.chunk_type,
                chunk.content,
                Json(chunk.attributes), 
                chunk.embedding
            )
            for chunk in chunks
        ]

        try:
            with self.conn.cursor() as cur:
                execute_values(cur, insert_query, values, page_size=100)
                self.conn.commit()
            logger.info(f"Successfully upserted {len(chunks)} chunks into {table_name}.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to upsert chunks: {e}")
            raise
    
    def _build_filter_clause(self, filters: dict = None) -> sql.Composed:

        if not filters:
            return sql.SQL("")
        
        conditions = []
        for key in filters.keys():

            if key.startswith("attr_"):
                json_key = key.replace("attr_", "")
                conditions.append(sql.SQL("attributes->>{} = {}").format(
                    sql.Literal(json_key), sql.Placeholder(key)
                ))
            else:
                conditions.append(sql.SQL("{} = {}").format(
                    sql.Identifier(key), sql.Placeholder(key)
                ))
                
        return sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions)
    
    def vector_search(self, table_name: str, query_embedding: List[float], limit: int = 5, filters: dict = None) -> List[dict]:
        
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty.")

        if self.conn is None:
            self.connect()
        assert self.conn is not None

        filters = filters or {}
        filter_clause = self._build_filter_clause(filters)
        
        query = sql.SQL("""
            SELECT chunk_id, doc_id, title, content, attributes, source_url, anchor,
                   1 - (embedding <=> {embedding}::vector) AS score
            FROM {table}
            {filters}
            ORDER BY embedding <=> {embedding}::vector
            LIMIT {limit};
        """).format(
            table=sql.Identifier(table_name),
            embedding=sql.Placeholder("embedding"),
            filters=filter_clause,
            limit=sql.Placeholder("limit")
        )

        params = {"embedding": query_embedding, "limit": limit, **filters}

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Vector search failed: {e}")
            raise
            
        return [
            {"chunk_id": r[0], "doc_id": r[1], "title": r[2], "content": r[3], "attributes": r[4], "source_url": r[5], "anchor": r[6], "score": r[7]} 
            for r in rows
        ]

    def keyword_search(self, table_name: str, query_text: str, limit: int = 5, filters: dict = None) -> List[dict]:

        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty.")

        if self.conn is None:
            self.connect()
        assert self.conn is not None

        filters = filters or {}
        filter_clause = self._build_filter_clause(filters)
        
        fts_placeholder = sql.Placeholder("query_text")
        fts_body = sql.SQL("to_tsvector('english', title || ' ' || content) @@ websearch_to_tsquery('english', {})").format(
            fts_placeholder
        )
        fts_condition = (sql.SQL(" AND ") if filters else sql.SQL(" WHERE ")) + fts_body

        query = sql.SQL("""
            SELECT chunk_id, doc_id, title, content, attributes, source_url, anchor,
                   ts_rank(to_tsvector('english', title || ' ' || content), websearch_to_tsquery('english', {query_text})) AS score
            FROM {table}
            {filters}
            {fts_condition}
            ORDER BY score DESC
            LIMIT {limit};
        """).format(
            table=sql.Identifier(table_name),
            filters=filter_clause,
            fts_condition=fts_condition,
            query_text=fts_placeholder,
            limit=sql.Placeholder("limit")
        )

        params = {"query_text": query_text, "limit": limit, **filters}

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Keyword search failed: {e}")
            raise

        return [
            {"chunk_id": r[0], "doc_id": r[1], "title": r[2], "content": r[3], "attributes": r[4], "source_url": r[5], "anchor": r[6], "score": r[7]} 
            for r in rows
        ]
    
    def hybrid_search(self, table_name: str, query_text: str, query_embedding: List[float], limit: int = 10, filters: dict = None) -> List[dict]:

        fetch_limit = limit * 2
        
        vector_results = self.vector_search(table_name, query_embedding, limit=fetch_limit, filters=filters)
        keyword_results = self.keyword_search(table_name, query_text, limit=fetch_limit, filters=filters)

        # RRF_Score = 1 / (k + rank)
        k = 60
        rrf_scores = {}
        chunk_data = {}

        for rank, item in enumerate(vector_results):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            chunk_data[cid] = item

        for rank, item in enumerate(keyword_results):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            chunk_data[cid] = item

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for cid, score in sorted_chunks[:limit]:
            result = chunk_data[cid]
            result["hybrid_score"] = score
            final_results.append(result)
            
        return final_results[:5]
