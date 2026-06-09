import tiktoken
import uuid
import logging
import re
from typing import List, Dict, Any
from src.ingestion.nodes import BaseNode, SmartChunk

logger = logging.getLogger(__name__)

class ChunkBuilder:
    def __init__(self, model_name: str = "gpt-4o"):
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.text_target_limit = 900
        self.hard_embedding_ceiling = 4000 

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _generate_slug(self, text: str) -> str:
        slug = text.lower().strip()
        slug = re.sub(r'[^a-z0-9\s\-]', '', slug)
        slug = re.sub(r'[\s\_]+', '-', slug)
        return re.sub(r'\-+', '-', slug)

    def _build_metadata_attributes(self, node_metadata: Dict[str, Any]) -> Dict[str, Any]:
        base_attrs = {
            "endpoint_paths": node_metadata.get("endpoint_paths", []),
            "http_methods": node_metadata.get("http_methods", []),
            "api_version": "v1.1"
        }
        if "semantic_analysis" in node_metadata:
            base_attrs["semantic_analysis"] = node_metadata["semantic_analysis"]
            
        return base_attrs

    def assemble_chunks(self, nodes: List[BaseNode], source_url: str, doc_id: str) -> List[SmartChunk]:
        raw_chunks: List[SmartChunk] = []
        current_text_buffer = []
        current_buffer_tokens = 0
        current_section_path = []

        for node in nodes:
            node_tokens = self._count_tokens(node.content)
            node_section_path = [v for k, v in sorted(node.headers.items()) if isinstance(k, int)]
            if not node_section_path:
                node_section_path = ["Document Root"]

            parent_hdr = node_section_path[-1]
            anchor_slug = self._generate_slug(parent_hdr)

            if node.type in ["code", "table"]:
                if current_text_buffer:
                    buffer_content = "\n\n".join([n.content for n in current_text_buffer])
                    raw_chunks.append(SmartChunk(
                        doc_id=doc_id,
                        chunk_id=str(uuid.uuid4()),
                        token_count=self._count_tokens(buffer_content),
                        source_url=source_url,
                        anchor=self._generate_slug(current_section_path[-1]),
                        title=current_text_buffer[0].title,
                        document_version=current_text_buffer[0].metadata.get("document_version", "v1.0"),
                        parent_header=current_section_path[-1],
                        section_path=current_section_path,
                        chunk_type="TEXT",
                        content=buffer_content,
                        attributes=self._build_metadata_attributes(current_text_buffer[0].metadata)
                    ))
                    current_text_buffer = []
                    current_buffer_tokens = 0

                if node_tokens > self.hard_embedding_ceiling:
                    logger.warning(f"CRITICAL: Element exceeds embedding ceiling ({node_tokens} tokens).")

                raw_chunks.append(SmartChunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    token_count=node_tokens,
                    source_url=source_url,
                    anchor=anchor_slug,
                    title=node.title,
                    document_version=node.metadata.get("document_version", "v1.0"),
                    parent_header=parent_hdr,
                    section_path=node_section_path,
                    chunk_type=node.type.upper(),
                    content=node.content,
                    attributes=self._build_metadata_attributes(node.metadata)
                ))
                continue

            section_changed = node_section_path != current_section_path
            buffer_overflow = (current_buffer_tokens + node_tokens) > self.text_target_limit

            if (section_changed or buffer_overflow) and current_text_buffer:
                buffer_content = "\n\n".join([n.content for n in current_text_buffer])
                raw_chunks.append(SmartChunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    token_count=self._count_tokens(buffer_content),
                    source_url=source_url,
                    anchor=self._generate_slug(current_section_path[-1]),
                    title=current_text_buffer[0].title,
                    document_version=current_text_buffer[0].metadata.get("document_version", "v1.0"),
                    parent_header=current_section_path[-1],
                    section_path=current_section_path,
                    chunk_type="TEXT",
                    content=buffer_content,
                    attributes=self._build_metadata_attributes(current_text_buffer[0].metadata)
                ))
                current_text_buffer = []
                current_buffer_tokens = 0

            if not current_text_buffer:
                current_section_path = node_section_path

            current_text_buffer.append(node)
            current_buffer_tokens += node_tokens

        if current_text_buffer:
            buffer_content = "\n\n".join([n.content for n in current_text_buffer])
            raw_chunks.append(SmartChunk(
                doc_id=doc_id,
                chunk_id=str(uuid.uuid4()),
                token_count=self._count_tokens(buffer_content),
                source_url=source_url,
                anchor=self._generate_slug(current_section_path[-1]),
                title=current_text_buffer[0].title,
                document_version=current_text_buffer[0].metadata.get("document_version", "v1.0"),
                parent_header=current_section_path[-1],
                section_path=current_section_path,
                chunk_type="TEXT",
                content=buffer_content,
                attributes=self._build_metadata_attributes(current_text_buffer[0].metadata)
            ))

        total_created = len(raw_chunks)
        for idx in range(total_created):
            if idx > 0:
                raw_chunks[idx].prev_chunk_id = raw_chunks[idx - 1].chunk_id
            if idx < total_created - 1:
                raw_chunks[idx].next_chunk_id = raw_chunks[idx + 1].chunk_id

        return raw_chunks