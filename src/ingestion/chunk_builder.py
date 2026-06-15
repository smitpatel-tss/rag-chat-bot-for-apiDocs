import tiktoken
import uuid
import logging
import re
from typing import List, Dict, Any
from src.ingestion.nodes import BaseNode, SmartChunk
from src.config.settings import settings

logger = logging.getLogger(__name__)

class ChunkBuilder:
    def __init__(self):
        
        try:
            self.encoder = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)

        except KeyError as exc:
            raise ValueError(
                f"Unsupported embedding model '{settings.EMBEDDING_MODEL}'. "
                "Please verify EMBEDDING_MODEL in your environment configuration."
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize tokenizer for model "
                f"'{settings.EMBEDDING_MODEL}'."
            ) from exc

        if self.encoder is None:
            raise ValueError(
                "Tokenizer initialization failed: EMBEDDING_MODEL is missing or invalid."
            )
        
        self.text_target_limit = 900
        self.hard_embedding_ceiling = 4000

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _generate_slug(self, text: str) -> str:

        text = text.strip().lower()
        text = text.replace('.', '-')
        text = re.sub(r'[^a-z0-9\s\-]', '', text)
        text = re.sub(r'[\s_]+', '-', text)
        text = re.sub(r'-+', '-', text)
        return text.strip('-')

    def _build_metadata_attributes(self, node_metadata: Dict[str, Any], api_version: str | None = None) -> Dict[str, Any]:
        base_attrs = {}

        endpoint_paths = node_metadata.get("endpoint_paths") or []
        http_methods = node_metadata.get("http_methods") or []

        if endpoint_paths:
            base_attrs["endpoint_paths"] = sorted(set(endpoint_paths))
        if http_methods:
            base_attrs["http_methods"] = sorted(set(http_methods))

        semantic_analysis = node_metadata.get("semantic_analysis")
        if semantic_analysis:
            base_attrs["semantic_analysis"] = semantic_analysis

        if api_version:
            base_attrs["api_version"] = api_version

        return base_attrs

    def _build_metadata_attributes_from_list(self, node_list: List[BaseNode], api_version: str | None = None) -> Dict[str, Any]:
        base_attrs = {}
        endpoints, methods = set(), set()

        for node in node_list:
            endpoints.update(node.metadata.get("endpoint_paths") or [])
            methods.update(node.metadata.get("http_methods") or [])

        if endpoints:
            base_attrs["endpoint_paths"] = sorted(endpoints)
        if methods:
            base_attrs["http_methods"] = sorted(methods)

        if api_version:
            base_attrs["api_version"] = api_version

        return base_attrs

    def assemble_chunks(self, nodes: List[BaseNode], source_url: str, doc_id: str) -> List[SmartChunk]:
        raw_chunks: List[SmartChunk] = []
        current_text_buffer: List[BaseNode] = []
        current_buffer_tokens = 0
        current_section_path: List[str] = []

        for node in nodes:
            if not node.content.strip():
                continue

            node_tokens = self._count_tokens(node.content)

            if node.type not in ["code", "table"] and node_tokens > self.text_target_limit:
                logger.warning(f"TEXT node exceeds target chunk size ({node_tokens} tokens)")

            node_section_path = [v for k, v in sorted(node.headers.items()) if isinstance(k, int)]

            if not node_section_path:
                node_section_path = ["Document Root"]

            parent_hdr = node_section_path[-1]
            anchor_slug = self._generate_slug(parent_hdr)

            if node.type in ["code", "table"]:
                headers=""
                if current_text_buffer:

                    for i in range(len(current_text_buffer)-1, -1 , -1):
                        if current_text_buffer[i].type == "header":
                            chunk = current_text_buffer.pop(i)
                            headers = chunk.content + ("\n" + headers if headers else "")
                        else:
                            break


                    if current_text_buffer:
                        buffer_content = "\n\n".join([n.content for n in current_text_buffer])

                        raw_chunks.append(
                            SmartChunk(
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
                                attributes=self._build_metadata_attributes_from_list(current_text_buffer)
                            )
                        )

                    current_text_buffer = []
                    current_buffer_tokens = 0

                final_content = f"{headers}\n{node.content}" if headers else node.content
                actual_node_tokens = self._count_tokens(final_content)

                raw_chunks.append(
                    SmartChunk(
                        doc_id=doc_id,
                        chunk_id=str(uuid.uuid4()),
                        token_count=actual_node_tokens,
                        source_url=source_url,
                        anchor=anchor_slug,
                        title=node.title,
                        document_version=node.metadata.get("document_version", "v1.0"),
                        parent_header=parent_hdr,
                        section_path=node_section_path,
                        chunk_type=node.type.upper(),
                        content= final_content,
                        attributes=self._build_metadata_attributes(node.metadata)
                    )
                )

                if actual_node_tokens > self.hard_embedding_ceiling:
                    logger.warning(f"CRITICAL: Element exceeds embedding ceiling ({node_tokens} tokens).")
                    print(f"This node is bigger than the limit. Node Type: {node.type} | Token Count: {node_tokens} | Node Id: {raw_chunks[-1].chunk_id}")

                continue

            section_changed = node_section_path != current_section_path
            buffer_overflow = (current_buffer_tokens + node_tokens) > self.text_target_limit

            is_only_headers = all(n.type == "header" for n in current_text_buffer)

            if (section_changed or buffer_overflow) and current_text_buffer and not is_only_headers:

                buffer_content = "\n\n".join([n.content for n in current_text_buffer])

                raw_chunks.append(
                    SmartChunk(
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
                        attributes=self._build_metadata_attributes_from_list(current_text_buffer)
                    )
                )

                current_text_buffer = []
                current_buffer_tokens = 0

            if not current_text_buffer:
                current_section_path = node_section_path

            current_text_buffer.append(node)
            current_buffer_tokens += node_tokens

        if current_text_buffer:
            buffer_content = "\n\n".join([n.content for n in current_text_buffer])

            raw_chunks.append(
                SmartChunk(
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
                    attributes=self._build_metadata_attributes_from_list(current_text_buffer)
                )
            )

        total_created = len(raw_chunks)

        for idx in range(total_created):
            if idx > 0:
                raw_chunks[idx].prev_chunk_id = raw_chunks[idx - 1].chunk_id

            if idx < total_created - 1:
                raw_chunks[idx].next_chunk_id = raw_chunks[idx + 1].chunk_id

        return raw_chunks