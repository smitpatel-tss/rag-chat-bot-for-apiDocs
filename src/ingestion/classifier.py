import re
import json
from typing import List
from .nodes import BaseNode, CodeNode, ParagraphNode


class DeterministicClassifier:

    MAX_JSON_BUFFER_NODES = 100

    def _get_depth_delta_for_line(self, text: str) -> int:
        delta = 0
        escape_next = False
        in_string = False

        for char in text:
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char in '{[':
                    delta += 1
                elif char in '}]':
                    delta -= 1

        return delta

    def _is_json(self, text: str) -> bool:
        t = text.lstrip()
        if t.startswith("{"):
            return True
        if t.startswith("[{") or t.startswith("[\"") or t.startswith("[\n"):
            return True
        return False

    def classify_and_merge(self, nodes: List[BaseNode]) -> List[BaseNode]:
        merged_nodes = []

        in_json_block = False
        json_buffer = []
        original_nodes_buffer = []
        json_headers = {}
        json_metadata = {}
        json_title = "Untitled Document"

        depth = 0

        for node in nodes:

            is_evaluable = isinstance(node, ParagraphNode) or (isinstance(node, CodeNode) and not node.language)

            if not in_json_block:

                if not is_evaluable:
                    merged_nodes.append(node)
                    continue

                content_stripped = node.content.strip()
                is_json_start = self._is_json(content_stripped)

                if is_json_start:
                    in_json_block = True
                    json_headers = node.headers.copy()
                    json_metadata = node.metadata.copy()
                    json_title = node.title

                    depth += self._get_depth_delta_for_line(node.content)

                    json_buffer.append(node.content)
                    original_nodes_buffer.append(node)

                    if depth == 0:
                        merged_nodes.append(CodeNode(
                            type="code",
                            content="\n".join(json_buffer),
                            language="json",
                            headers=json_headers,
                            title=json_title,
                            metadata={
                                **json_metadata,
                                "valid_json": False,
                                "incomplete": True,
                                "reason": "zero_depth_immediate_close"
                            }
                        ))
                        in_json_block = False
                        json_buffer, original_nodes_buffer = [], []
                        depth = 0
                        json_headers = {}
                        json_metadata = {}
                        json_title = "Untitled Document"

                else:
                    merged_nodes.append(node)

            else:

                if len(json_buffer) >= self.MAX_JSON_BUFFER_NODES:
                    merged_nodes.append(CodeNode(
                        type="code",
                        content="\n".join(json_buffer),
                        language="json",
                        headers=json_headers,
                        title=json_title,
                        metadata={
                            "valid_json": False,
                            "incomplete": True,
                            "reason": "safety_abort_max_buffer"
                        }
                    ))

                    in_json_block = False
                    json_buffer, original_nodes_buffer = [], []
                    depth = 0
                    json_headers = {}
                    json_metadata = {}
                    json_title = "Untitled Document"
                    continue

                depth += self._get_depth_delta_for_line(node.content)

                json_buffer.append(node.content)
                original_nodes_buffer.append(node)

                if depth == 0:
                    merged_text = "\n".join(json_buffer)

                    try:
                        json.loads(merged_text)
                        is_valid_json = True
                        reason = "success"
                    except json.JSONDecodeError:
                        is_valid_json = False
                        reason = "parse_failure"

                    merged_nodes.append(CodeNode(
                        type="code",
                        content=merged_text,
                        language="json",
                        headers=json_headers,
                        title=json_title,
                        metadata={
                            "valid_json": is_valid_json,
                            "incomplete": not is_valid_json,
                            "reason": reason
                        }
                    ))

                    in_json_block = False
                    json_buffer, original_nodes_buffer = [], []
                    depth = 0
                    json_headers = {}
                    json_metadata = {}
                    json_title = "Untitled Document"

        if in_json_block and json_buffer:
            merged_nodes.append(CodeNode(
                type="code",
                content="\n".join(json_buffer),
                language="json",
                headers=json_headers,
                title=json_title,
                metadata={
                    "valid_json": False,
                    "incomplete": True,
                    "reason": "unclosed_json_block"
                }
            ))

        return merged_nodes