import re
from typing import List, Tuple
from .nodes import BaseNode, CodeNode, ParagraphNode

class DeterministicClassifier:

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

    def classify_and_merge(self, nodes: List[BaseNode]) -> List[BaseNode]:
        merged_nodes = []
        
        in_json_block = False
        json_buffer = []
        original_nodes_buffer = [] 
        json_headers = {}
        
        depth = 0

        for node in nodes:
            is_evaluable = isinstance(node, ParagraphNode) or (isinstance(node, CodeNode) and not node.language)

            if not in_json_block:
                if not is_evaluable:
                    merged_nodes.append(node)
                    continue

                content_stripped = node.content.strip()
                is_json_object = content_stripped.startswith("{")
                is_json_array = bool(re.match(r'^\[\s*(?:\{|\[|\"|\d|\]|$)', content_stripped))
                
                if is_json_object or is_json_array:
                    in_json_block = True
                    json_headers = node.headers.copy()
                    json_metadata = node.metadata.copy()
                    
                    depth += self._get_depth_delta_for_line(node.content)
                    json_buffer.append(node.content)
                    original_nodes_buffer.append(node)
                    
                    if depth == 0:
                        merged_nodes.append(CodeNode(
                            type="code",
                            content="\n".join(json_buffer),
                            language="json",
                            headers=json_headers,
                            metadata=json_metadata
                        ))
                        in_json_block = False
                        json_buffer, original_nodes_buffer = [], []
                else:
                    merged_nodes.append(node)
                    
            else:

                if not is_evaluable or len(json_buffer) > 150:
                    merged_nodes.extend(original_nodes_buffer)
                    
                    in_json_block = False
                    json_buffer, original_nodes_buffer = [], []
                    depth = 0
                    
                    merged_nodes.append(node)
                    continue

                depth += self._get_depth_delta_for_line(node.content)
                json_buffer.append(node.content)
                original_nodes_buffer.append(node)
                
                if depth == 0:
                    merged_nodes.append(CodeNode(
                        type="code",
                        content="\n".join(json_buffer),
                        language="json",
                        headers=json_headers
                    ))
                    in_json_block = False
                    json_buffer, original_nodes_buffer = [], []
                    depth = 0

        if in_json_block:
            merged_nodes.extend(original_nodes_buffer)

        return merged_nodes