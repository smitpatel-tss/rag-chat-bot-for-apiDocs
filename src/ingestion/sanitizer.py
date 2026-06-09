import re
from typing import List
from .nodes import BaseNode, ParagraphNode

class GenericDocumentSanitizer:
    
    def sanitize(self, nodes: List[BaseNode]) -> List[BaseNode]:
        for node in nodes:
            if isinstance(node, ParagraphNode):
                
                node.content = re.sub(r'[“”]', '"', node.content)
                
                stripped = node.content.strip()
                if stripped.startswith('"') and not stripped.endswith('"'):
                    if not stripped.endswith((',', '}', ']')):
                        node.content = node.content.rstrip() + '"'
                        
        return nodes