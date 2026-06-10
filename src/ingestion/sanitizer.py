import re
from typing import List
from .nodes import BaseNode, ParagraphNode

class GenericDocumentSanitizer:
    
    def sanitize(self, nodes: List[BaseNode]) -> List[BaseNode]:
        for node in nodes:

            if isinstance(node, ParagraphNode):

                content = node.content

                content = content.translate(str.maketrans({
                    "“": '"', "”": '"',
                    "‘": "'", "’": "'"
                }))

                content = content.replace('\u00a0', ' ')

                content = re.sub(r'[ \t]+', ' ', content)
                content = re.sub(r'\n{2,}', '\n', content)

                node.content = content.strip()

        return nodes