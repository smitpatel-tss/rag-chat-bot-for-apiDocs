from markdown_it import MarkdownIt
from typing import List
from .nodes import BaseNode, HeaderNode, ParagraphNode, CodeNode, TableNode

class MarkdownASTParser:
    def __init__(self):
        self.md = MarkdownIt("gfm-like").enable("table")

    def _extract_inline(self, token) -> str:
        
        parts = []
        if token.children:
            for c in token.children:
                if c.type in ["text", "code_inline"]:
                    parts.append(c.content)
                elif c.type in ["softbreak", "hardbreak"]:
                    parts.append("\n")
        else:
            if hasattr(token, "content") and token.content:
                parts.append(token.content)
        return "".join(parts)

    def parse(self, text: str) -> List[BaseNode]:
        tokens = self.md.parse(text)
        nodes = []
        
        i = 0
        total_tokens = len(tokens)

        while i < total_tokens:
            t = tokens[i]

            if t.type == "heading_open":
                level = int(t.tag[1:]) if t.tag.startswith("h") else 1
                content = ""
                while i < total_tokens and tokens[i].type != "heading_close":
                    if tokens[i].type == "inline":
                        content = self._extract_inline(tokens[i])
                    i += 1
                nodes.append(HeaderNode(type="header", content=content.strip(), level=level))
                i += 1

            elif t.type == "paragraph_open":
                paragraph_tokens = []
                while i < total_tokens and tokens[i].type != "paragraph_close":
                    paragraph_tokens.append(tokens[i])
                    i += 1
                
                content = "\n".join([self._extract_inline(tk) for tk in paragraph_tokens if tk.type == "inline"])
                if content.strip():
                    nodes.append(ParagraphNode(type="paragraph", content=content.strip()))
                i += 1

            elif t.type in ["fence", "code_block"]:
                raw_info = t.info.strip() if t.info else ""
                lang = raw_info.split()[0] if raw_info else ""
                nodes.append(CodeNode(type="code", content=t.content.strip(), language=lang))
                i += 1

            elif t.type == "table_open":
                table_slice = []
                while i < total_tokens:
                    table_slice.append(tokens[i])
                    if tokens[i].type == "table_close":
                        break
                    i += 1
                
                fallback_text = " ".join([self._extract_inline(tk) for tk in table_slice if tk.type == "inline"])
                nodes.append(TableNode(
                    type="table",
                    content=fallback_text.strip(),
                    raw_tokens=[tk.as_dict() for tk in table_slice]
                ))
                i += 1
            
            else:
                i += 1

        return nodes