from markdown_it import MarkdownIt
from typing import List
from .nodes import BaseNode, HeaderNode, ParagraphNode, CodeNode, TableNode, ListNode

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
                parts=[]
                while i < total_tokens and tokens[i].type != "heading_close":
                    if tokens[i].type == "inline":
                        parts.append(self._extract_inline(tokens[i]))
                    i += 1
                
                content = "".join(parts)
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
                table_lines = []
                current_row_cells = []

                while i < total_tokens:
                    current_token = tokens[i]
                    table_slice.append(current_token)

                    if current_token.type == "table_close":
                        break

                    if current_token.type == "tr_open":
                        current_row_cells = []

                    elif current_token.type == "tr_close":
                        if current_row_cells:
                            table_lines.append(" | ".join(current_row_cells))

                    elif current_token.type in ["th_open", "td_open"]:
                        cell_content = ""
                        i += 1
                        while i < total_tokens and tokens[i].type not in ["th_close", "td_close"]:
                            table_slice.append(tokens[i])
                            if tokens[i].type == "inline":
                                cell_content += self._extract_inline(tokens[i])
                            i += 1

                        current_row_cells.append(cell_content.strip())
                        table_slice.append(tokens[i])


                    i += 1
                nodes.append(TableNode(
                    type="table",
                    content="\n".join(table_lines),
                    raw_tokens=[tk.as_dict() for tk in table_slice],
                    metadata={
                        "row_count": len(table_lines),
                        "column_count": max(
                            (len(r.split(" | ")) for r in table_lines),
                            default=0
                        )
                    }
                ))
            
            elif t.type in ["bullet_list_open", "ordered_list_open"]:
                list_slice = []
                items = []
                current_item = []
                capture_item = False

                while i < total_tokens:
                    token = tokens[i]
                    list_slice.append(token)

                    if token.type == "list_item_open":
                        current_item = []
                        capture_item = True

                    elif token.type == "list_item_close":
                        if current_item:
                            items.append("".join(current_item).strip())
                        capture_item = False

                    elif token.type in ["paragraph_open", "paragraph_close"]:
                        pass

                    elif token.type == "inline" and capture_item:
                        current_item.append(self._extract_inline(token))

                    elif token.type in ["bullet_list_close", "ordered_list_close"]:
                        break

                    i += 1

                nodes.append(ListNode(
                    type="list",
                    content="\n".join(f"- {item}" for item in items if item),
                    raw_tokens=[t.as_dict() for t in list_slice]
                ))
            
            else:
                i += 1

        return nodes