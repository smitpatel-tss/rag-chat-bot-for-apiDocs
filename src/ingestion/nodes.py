from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class BaseNode(BaseModel):

    type: str
    content: str
    title: str = "Untitled Document" 
    headers: Dict[int, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class HeaderNode(BaseNode):
    level: int

class ParagraphNode(BaseNode):
    pass

class CodeNode(BaseNode):
    language: str = ""

class TableNode(BaseNode):
    raw_tokens: List[Dict[str, Any]] = Field(default_factory=list)

class ListNode(BaseNode):
    raw_tokens: List[Dict[str, Any]] = Field(default_factory=list)

class SmartChunk(BaseModel):

    doc_id: str
    chunk_id: str
    
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    token_count: int
    
    source_url: str  
    anchor: str
    
    title: str
    document_version: str
    parent_header: str
    section_path: List[str]
    
    chunk_type: str 
    content: str
    
    attributes: Dict[str, Any] = Field(default_factory=dict)