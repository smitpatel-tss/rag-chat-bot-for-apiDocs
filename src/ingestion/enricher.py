import re
import time
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Set
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.ingestion.nodes import BaseNode, HeaderNode

class DocumentIdentitySchema(BaseModel):
    extracted_title: str = Field(description="The primary name or title of the API document.")
    extracted_version: str = Field(description="The version string found on the cover, e.g., v1.1.")

class HierarchyEnricher:
    def __init__(self, llm_provider):
        
        self.method_rx = re.compile(r'\b(GET|POST|PUT|DELETE|PATCH)\b', re.IGNORECASE)
        
        self.path_rx = re.compile(
            r'(?:https?://)?(?:[a-zA-Z0-9-]+\.)*trackwizz\.app/api(?:/[a-zA-Z0-9_{}.-]+)+'
        )
        
        self.model = llm_provider.get_model()
        self.parser = PydanticOutputParser(pydantic_object=DocumentIdentitySchema)

    def bootstrap_document_identity(self, raw_text: str, max_retries: int = 3) -> DocumentIdentitySchema:
        front_page_sample = raw_text[:1200]
        
        prompt_template = PromptTemplate(
            template="Extract the core document title and the exact document version from this API document front page cover content.\n{format_instructions}\n\nFront Page Text:\n{text}",
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        
        chain = prompt_template | self.model | self.parser
        
        for attempt in range(max_retries):
            try:
                return chain.invoke({"text": front_page_sample})
            except Exception as e:
                print(f"[Attempt {attempt + 1}/{max_retries}] Identity Extraction failed parsing. Retrying... Error: {str(e)}")
                time.sleep(1)
                
        raise ValueError("CRITICAL: LLM consistently failed to return valid Pydantic Identity Schema after 3 retries.")


    def extract_global_metadata(self, raw_text: str) -> Dict[str, List[str]]:
        global_methods: Set[str] = set()
        global_paths: Set[str] = set()

        for m in self.method_rx.findall(raw_text):
            global_methods.add(m.upper())

        for p in self.path_rx.findall(raw_text):

            clean_path = p.rstrip('.,)]}"\'')

            if "trackwizz.app" in clean_path:
                clean_path = clean_path.split("trackwizz.app", 1)[-1]

            segments = [s for s in clean_path.split('/') if s]

            if not segments or segments[0] != "api":
                continue

            if len(segments) < 3:
                continue

            global_paths.add(clean_path)

        return {
            "endpoint_paths": sorted(global_paths),
            "http_methods": sorted(global_methods),
        }

    def enrich(self, nodes: List[BaseNode], global_metadata: Dict[str, List[str]], identity: DocumentIdentitySchema) -> List[BaseNode]:
        current_headers: Dict[int, str] = {}
        for node in nodes:
            if isinstance(node, HeaderNode):
                level = node.level
                current_headers[level] = node.content
                keys_to_delete = [k for k in current_headers.keys() if k > level]
                for k in keys_to_delete:
                    del current_headers[k]
            
            node.headers = current_headers.copy()
            node.title = identity.extracted_title
            node.metadata = global_metadata.copy()
            node.metadata["document_version"] = identity.extracted_version
            
        return nodes