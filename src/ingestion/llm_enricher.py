import time
from pydantic import BaseModel, Field, model_validator
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from src.ingestion.nodes import BaseNode, CodeNode, TableNode
from src.prompts.factory import PromptFactory
from src.schemas.semantic_metadata import SemanticApiMetadata

from pydantic import BaseModel, Field, model_validator
from typing import List

class LLMMetadataEnricher:
    def __init__(self, llm_provider):
        self.model = llm_provider.get_model()
        self.parser = PydanticOutputParser(pydantic_object=SemanticApiMetadata)
        self.prompt_template=PromptFactory.get("llm_metadata_enricher", "v1")

    def enrich_nodes_with_llm(self, nodes: List[BaseNode], max_retries: int = 3) -> List[BaseNode]:
        

        chain = self.prompt_template | self.model | self.parser

        for node in nodes:
            if isinstance(node, (CodeNode, TableNode)):
                section_context = " > ".join([v for k, v in sorted(node.headers.items()) if isinstance(k, int)])
                document_title = getattr(node, "title", "Unknown Title")
                
                success = False
                for attempt in range(max_retries):
                    try:
                        llm_data = chain.invoke({
                        "section_context": section_context,
                        "document_title": document_title,
                        "content_body": node.content,
                        "format_instructions": self.parser.get_format_instructions()
                        })
                        
                        node.metadata["semantic_analysis"] = llm_data.model_dump()
                        success = True
                        break 
                        
                    except Exception as e:
                        print(f" [Attempt {attempt + 1}/{max_retries}] Semantic Analysis failed parsing for node. Retrying... Error: {str(e)}")
                        time.sleep(1)

                if not success:
                    raise ValueError(f"CRITICAL: Failed to validate Semantic Schema for section '{section_context}' after 3 retries.")
                    
        return nodes