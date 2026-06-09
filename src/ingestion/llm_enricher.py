import time
from pydantic import BaseModel, Field, model_validator
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from src.ingestion.nodes import BaseNode, CodeNode, TableNode

from pydantic import BaseModel, Field, model_validator
from typing import List

class SemanticApiMetadata(BaseModel):
    functional_description: str = Field(
        description="A clear summary of what this code or table block actually maps out."
    )
    mandatory_fields_extracted: List[str] = Field(
        description="List of API parameters explicitly marked as required."
    )
    optional_fields_extracted: List[str] = Field(
        description="List of API JSON parameters marked as optional."
    )
    payload_purpose: str = Field(
        description="Categorization tag. Choose from: REQUEST_BODY, SUCCESS_RESPONSE, ERROR_RESPONSE, ENUMERATIONS, CONFIG_TABLE, GENERAL_INFO, or METADATA."
    )

    @model_validator(mode='after')
    def enforce_empty_lists_for_non_payloads(self) -> 'SemanticApiMetadata':
        """Forcefully clears field arrays ONLY if the chunk is explicitly a non-parameter table."""
        
        if self.payload_purpose in ['GENERAL_INFO', 'ENUMERATIONS']:
            self.mandatory_fields_extracted = []
            self.optional_fields_extracted = []
        return self

class LLMMetadataEnricher:
    def __init__(self, llm_provider):
        self.model = llm_provider.get_model()
        self.parser = PydanticOutputParser(pydantic_object=SemanticApiMetadata)

    def enrich_nodes_with_llm(self, nodes: List[BaseNode], max_retries: int = 3) -> List[BaseNode]:
        
        prompt_template = PromptTemplate(
            template="""
            Analyze the following technical structure extracted from section context: '{section_context}'
            Document Title: '{document_title}'
            
            {format_instructions}
            
            Content Body:
            {content_body}
            """,
            input_variables=["section_context", "document_title", "content_body"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        
        chain = prompt_template | self.model | self.parser

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
                            "content_body": node.content
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