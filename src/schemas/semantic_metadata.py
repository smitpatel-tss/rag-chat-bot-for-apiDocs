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
        
        if self.payload_purpose not in ["REQUEST_BODY", "SUCCESS_RESPONSE", "ERROR_RESPONSE"]:
            self.mandatory_fields_extracted = []
            self.optional_fields_extracted = []
        return self