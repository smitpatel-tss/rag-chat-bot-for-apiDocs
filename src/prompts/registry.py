from src.prompts.definitions.api_assistant import API_ASSISTANT_PROMPT
from src.prompts.definitions.llm_metadata_enricher import LLM_METADATA_PROMPT, LLM_COMMAN_METADATA_PROMPT

PROMPT_REGISTRY = {
    "api_assistant": {
        "v1": API_ASSISTANT_PROMPT
    },
    "llm_metadata_enricher": {
        "v1": LLM_METADATA_PROMPT
    },
    "llm_comman_metadata_prompt":{
        "v1": LLM_COMMAN_METADATA_PROMPT
    }
}