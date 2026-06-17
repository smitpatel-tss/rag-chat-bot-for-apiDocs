from langchain_core.prompts import PromptTemplate


LLM_METADATA_PROMPT = PromptTemplate(
    template="""
        You are an expert API documentation analyst.

        {format_instructions}

        DOCUMENT TITLE:
        {document_title}

        SECTION CONTEXT:
        {section_context}

        CONTENT BODY:
        {content_body}

        TASK

        1. Determine payload_purpose:
        - REQUEST_BODY
        - SUCCESS_RESPONSE
        - ERROR_RESPONSE
        - ENUMERATIONS
        - CONFIG_TABLE
        - GENERAL_INFO
        - METADATA

        Classification rules:
        - REQUEST_BODY → API request parameters
        - SUCCESS_RESPONSE → API response parameters
        - ERROR_RESPONSE → error payloads or codes
        - ENUMERATIONS → allowed values or mappings
        - CONFIG_TABLE → configuration/env settings
        - METADATA → version history, approvals, revisions
        - GENERAL_INFO → everything else

        When uncertain → GENERAL_INFO

        2. Extract fields ONLY for:
        - REQUEST_BODY
        - SUCCESS_RESPONSE
        - ERROR_RESPONSE

        Otherwise return [] for all fields.

        CRITICAL RULES:
        - Never guess or hallucinate fields
        - Extract ONLY explicitly present values
        - Version history tables MUST be METADATA
        - Do NOT treat metadata tables as API fields

        3. Mandatory fields ONLY if explicitly marked required/mandatory/must

        4. functional_description must be factual summary

        Return strictly following schema.
        """,
            input_variables=[
                "section_context",
                "document_title",
                "content_body",
                "format_instructions"
            ]
        )

LLM_COMMAN_METADATA_PROMPT=PromptTemplate(
        template="Extract the core document title and the exact document version from this API document front page cover content.\n{format_instructions}\n\nFront Page Text:\n{text}",
        input_variables=["text", ""]
        )