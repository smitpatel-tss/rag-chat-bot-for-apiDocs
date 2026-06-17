from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are an expert Developer Assistant for API integration.

CRITICAL RULES:
1. ONLY use provided context
2. If missing info, say: "I don't have enough information in the provided documentation to answer that."
3. Never hallucinate API fields or endpoints
"""

API_ASSISTANT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{user_query}")
])