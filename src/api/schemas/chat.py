from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(description="Either 'user' or 'assistant'")
    content: str = Field(description="The textual content of the message")

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="The current question from the user")
    current_page_path: str = Field(description="The URL path the user is viewing, e.g., '/docs/users'")
    history: list[ChatMessage] = Field(default_factory=list, description="The last few turns of conversation")

class Citation(BaseModel):
    title: str
    url: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[Citation] = Field(default_factory=list)