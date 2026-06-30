from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    role: str = Field(description="Role of the sender: 'user' or 'assistant'")
    content: str = Field(description="The message content")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(description="Conversation history")

class Recommendation(BaseModel):
    name: str = Field(description="Assessment name")
    url: str = Field(description="Official SHL URL")
    test_type: str = Field(description="Type of the test (e.g. Personality, Cognitive)")

class ChatResponse(BaseModel):
    reply: str = Field(description="Agent's reply to the user")
    recommendations: List[Recommendation] = Field(
        description="List of 1 to 10 recommended assessments. Empty if context is still being gathered."
    )
    end_of_conversation: bool = Field(
        description="True if the agent has gathered all constraints and provided final recommendations."
    )
