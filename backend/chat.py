import json
from backend.agent import Agent
from backend.retriever import SemanticRetriever
from backend.catalog_loader import CatalogLoader
from backend.models import ChatResponse, Message
from typing import List

class ChatController:
    def __init__(self):
        self.loader = CatalogLoader()
        self.retriever = SemanticRetriever(self.loader)
        self.agent = Agent()

    def handle_chat(self, messages: List[Message]) -> ChatResponse:
        # 1. Extract the latest user intent to search the catalog
        # A simple approach: use the last user message as the query for semantic search
        last_user_message = next((m.content for m in reversed(messages) if m.role == "user"), "")
        
        # 2. Retrieve top matching assessments based on the latest context
        # If the conversation is long, ideally we'd extract constraints. 
        # For this assignment, semantic search on the last message (or concatenated context) works.
        # Let's concatenate all user messages to capture the full context for search.
        full_user_context = " ".join([m.content for m in messages if m.role == "user"])
        
        retrieved_items = self.retriever.search(full_user_context, top_k=10)
        
        if not retrieved_items:
            # Fallback if no specific semantic matches, just provide everything 
            # and let the LLM filter if it fits in context, or provide a default message.
            # We'll inject all for the LLM to decide (max 10 for safety if catalog is large).
            retrieved_items = self.loader.get_all()[:15]
            
        retrieval_context_str = json.dumps(retrieved_items, indent=2)
        
        # 3. Call the Agent with the conversation history and the targeted retrieval context
        response = self.agent.generate_response(messages, retrieval_context_str)
        
        return response
