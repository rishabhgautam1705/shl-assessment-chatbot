import os
from google import genai
from google.genai import types
from backend.models import ChatResponse

class Agent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        
        with open("prompts/system_prompt.txt", "r") as f:
            self.system_prompt_template = f.read()

    def generate_response(self, messages, retrieval_context) -> ChatResponse:
        if not self.client:
            return ChatResponse(
                reply="GEMINI_API_KEY is not set. Cannot process the request.",
                recommendations=[],
                end_of_conversation=True
            )
            
        system_instruction = self.system_prompt_template.replace("{retrieval_context}", retrieval_context)
        
        contents = []
        for m in messages:
            role = "user" if m.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ChatResponse,
            temperature=0.0
        )
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=config
            )
            
            # Since the response schema is enforced by structured outputs, we can parse it safely
            import json
            data = json.loads(response.text)
            return ChatResponse(**data)
            
        except Exception as e:
            return ChatResponse(
                reply=f"Error generating response: {str(e)}",
                recommendations=[],
                end_of_conversation=True
            )
