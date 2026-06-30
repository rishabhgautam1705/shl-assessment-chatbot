from fastapi import FastAPI, HTTPException
from backend.models import ChatRequest, ChatResponse
from backend.chat import ChatController

app = FastAPI(title="SHL Assessment Chatbot")
chat_controller = ChatController()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        return chat_controller.handle_chat(request.messages)
    except Exception as e:
        # Avoid 500s unless catastrophic, wrap in a structured error if possible
        # but for safety, raise HTTPException
        raise HTTPException(status_code=500, detail=str(e))
