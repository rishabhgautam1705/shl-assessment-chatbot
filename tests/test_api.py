from fastapi.testclient import TestClient
from backend.main import app
import pytest

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_chat_schema_compliance():
    # Test that the response perfectly matches the required schema
    payload = {
        "messages": [
            {"role": "user", "content": "I am looking for an assessment for a Java developer."}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "reply" in data
    assert "recommendations" in data
    assert "end_of_conversation" in data
    
    assert isinstance(data["reply"], str)
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["end_of_conversation"], bool)

def test_clarify_behavior():
    # If a query is very vague, the agent should not end the conversation
    # and should return empty recommendations.
    payload = {
        "messages": [
            {"role": "user", "content": "I need an assessment."}
        ]
    }
    response = client.post("/chat", json=payload)
    data = response.json()
    # It should ask a question
    assert len(data["reply"]) > 0
    # It might not end conversation if it's strictly following prompt
    # Note: Depending on the API key absence, it might return the fallback
    # If fallback triggers, end_of_conversation is True.
    if data["reply"].startswith("GEMINI_API_KEY is not set"):
        assert data["end_of_conversation"] is True
    else:
        assert data["end_of_conversation"] is False
        assert len(data["recommendations"]) == 0
