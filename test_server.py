import pytest
import os
import sys
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock the model loading
class MockModelConfig:
    def __init__(self, model_path, device="cpu", max_tokens=2048):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.model_info = {
            "model_args": {"n_layer": 2, "n_head": 2, "n_embd": 64},
            "config": {"dataset": "cogprime"},
            "iter_num": 0,
            "best_val_loss": 999.0,
            "checkpoint_path": model_path
        }
    
    def load_model(self):
        return True
    
    def generate(self, prompt, max_new_tokens=500, temperature=0.7, top_k=200, stream=False, callback=None):
        if stream and callback:
            callback("This ")
            callback("is ")
            callback("a ")
            callback("test ")
            callback("response.")
            return "This is a test response."
        return "This is a test response."

# Patch the ModelConfig in server.py
with patch('server.ModelConfig', MockModelConfig):
    from server import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "status" in response.json()

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_loaded" in response.json()

def test_chat():
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Hello, NanoCog!"}
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "top_k": 200,
            "stream": False
        }
    )
    assert response.status_code == 200
    assert "text" in response.json()
    assert "model" in response.json()
    assert "created_at" in response.json()
    assert "tokens_generated" in response.json()

@pytest.mark.asyncio
async def test_chat_stream():
    response = client.post(
        "/chat/stream",
        json={
            "messages": [
                {"role": "user", "content": "Hello, NanoCog!"}
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "top_k": 200,
            "stream": True
        }
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    
    # Check that we get some data in the stream
    content = response.content.decode("utf-8")
    assert "data:" in content