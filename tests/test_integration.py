import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_check_similarity_endpoint():
    # Test with similar prompts
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "Explain machine learning to me",
        "similarity_method": "cosine"
    }
    response = client.post("/check_prompt_similarity", json = payload)
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_similar" in data
    assert "sanitized_prompt1" in data
    assert "sanitized_prompt2" in data
 
def test_check_similarity_invalid_method():
    # Test with invalid similarity method
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "Explain machine learning to me",
        "similarity_method": "invalid_method"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 400  # Bad request
    assert response.json()['detail'] == "Invalid similarity method"

 
def test_process_endpoint_similar():
    # Test with similar prompts
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "Explain machine learning to me",
        "similarity_method": "cosine"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_similar" in data
    # assert "sanitized_prompt" in data['sanitized_prompt1']
    # assert "llm_response" in data
 
def test_process_endpoint_different():
    # Test with different prompts
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "What's the weather like today?",
        "similarity_method": "cosine"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_similar" in data
    # assert "sanitized_prompt" in data
    # assert "llm_response" in data
    # Check if response indicates prompts are not similar
    if not data["is_similar"]:
        assert "not similar enough" in data["llm_response"].lower()
 
def test_input_validation():
    # Test with missing prompt
    payload = {
        "prompt1": "Tell me about machine learning",
        # prompt2 is missing
        "similarity_method": "cosine"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 422 # Unprocessable Entity