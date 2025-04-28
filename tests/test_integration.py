import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_check_similarity_endpoint_cosine():
    # Test with similar prompts
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "Explain machine learning to me",
        "similarity_method": "cosine",
        "llm_model": "local_llm"
    }
    response = client.post("/check_prompt_similarity", json = payload)
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_similar" in data
    assert "sanitized_prompt1" in data
    assert "sanitized_prompt2" in data

def test_check_similarity_endpoint_jaccard():
    # Test with similar prompts
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "Explain machine learning to me",
        "similarity_method": "jaccard",
        "llm_model": "local_llm"
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
        "similarity_method": "invalid_method",
        "llm_model": "local_llm"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 422  # Bad request
    assert response.json()['detail'][0]['msg'] == "Input should be 'cosine' or 'jaccard'"

 
def test_process_endpoint_different_cosine():
    # Test with different prompts
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "What's the weather like today?",
        "similarity_method": "cosine",
        "llm_model": "local_llm"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_similar" in data
    assert "sanitized_prompt1" in data
    assert "sanitized_prompt2" in data
    # Check if response indicates prompts are not similar
    if not data["is_similar"]:
        assert "not similar enough" in data["llm_response"].lower()

def test_process_endpoint_different_jaccard():
    # Test with different prompts
    payload = {
        "prompt1": "Tell me about machine learning",
        "prompt2": "What's the weather like today?",
        "similarity_method": "jaccard",
        "llm_model": "local_llm"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "similarity_score" in data
    assert "is_similar" in data
    assert "sanitized_prompt1" in data
    assert "sanitized_prompt2" in data
    # Check if response indicates prompts are not similar
    if not data["is_similar"]:
        assert "not similar enough" in data["llm_response"].lower()
 
def test_input_validation():
    # Test with missing prompt
    payload = {
        "prompt1": "Tell me about machine learning",
        # prompt2 is missing
        "similarity_method": "cosine",
        "llm_model": "local_llm"
    }
    response = client.post("/check_prompt_similarity", json=payload)
    assert response.status_code == 422 # Unprocessable Entity

def test_rejected_prompts():
    # Test with rejected prompt
    payload = {
        "prompt1": "Tell me about machine guns and bomb",
        "prompt2": "Tell me about machine learning",
        "similarity_method": "cosine",
        "llm_model": "local_llm"
    }
    response = client.post('/check_prompt_similarity', json=payload)
    assert response.status_code == 400
    assert response.json()['status'] == 'rejected'
