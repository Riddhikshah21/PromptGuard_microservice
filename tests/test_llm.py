import os
import pytest
from unittest.mock import patch, MagicMock
from app.llm import get_llm_response, get_local_llm_response


@patch('openai.OpenAI')
def test_get_llm_response_missing_api_key(mock_openai):
    # Test with OpenAI missing API key
    os.environ.pop('OPENAI_API_KEY', None) 
    mock_openai.side_effect = Exception("No API key provided")

    with pytest.raises(Exception) as exc_info:
        get_llm_response("This should fail due to missing key")

    assert "No API key provided" in str(exc_info.value)


@patch('openai.OpenAI')
def test_get_llm_response_success(mock_openai):
    # Test for OpenAI response Success
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Mock GPT response"
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    result = get_llm_response("test prompt")
    assert result == "Mock GPT response"


@patch('app.llm.generator')
def test_get_local_llm_response(mock_generator):
    # Test for Local LLM Success
    mock_generator.return_value = [{
        "generated_text": "I didn't know you could tell jokes like that. Want to hear another?"
    }]
    
    result = get_local_llm_response("Tell me a joke.")
    mock_generator.assert_called_once()
    assert "i didn't know you could tell jokes like that" in result.lower()


@patch('app.llm.generator')
def test_get_local_llm_response_unexpected(mock_generator):
    # Test for Local LLM Unexpected Response
    mock_generator.return_value = [{
        "generated_text": "Tell me a joke.\nThe model's response was not as expected."
    }]

    result = get_local_llm_response("Tell me a joke.")
    assert "response was not as expected" in result.lower()


@patch('openai.OpenAI')
def test_get_llm_response_generic_exception(mock_openai):
    # Test for LLM request failed
    mock_client = MagicMock()
    
    mock_client.chat.completions.create.side_effect = Exception("Unexpected failure")
    mock_openai.return_value = mock_client

    with pytest.raises(RuntimeError) as exc_info:
        get_llm_response("Trigger unexpected error")

    assert "LLM request failed: Unexpected failure" in str(exc_info.value)


@patch('app.llm.generator')
def test_missing_response_from_local_llm(mock_generator):
    # Test for Local llm missing response
    mock_generator.side_effect = Exception("Model response is missing")
    with pytest.raises(Exception) as exc_info:
        get_local_llm_response("Tell me a joke")
    assert "response is missing" in str(exc_info)


@patch('openai.OpenAI')
def test_empty_resposne_from_openai(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = []
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    with pytest.raises(Exception) as exc_info:
        get_llm_response("Tell me a joke")
    assert "LLM request failed" in str(exc_info.value)