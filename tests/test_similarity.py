import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

# Import the functions to test
from apps.similarity import cosine_similarity_tfidf, jaccard_similarity
# ,sentence_transformer_similarity


class TestCosineSimilarityTfidf:
    def test_identical_texts(self):
        """Test cosine similarity with identical texts"""
        text1 = "This is a test sentence"
        text2 = "This is a test sentence"
        similarity = cosine_similarity_tfidf(text1, text2)
        assert similarity == pytest.approx(1.0)
    
    def test_completely_different_texts(self):
        """Test cosine similarity with completely different texts"""
        text1 = "This is about apples"
        text2 = "Those are oranges and bananas"
        similarity = cosine_similarity_tfidf(text1, text2)
        assert similarity < 0.5
    
    def test_partially_similar_texts(self):
        """Test cosine similarity with partially similar texts"""
        text1 = "I like machine learning algorithms"
        text2 = "Machine learning is interesting"
        similarity = cosine_similarity_tfidf(text1, text2)
        assert 0 < similarity < 1
    
    def test_case_insensitivity(self):
        """Test if the similarity function is case insensitive"""
        text1 = "This Is A Test"
        text2 = "this is a test"
        similarity = cosine_similarity_tfidf(text1, text2)
        assert similarity == pytest.approx(1.0)
    
    def test_with_punctuation(self):
        """Test with texts containing punctuation"""
        text1 = "Hello, world! How are you?"
        text2 = "Hello world. How are you"
        similarity = cosine_similarity_tfidf(text1, text2)
        assert similarity > 0.9
    
    def test_with_empty_strings(self):
        """Test with empty strings - should handle this case gracefully"""
        with pytest.raises(Exception):
            # This will likely raise an exception, which is expected behavior
            cosine_similarity_tfidf("", "")


# class TestSentenceTransformerSimilarity:
#     def test_with_mocked_transformer(self):
#         """Test sentence transformer similarity with mocked model"""
#         # Create mock embeddings that are identical
#         mock_embedding1 = torch.tensor([[0.1, 0.2, 0.3]])
#         mock_embedding2 = torch.tensor([[0.1, 0.2, 0.3]])
#         # Expected cosine similarity is 1.0 for identical vectors
#         mock_similarity = torch.tensor([[1.0]])
        
#         with patch('sentence_transformers.SentenceTransformer') as mock_model:
#             instance = mock_model.return_value
#             instance.encode.side_effect = [mock_embedding1, mock_embedding2]
            
#             with patch('sentence_transformers.util.pytorch_cos_sim', return_value=mock_similarity):
#                 similarity = sentence_transformer_similarity("Text 1", "Text 2")
#                 assert similarity == 1.0
    
#     def test_with_orthogonal_vectors(self):
#         """Test with orthogonal vectors (should return 0 similarity)"""
#         mock_embedding1 = torch.tensor([[1.0, 0.0]])
#         mock_embedding2 = torch.tensor([[0.0, 1.0]])
#         # Cosine similarity of orthogonal vectors is 0
#         mock_similarity = torch.tensor([[0.0]])
        
#         with patch('sentence_transformers.SentenceTransformer') as mock_model:
#             instance = mock_model.return_value
#             instance.encode.side_effect = [mock_embedding1, mock_embedding2]
            
#             with patch('sentence_transformers.util.pytorch_cos_sim', return_value=mock_similarity):
#                 similarity = sentence_transformer_similarity("Text 1", "Text 2")
#                 assert similarity == 0.0
    
#     def test_with_partly_similar_vectors(self):
#         """Test with vectors that are partly similar"""
#         mock_embedding1 = torch.tensor([[0.8, 0.6]])
#         mock_embedding2 = torch.tensor([[0.6, 0.8]])
#         # Calculated similarity value
#         mock_similarity = torch.tensor([[0.96]])
        
#         with patch('sentence_transformers.SentenceTransformer') as mock_model:
#             instance = mock_model.return_value
#             instance.encode.side_effect = [mock_embedding1, mock_embedding2]
            
#             with patch('sentence_transformers.util.pytorch_cos_sim', return_value=mock_similarity):
#                 similarity = sentence_transformer_similarity("Text 1", "Text 2")
#                 assert similarity == pytest.approx(0.96)


class TestJaccardSimilarity:
    def test_identical_texts(self):
        """Test Jaccard similarity with identical texts"""
        text1 = "the quick brown fox"
        text2 = "the quick brown fox"
        similarity = jaccard_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_completely_different_texts(self):
        """Test Jaccard similarity with completely different texts"""
        text1 = "the quick brown fox"
        text2 = "lazy dog sleeps quietly"
        similarity = jaccard_similarity(text1, text2)
        assert similarity == 0.0
    
    def test_partially_overlapping_texts(self):
        """Test Jaccard similarity with partially overlapping texts"""
        text1 = "the quick brown fox jumps"
        text2 = "the fox jumps over lazily"
        # Expected: 3 common words (the, fox, jumps) out of 7 unique words
        expected = 3/7
        similarity = jaccard_similarity(text1, text2)
        assert similarity == pytest.approx(expected)
    
    def test_case_insensitivity(self):
        """Test if Jaccard similarity is case insensitive"""
        text1 = "The Quick Brown"
        text2 = "the quick brown"
        similarity = jaccard_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_with_punctuation(self):
        """Test with texts containing punctuation"""
        text1 = "Hello, world! How are you?"
        text2 = "Hello world. How are you"
        # Punctuation should be removed/ignored in tokenization
        similarity = jaccard_similarity(text1, text2)
        assert similarity == 1.0
    
    def test_with_empty_strings(self):
        """Test with empty strings - should handle this case"""
        with pytest.raises(ZeroDivisionError):
            jaccard_similarity("", "")
    
    def test_with_one_empty_string(self):
        """Test with one empty string"""
        text1 = "the quick brown fox"
        text2 = ""
        similarity = jaccard_similarity(text1, text2)
        assert similarity == 0.0