import pytest
from unittest.mock import patch
from apps.sanitize import (
    ContentModerator, sanitize_input_prompt, sanitize_output_response,
    contains_disallowed_phrases
)

class TestContentModerator:
    def setup_method(self):
        self.mod = ContentModerator()

    def test_initialization(self):
        assert self.mod.risk_threshold == 0.7
        assert self.mod.output_risk_threshold == 0.6
        assert 'profanity' in self.mod.risk_weights

    def test_profanity_score(self):
        assert self.mod.profanity_score("") == 0.0
        assert self.mod.profanity_score("clean text") == 0.0
        with patch('better_profanity.profanity.contains_profanity', side_effect=lambda w: w in ['bad']):
            assert self.mod.profanity_score("bad word here") == 1/3

    def test_risk_calculation(self):
        with patch.object(self.mod, 'profanity_score', return_value=0.5):
            result = self.mod.calculate_risk("some profane text")
            assert result['total_risk'] > 0
            assert result['category_risks']['profanity'] == 0.5

        with patch('apps.sanitize.DISALLOWED_PHRASES', ['bomb']):
            result = self.mod.calculate_risk("use a bomb")
            assert result['category_risks']['disallowed_phrase'] > 0


class TestHelpers:
    def test_disallowed_phrase_detection(self):
        with patch('apps.sanitize.DISALLOWED_PHRASES', ['bomb']):
            assert contains_disallowed_phrases("drop a bomb")
            assert not contains_disallowed_phrases("all good")

    def test_case_insensitivity(self):
        with patch('apps.sanitize.DISALLOWED_PHRASES', ['bomb']):
            assert contains_disallowed_phrases("BOMB")

class TestSanitizeInput:
    def test_safe_prompt(self):
        with patch('apps.sanitize.ContentModerator.calculate_risk', return_value={'total_risk': 0.1, 'category_risks': {}}):
            result = sanitize_input_prompt("Hello world!")
            assert result['action'] == 'accept'

    def test_unsafe_prompt(self):
        with patch('apps.sanitize.ContentModerator.calculate_risk', return_value={'total_risk': 0.9, 'category_risks': {'profanity': 0.5}}):
            result = sanitize_input_prompt("bad input")
            assert result['action'] == 'reject'

    def test_redaction_and_censor(self):
        with patch('apps.sanitize.ContentModerator.calculate_risk', return_value={'total_risk': 0.3, 'category_risks': {}}):
            with patch('better_profanity.profanity.contains_profanity', return_value=True), \
                 patch('better_profanity.profanity.censor', return_value="clean ****"):
                result = sanitize_input_prompt("dirty word")
                assert "****" in result['sanitized_prompt']

class TestSanitizeOutput:
    def test_safe_output(self):
        with patch('apps.sanitize.ContentModerator.calculate_risk', return_value={'total_risk': 0.1, 'category_risks': {}}):
            result = sanitize_output_response("Looks good.")
            assert result['action'] == 'accept'

    def test_unsafe_output(self):
        with patch('apps.sanitize.ContentModerator.calculate_risk', return_value={'total_risk': 0.9, 'category_risks': {'profanity': 0.8}}):
            result = sanitize_output_response("bad output")
            assert result['action'] == 'reject'