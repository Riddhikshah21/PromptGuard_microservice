import re
from typing import Dict
from collections import defaultdict
from better_profanity import profanity
from unidecode import unidecode

profanity.load_censor_words()

# List of disallowed phrases
DISALLOWED_PHRASES = [
    "kill", "bomb", "attack", "suicide", "nsfw", "nazi", "rape", "execute",
    "drop database", "shutdown", "hack", "backdoor", "exploit", "killer"
]

# List of injection patterns to disregard
INJECTION_PATTERNS = [
    r"(ignore|disregard)\s+(all|previous)?\s*(instructions|rules)",
    r"pretend\s+(to|you are)",
    r"bypass.*filter",
    r"(as\s+an\s+AI\s+language\s+model)",
    r"<\|.*?\|>",
    r"system:\s*",
]

# Max query length
MAX_QUERY_LENGTH = 512

# Content moderator to check and calculate profanity score and input risk
class ContentModerator:
    def __init__(self):
        self.risk_threshold = 0.7
        self.output_risk_threshold = 0.6
        self.risk_weights = {
            'injection': 0.5,
            'profanity': 0.3,
            'disallowed_phrase': 0.4
        }

    def profanity_score(self, text):
        """Calculate profanity score of input prompt"""
        words = text.split()
        bad_words = [word for word in words if profanity.contains_profanity(word)]
        return len(bad_words) / len(words) if words else 0.0
        
    def calculate_risk(self, text):
        """Calculate risk score of the input prompt"""
        risk_score = defaultdict(float)
        
        # Detecting injection pattern 
        injection_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in INJECTION_PATTERNS
        )
        risk_score['injection'] = min(injection_count * 0.2, 1.0)
        
        # Calculating profanity score
        risk_score['profanity'] = self.profanity_score(text)
       
        # Matching disallowed phrase  
        found_phrases = sum(1 for p in DISALLOWED_PHRASES if p in text.lower())
        risk_score['disallowed_phrase'] = min(found_phrases * 0.33, 1.0)
        
        # Calculating total risk of the input prompt
        total_risk = sum(
            weight * risk_score[category]
            for category, weight in self.risk_weights.items()
        )
        
        return {
            'total_risk': total_risk,
            'category_risks': dict(risk_score)
        }

def sanitize_input_prompt(prompt: str) -> Dict:
    """Process and sanitize input"""
    moderator = ContentModerator()
    
    # Initial sanitization
    sanitized_prompt = unidecode(prompt)
    sanitized_prompt = re.sub(r"[^\w\s.,!?'\"]", '', sanitized_prompt)
    sanitized_prompt = re.sub(r"\s+", ' ', sanitized_prompt).strip()[:MAX_QUERY_LENGTH]

    # Injection pattern removal
    for pattern in INJECTION_PATTERNS:
        sanitized_prompt = re.sub(pattern, "[redacted]", sanitized_prompt, flags=re.IGNORECASE)

    # Calculating risk
    risk_result = moderator.calculate_risk(prompt)
    
    # Rejecting the input prompt if beyond threshold
    if risk_result['total_risk'] > moderator.risk_threshold:
        result = {
            "action": "reject",
            "risk_score": risk_result['total_risk'],
            "risk_breakdown": risk_result['category_risks'],
            "message": "Content violates safety policies"
        }
    
        return result
    # Final content sanitization
    if contains_disallowed_phrases(sanitized_prompt):
        sanitized_prompt = re.sub(
            r"\b(" + "|".join(re.escape(p) for p in DISALLOWED_PHRASES) + r")\b",
            "[redacted]",
            sanitized_prompt,
            flags=re.IGNORECASE
        )
    
    if profanity.contains_profanity(sanitized_prompt):
        sanitized_prompt = profanity.censor(sanitized_prompt)

    return {"action": "accept", "sanitized_prompt":sanitized_prompt}

def contains_disallowed_phrases(text: str) -> bool:
    return any(phrase in text.lower() for phrase in DISALLOWED_PHRASES)

def sanitize_output_response(response: str) -> Dict:
    """Sanitize and validate LLM generated content"""
    moderator = ContentModerator()
    
    # Initial sanitization
    sanitized_prompt = unidecode(response)
    sanitized_prompt = re.sub(r"[^\w\s.,!?'\"\n-]", '', sanitized_prompt)
    sanitized_prompt = re.sub(r"\s+", ' ', sanitized_prompt).strip()[:MAX_QUERY_LENGTH]

    # Risk analysis
    risk_result = moderator.calculate_risk(sanitized_prompt)
    
    # High-risk rejection
    if risk_result['total_risk'] > moderator.output_risk_threshold:
        return {
            "action": "reject",
            "risk_score": risk_result['total_risk'],
            "risk_breakdown": risk_result['category_risks'],
            "message": "Content violates safety policies"
        }

    # Content sanitization
    if contains_disallowed_phrases(sanitized_prompt):
        sanitized_prompt = re.sub(
            r"\b(" + "|".join(re.escape(p) for p in DISALLOWED_PHRASES) + r")\b",
            "[redacted]",
            sanitized_prompt,
            flags=re.IGNORECASE
        )
    
    if profanity.contains_profanity(sanitized_prompt):
        sanitized_prompt = profanity.censor(sanitized_prompt)

    return {"action": "accept", "sanitized_prompt":sanitized_prompt}
