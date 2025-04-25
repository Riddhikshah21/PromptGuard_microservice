import re
from typing import Dict
from collections import defaultdict
from better_profanity import profanity
from unidecode import unidecode
import os

profanity.load_censor_words()

# List of disallowed phrases
DISALLOWED_PHRASES = [
    "kill", "bomb", "attack", "suicide", "nazi", "rape", "execute", "murder", "harm yourself", "stab"
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
MAX_QUERY_LENGTH = os.getenv("MAX_QUERY_LENGTH")

# Content moderator to check and calculate profanity score and input risk
class ContentModerator:
    
    def __init__(self):
        self.risk_threshold = 0.7
        self.output_risk_threshold = 0.6
        self.individual_risk_thresholds = {
            'injection': 0.5,  # Threshold for injection risk
            'profanity': 0.3,  # Threshold for profanity risk
            'disallowed_phrase': 0.4  # Threshold for disallowed phrases
        }
        self.risk_weights = {
            'injection': 0.5,
            'profanity': 0.3,
            'disallowed_phrase': 0.4
        }

    def profanity_score(self, text: str) -> int:
        """Calculate profanity score of input prompt"""
        words = text.split()
        bad_words = [word for word in words if profanity.contains_profanity(word)]
        return len(bad_words) / len(words) if words else 0.0
        
    def calculate_risk(self, text: str) -> Dict:
        """Calculate risk score of the input prompt"""
        risk_score = defaultdict(float)
        
        # Calculating profanity score
        risk_score['profanity'] = self.profanity_score(text)
       
        # Matching disallowed phrase  
        found_phrases = sum(1 for p in DISALLOWED_PHRASES if p in text.lower())
        risk_score['disallowed_phrase'] = min(found_phrases * 0.5, 1.0)
        
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
    """Process and sanitize input prompt"""
    moderator = ContentModerator()
    
    # Initial sanitization
    sanitized_prompt = unidecode(prompt)
    sanitized_prompt = re.sub(r"[^\w\s.,!?'\"]", '', sanitized_prompt)
    sanitized_prompt = re.sub(r"\s+", ' ', sanitized_prompt).strip()[:int(MAX_QUERY_LENGTH)]

    # Injection pattern removal
    for pattern in INJECTION_PATTERNS:
        sanitized_prompt = re.sub(pattern, "[redacted]", sanitized_prompt, flags=re.IGNORECASE)

    # Calculate risk score
    risk_result = moderator.calculate_risk(prompt)
    category_risks = risk_result['category_risks']
    print("Category risks:", category_risks)
    
    # Reject if any individual risk exceeds the threshold
    for category, risk_score in category_risks.items():
        if risk_score > moderator.individual_risk_thresholds.get(category, 1.0):
            return {
                "action": "reject",
                "risk_score": risk_score,
                "category": category,
                "message": f"Content violates safety policy: {category} risk is too high"
            }

    # Reject if total risk exceeds the overall threshold
    if risk_result['total_risk'] > moderator.risk_threshold:
        return {
            "action": "reject",
            "risk_score": risk_result['total_risk'],
            "risk_breakdown": risk_result['category_risks'],
            "message": "Content violates safety policies"
        }
    
    # Content sanitization
    if any(phrase in sanitized_prompt.lower() for phrase in DISALLOWED_PHRASES):
        sanitized_prompt = re.sub(
            r"\b(" + "|".join(re.escape(phrase) for phrase in DISALLOWED_PHRASES) + r")\b",
            "[redacted]",
            sanitized_prompt,
            flags=re.IGNORECASE
        )
    
    if profanity.contains_profanity(sanitized_prompt):
        sanitized_prompt = profanity.censor(sanitized_prompt)  # Censor profanity

    return {"action": "accept", "sanitized_prompt": sanitized_prompt}

def contains_disallowed_phrases(text: str) -> bool:
    return any(phrase in text.lower() for phrase in DISALLOWED_PHRASES)

def sanitize_output_response(response: str) -> Dict:
    """Sanitize and validate LLM generated content"""
    moderator = ContentModerator()
    
    # Initial sanitization
    sanitized_output = unidecode(response)
    sanitized_output = re.sub(r"[^\w\s.,!?'\"\n-]", '', sanitized_output)

    # Calculae risk 
    risk_result = moderator.calculate_risk(sanitized_output)
    
    # High-risk rejection
    if risk_result['total_risk'] > moderator.output_risk_threshold:
        return {
            "action": "reject",
            "risk_score": risk_result['total_risk'],
            "risk_breakdown": risk_result['category_risks'],
            "message": "Content violates safety policies"
        }

    # Content sanitization
    if contains_disallowed_phrases(sanitized_output):
        sanitized_output = re.sub(
            r"\b(" + "|".join(re.escape(p) for p in DISALLOWED_PHRASES) + r")\b",
            "[redacted]",
            sanitized_output,
            flags=re.IGNORECASE
        )
    
    if profanity.contains_profanity(sanitized_output):
        sanitized_output = profanity.censor(sanitized_output)

    return {"action": "accept", "sanitized_output":sanitized_output}