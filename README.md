# PromptGuard_microservice
A scalable microservice that evaluates text similarity between prompts and conditionally forwards one to an LLM, with comprehensive input/output sanitization and testing.
Features

Input Sanitization: Detects and filters harmful content, prompt injections, HTML/XML tags, sensitive data, and more
Text Similarity Analysis: Multiple similarity algorithms:

Cosine similarity 
Jaccard similarity 
Sentence similarity 

LLM Integration: Forwards prompts to LLM API if similarity threshold is met
Output Sanitization: Ensures responses are safe by detecting refusal patterns, data leaks, and inappropriate content
REST API: Clean, well-documented API built with FastAPI
Comprehensive Testing: Unit, integration, and load tests included
Containerization: Ready for deployment with Docker