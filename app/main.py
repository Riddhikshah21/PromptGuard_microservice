from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from similarity import cosine_similarity_tfidf, jaccard_similarity, sentence_transformer_similarity
from sanitize import sanitize_input_prompt, sanitize_output_response

app = FastAPI(title="PromptGuard", version="1.0")

SIMILARITY_THRESHOLD = 0.8

# Request body
class PromptRequest(BaseModel):
    prompt1: str
    prompt2: str

# Response body
class PromptResponse(BaseModel):
    llm_response: str = None

@app.get("/")
def root():
    return {"message": "Server is running"}

@app.get("/health_check")
def health_check():
    return {"status": "ok"}

@app.post("/check_prompt_similarity", response_model=PromptResponse)
def check_prompt_similarity(payload: PromptRequest):
    try:
        # sanitized_prompt1 = sanitize_input_prompt(payload.prompt1)
        # sanitized_prompt2 = sanitize_input_prompt(payload.prompt2)
        sanitized_prompt1 = payload.prompt1
        sanitized_prompt2 = payload.prompt2
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    cosine_similarity = cosine_similarity_tfidf(sanitized_prompt1, sanitized_prompt2)
    jac_similarity = jaccard_similarity(sanitized_prompt1, sanitized_prompt2)
    sentence_similarity = sentence_transformer_similarity(sanitized_prompt1, sanitized_prompt2)

    print('Cosine similarity:', cosine_similarity)
    print('Jaccard similarity:', jac_similarity)
    print('Sentence similarity:', sentence_similarity)
    
    if sentence_similarity >= SIMILARITY_THRESHOLD:
        # Send the sanitized prompt to LLM for generating response
        llm_response = llm(sanitized_prompt1)
        sanitized_response = sanitize_output_response(llm_response)
        return PromptResponse(
            llm_response=sanitized_response
        )