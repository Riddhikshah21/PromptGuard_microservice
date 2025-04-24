from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from similarity import cosine_similarity_tfidf, jaccard_similarity, sentence_transformer_similarity
from sanitize import sanitize_input_prompt, sanitize_output_response
from llm import get_llm_response, get_local_llm_response
from fastapi.responses import JSONResponse

app = FastAPI(title="PromptGuard", version="1.0")

SIMILARITY_THRESHOLD = 0.6

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
        sanitized_prompt1 = sanitize_input_prompt(payload.prompt1)
        sanitized_prompt2 = sanitize_input_prompt(payload.prompt2)

        if sanitized_prompt1.get('action') == "reject" or sanitized_prompt2.get('action') == "reject":
            return JSONResponse(
                status_code=400, 
                content={"status": "rejected", "message": "Content violates safety policies."}
            )
        else:
            sanitized_prompt1 = sanitized_prompt1.get("sanitized_prompt")
            sanitized_prompt2 = sanitized_prompt2.get("sanitized_prompt")

        
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
        # llm_response = get_llm_response(sanitized_prompt1)
        llm_response = get_local_llm_response(sanitized_prompt1)
        sanitized_response = sanitize_output_response(llm_response)
        return PromptResponse(
            llm_response=sanitized_response
        )
    
    # If similarity does not meet the threshold, reject it
    return JSONResponse(
        status_code=400, 
        content={"status": "rejected", "message": "Prompts are not similar to each other."}
    )