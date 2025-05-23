from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.similarity import cosine_similarity_tfidf, jaccard_similarity
from app.sanitize import sanitize_input_prompt, sanitize_output_response
from app.llm import get_llm_response, get_local_llm_response
from fastapi.responses import JSONResponse
from typing import Literal

app = FastAPI(title="PromptGuard", version="1.0")

SIMILARITY_THRESHOLD = 0.4

# Request body
class PromptRequest(BaseModel):
    prompt1: str
    prompt2: str
    # Restrict llm_model and similarity method to specific options. Default set to "cosine"
    similarity_method: Literal["cosine", "jaccard"] = "cosine"
    llm_model: Literal["openai", "local_llm"] = "local_llm"

# Response body
class PromptResponse(BaseModel):
    status_code: int = None
    llm_response: str = None
    similarity_score: float = None
    is_similar: bool = False
    sanitized_prompt1: str = None
    sanitized_prompt2: str = None


@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/check_prompt_similarity", response_model=PromptResponse)
def check_prompt_similarity(payload: PromptRequest):
    try:
        sanitized_prompt1 = sanitize_input_prompt(payload.prompt1)
        sanitized_prompt2 = sanitize_input_prompt(payload.prompt2)

        # Rejecting prompts if the prompts cannot be sanitized further
        if sanitized_prompt1.get('action') == "reject" or sanitized_prompt2.get('action') == "reject":
            return JSONResponse(
                status_code=400, 
                content={"status": "rejected", "message": "Content violates safety policies."}
            )

        # Accepted prompts after sanitization
        sanitized_prompt1 = sanitized_prompt1.get("sanitized_prompt")
        sanitized_prompt2 = sanitized_prompt2.get("sanitized_prompt")

        similarity_method = payload.similarity_method  

        # Calculate similarity based on the similarity_method in payload
        if similarity_method == "cosine":
            similarity_score = cosine_similarity_tfidf(sanitized_prompt1, sanitized_prompt2)
        elif similarity_method == "jaccard":
            similarity_score = jaccard_similarity(sanitized_prompt1, sanitized_prompt2)
        else:
            raise HTTPException(status_code=422, detail="Invalid similarity method")

        # Compare similarity score and threshold defined
        if similarity_score >= SIMILARITY_THRESHOLD:
            if payload.llm_model == 'openai':
                llm_response = get_llm_response(sanitized_prompt1)
            elif payload.llm_model == 'local_llm':
                llm_response = get_local_llm_response(sanitized_prompt1)
            response  = sanitize_output_response(llm_response)
            sanitized_response = response['sanitized_output']
            # Return santized output LLM response
            return PromptResponse(
                status_code=200,
                llm_response=sanitized_response,
                similarity_score=similarity_score,
                is_similar=True,
                sanitized_prompt1=sanitized_prompt1,
                sanitized_prompt2=sanitized_prompt2
            )
        
        return PromptResponse(
            status_code=200,
            llm_response="The prompts are not similar enough to generate a meaningful response.",
            similarity_score=similarity_score,
            is_similar=False,
            sanitized_prompt1=sanitized_prompt1,
            sanitized_prompt2=sanitized_prompt2
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
