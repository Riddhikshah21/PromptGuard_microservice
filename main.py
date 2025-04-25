# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from apps.similarity import cosine_similarity_tfidf, jaccard_similarity
# # , sentence_transformer_similarity
# from apps.sanitize import sanitize_input_prompt, sanitize_output_response
# from apps.llm import get_llm_response, get_local_llm_response
# from fastapi.responses import JSONResponse

# app = FastAPI(title="PromptGuard", version="1.0")

# SIMILARITY_THRESHOLD = 0.5

# # Request body
# class PromptRequest(BaseModel):
#     prompt1: str
#     prompt2: str

# # Response body
# class PromptResponse(BaseModel):
#     llm_response: str = None

# @app.get("/")
# def root():
#     return {"message": "Server is running"}

# @app.post("/check_prompt_similarity", response_model=PromptResponse)
# def check_prompt_similarity(payload: PromptRequest):
    
#     try:
#         sanitized_prompt1 = sanitize_input_prompt(payload.prompt1)
#         sanitized_prompt2 = sanitize_input_prompt(payload.prompt2)

#         # Rejecting prompts if the prompts can not be sanitized further
#         if sanitized_prompt1.get('action') == "reject" or sanitized_prompt2.get('action') == "reject":
#             return JSONResponse(
#                 status_code=400, 
#                 content={"status": "rejected", "message": "Content violates safety policies."}
#             )
#         else:
#             sanitized_prompt1 = sanitized_prompt1.get("sanitized_prompt")
#             sanitized_prompt2 = sanitized_prompt2.get("sanitized_prompt")

        
#     except ValueError as ve:
#         raise HTTPException(status_code=400, detail=str(ve))

#     # Calculating similarity between input prompts
#     cosine_similarity = cosine_similarity_tfidf(sanitized_prompt1, sanitized_prompt2)
#     jac_similarity = jaccard_similarity(sanitized_prompt1, sanitized_prompt2)
#     # sentence_similarity = sentence_transformer_similarity(sanitized_prompt1, sanitized_prompt2)

#     print('Cosine similarity:', cosine_similarity)
#     print('Jaccard similarity:', jac_similarity)
#     # print('Sentence similarity:', sentence_similarity)

#     if cosine_similarity >= SIMILARITY_THRESHOLD:
#         # Send the sanitized prompt to LLM for generating response
#         # llm_response = get_llm_response(sanitized_prompt1)
#         llm_response = get_local_llm_response(sanitized_prompt1)
#         sanitized_response = sanitize_output_response(llm_response)
#         return PromptResponse(
#             llm_response=sanitized_response
#         )
    
#     # If similarity does not meet the threshold, reject it
#     return JSONResponse(
#         status_code=400, 
#         content={"status": "rejected", "message": "Prompts are not similar to each other."}
#     )

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apps.similarity import cosine_similarity_tfidf, jaccard_similarity
from apps.sanitize import sanitize_input_prompt, sanitize_output_response
from apps.llm import get_llm_response, get_local_llm_response
from fastapi.responses import JSONResponse

app = FastAPI(title="PromptGuard", version="1.0")

SIMILARITY_THRESHOLD = 0.1

# Request body
class PromptRequest(BaseModel):
    prompt1: str
    prompt2: str
    similarity_method: str = "cosine"  # Default to "cosine"

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

        sanitized_prompt1 = sanitized_prompt1.get("sanitized_prompt")
        sanitized_prompt2 = sanitized_prompt2.get("sanitized_prompt")

        # Access similarity_method directly
        similarity_method = payload.similarity_method  

        # Calculate similarity based on the method
        if similarity_method == "cosine":
            similarity_score = cosine_similarity_tfidf(sanitized_prompt1, sanitized_prompt2)
        elif similarity_method == "jaccard":
            similarity_score = jaccard_similarity(sanitized_prompt1, sanitized_prompt2)
        else:
            raise HTTPException(status_code=400, detail="Invalid similarity method")

        if similarity_score >= SIMILARITY_THRESHOLD:

            llm_response = get_local_llm_response(sanitized_prompt1)

            response  = sanitize_output_response(llm_response)
            sanitized_response = response['sanitized_output']

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
