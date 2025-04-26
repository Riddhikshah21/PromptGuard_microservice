import openai
import os
from dotenv import load_dotenv

load_dotenv()

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "tiiuae/falcon-rw-1b"  

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def get_llm_response(prompt: str) -> str:
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  
    
    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
        
    except Exception as e: 
        raise RuntimeError(f"LLM request failed: {str(e)}")

def get_local_llm_response(prompt: str) -> str:
    # Get local llm model response
    response = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return response[0]['generated_text']