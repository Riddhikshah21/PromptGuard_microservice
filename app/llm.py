import openai
from sanitize import sanitize_output_response
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

def get_llm_response(prompt):

    api_key = os.getenv('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt}
        ]
    )
    llm_response = response.choices[0].message.content
    return llm_response


def get_local_llm_response(prompt: str) -> str:
    response = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return response[0]['generated_text']