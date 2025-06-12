import json
import torch
import os
import requests
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
import uvicorn

if not os.path.exists("finance_questions_formatted.json"):
    print("Downloading JSON dataset...")
    response = requests.get("https://github.com/goelvatsal/goelvatsal.github.io/releases/download/dataset/finance_questions_formatted.json")
    with open("finance_questions_formatted.json", "wb") as f:
        f.write(response.content)
    print("Download complete.")

with open("finance_questions_formatted.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

device = 0 if torch.cuda.is_available() else -1
text_gen = hf_pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# FastAPI setup
app = FastAPI()

# Enable CORS so React frontend can call it
app.add_middleware(
   CORSMiddleware,
   allow_origins=["http://localhost:5173"],  # your Vite dev server
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Pydantic model for request body
class Question(BaseModel):
   user_q: str

# Endpoint
@app.post("/ask")
def ask_question(data: Question):
   try:
       prompt = f"Explain the following finance concept in a detailed and informative way:\n\n{data.user_q}"
       result = text_gen(
           prompt,
           max_new_tokens=250,
           min_new_tokens=25,
           temperature=0.6,
           top_p=0.9,
           repetition_penalty=1.2,
           do_sample=True
       )[0]["generated_text"]
       return {"answer": result.strip()}
   except Exception as e:
       return {"answer": f"Something went wrong: {str(e)}"}

# Run server
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=10000)
