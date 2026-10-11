"""PyTorch server: batched requests + KV-cache decoding.  uvicorn serve:app --port 8000"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from bpe import BPE
from tinyllm import load

tok, model = BPE.load(), load()
app = FastAPI(title="TinyLLM-10K")

class Req(BaseModel):
    prompts: list[str] = Field(default=["the "], max_length=64)
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 8

@app.post("/generate")
def generate(r: Req):
    return {"texts": model.generate(tok, r.prompts, r.max_new_tokens, r.temperature, r.top_k)}