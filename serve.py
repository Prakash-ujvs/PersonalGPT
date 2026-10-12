"""PyTorch server: batched requests + KV-cache decoding.  uvicorn serve:app --port 8000"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from bpe import BPE
from tinyllm import load

tok = BPE.load()
model = load()
app = FastAPI(title="TinyLLM-1M")


class Req(BaseModel):
    prompts: list[str] = Field(default=["the "], min_length=1, max_length=64)
    max_new_tokens: int = Field(default=100, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=8, ge=1, le=256)


@app.post("/generate")
def generate(r: Req):
    return {"texts": model.generate(tok, r.prompts, r.max_new_tokens, r.temperature, r.top_k)}
