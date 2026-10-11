"""torch-free server (onnxruntime + numpy).  uvicorn serve_onnx:app --port 8000"""
import numpy as np, onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field
from bpe import BPE

CTX = 256                                   # must match tinyllm.CTX
tok = BPE.load()
sess = ort.InferenceSession("tinyllm.onnx", providers=["CPUExecutionProvider"])
rng = np.random.default_rng()
app = FastAPI(title="TinyLLM-1M-ONNX")

def run(prompts, n, temp, top_k):
    ids = [(tok.encode(p.rstrip()) or [0])[-(CTX - 1):] for p in prompts]
    B, Lp = len(ids), max(map(len, ids))
    x = np.array([[0] * (Lp - len(r)) + r for r in ids], dtype=np.int64)
    km = np.array([[False] * (Lp - len(r)) + [True] * len(r) for r in ids])
    for _ in range(n):
        xw, kw = x[:, -CTX:], km[:, -CTX:]                      # sliding window
        pos = np.maximum(np.cumsum(kw, 1) - 1, 0).astype(np.int64)
        lg = sess.run(None, {"idx": xw, "kmask": kw, "pos": pos})[0][:, -1, :tok.size]
        lg = lg / max(temp, 1e-4)
        kth = np.sort(lg, 1)[:, -min(top_k, tok.size)][:, None]
        p = np.exp(np.where(lg < kth, -np.inf, lg) - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        nxt = np.array([rng.choice(tok.size, p=p[b]) for b in range(B)], dtype=np.int64)
        x = np.concatenate([x, nxt[:, None]], 1)
        km = np.concatenate([km, np.ones((B, 1), bool)], 1)
    return [tok.decode([t for t, v in zip(x[b], km[b]) if v]) for b in range(B)]

class Req(BaseModel):
    prompts: list[str] = Field(default=["the "], max_length=64)
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 8

@app.post("/generate")
def generate(r: Req):
    return {"texts": run(r.prompts, r.max_new_tokens, r.temperature, r.top_k)}
