"""torch-free server (onnxruntime + numpy).  uvicorn serve_onnx:app --port 8000"""
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field
from bpe import BPE

CTX = 256                                   # must match tinyllm.CTX
tok = BPE.load()
sess = ort.InferenceSession("tinyllm.onnx", providers=["CPUExecutionProvider"])
app = FastAPI(title="TinyLLM-1M-ONNX")


def run(prompts: list[str], n: int, temp: float, top_k: int) -> list[str]:
    if not prompts or n <= 0:
        return list(prompts)

    rng = np.random.default_rng()
    ids = [(tok.encode(p.rstrip()) or [0])[-(CTX - 1):] for p in prompts]
    B = len(ids)
    Lp = max(map(len, ids))
    x = np.array([[0] * (Lp - len(r)) + r for r in ids], dtype=np.int64)
    km = np.array([[False] * (Lp - len(r)) + [True] * len(r) for r in ids])

    for _ in range(n):
        xw = x[:, -CTX:]
        kw = km[:, -CTX:]                      # sliding window
        pos = np.maximum(np.cumsum(kw, axis=1) - 1, 0).astype(np.int64)
        lg = sess.run(None, {"idx": xw, "kmask": kw, "pos": pos})[0][:, -1, :tok.size]

        if temp <= 1e-4:
            nxt = np.argmax(lg, axis=1)
        else:
            lg = lg / temp
            k = max(1, min(top_k, tok.size))
            kth = np.sort(lg, axis=1)[:, -k][:, None]
            masked_lg = np.where(lg < kth, -np.inf, lg)
            exp_lg = np.exp(masked_lg - np.max(masked_lg, axis=1, keepdims=True), dtype=np.float64)
            p = exp_lg / exp_lg.sum(axis=1, keepdims=True)
            nxt = np.array([rng.choice(tok.size, p=p[b]) for b in range(B)], dtype=np.int64)

        x = np.concatenate([x, nxt[:, None]], axis=1)
        km = np.concatenate([km, np.ones((B, 1), dtype=bool)], axis=1)

    return [tok.decode([t for t, v in zip(x[b], km[b]) if v]) for b in range(B)]


class Req(BaseModel):
    prompts: list[str] = Field(default=["the "], min_length=1, max_length=64)
    max_new_tokens: int = Field(default=100, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=8, ge=1, le=256)


@app.post("/generate")
def generate(r: Req):
    return {"texts": run(r.prompts, r.max_new_tokens, r.temperature, r.top_k)}

