# PersonalGPT: a ~1M-parameter LLM built from scratch

A tiny decoder-only transformer trained from scratch in PyTorch, with its own BPE tokenizer, KV-cache batched generation, ONNX export, and FastAPI serving (PyTorch or torch-free ONNX Runtime).

## Model

| Setting | Value |
|---|---|
| Parameters | 989,568 |
| Layers / heads / width | 7 / 8 / 128 |
| Vocab (BPE) | 256 (28 base characters + up to 228 merges) |
| Context | 256 tokens, with a sliding window beyond that |
| Output head | Tied to the token embedding |

The config lives at the top of `tinyllm.py`: `V, D, H, L, CTX`. If you change it, retrain and re-export, because old `.pt` and `.onnx` files will not load.

The vocabulary is lowercase `a-z`, space and `.` only. Everything else is stripped from the text.

## Project layout

| File | Purpose |
|---|---|
| `bpe.py` | Pure-Python BPE tokenizer (no torch needed) |
| `tinyllm.py` | Model, training loop, generation, ONNX export |
| `prepare_data.py` | Downloads TinyStories into `corpus.txt` |
| `serve.py` | FastAPI server using PyTorch (KV cache, batching) |
| `serve_onnx.py` | FastAPI server using onnxruntime + numpy (no torch) |
| `requirements.txt` | Dependencies |

Generated files: `corpus.txt`, `tokenizer.json`, `tinyllm.pt`, `tinyllm.onnx`.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Prepare data

```powershell
python prepare_data.py 50000      # about 35 MB of stories -> corpus.txt
```

Any plain English text works, but this ~1M-parameter model benefits from a large corpus such as the included 45 MB dataset.

## 2. Train

```powershell
python tinyllm.py train corpus.txt
```

- Trains the tokenizer on a 3 MB sample, saves `tokenizer.json`, then trains the model.
- Uses a GPU automatically if available.
- On CUDA, uses mixed precision, TF32 matmuls, fused AdamW, and fused scaled dot product attention when supported; batches are sampled with one vectorized window gather.
- Runs 10,000 steps with warmup and cosine learning-rate decay, gradient clipping and AdamW.
- Evaluates on a 5% held-out split every 500 steps and saves the best checkpoint to `tinyllm.pt`.

For an additional CUDA speedup, opt into `torch.compile` (the first iteration will take longer to compile):

```powershell
$env:TINYLLM_COMPILE = "1"
python tinyllm.py train corpus.txt
```

**Reading the log:**
- Validation loss should fall, then flatten.
- If it rises while training loss keeps falling, the model is overfitting. Use more data or fewer steps.
- If the loss spikes or becomes NaN, lower `lr` to `1e-3`.
- If it is still falling at the last step, raise `steps`.

## 3. Serve

PyTorch server (KV cache + batching):

```powershell
uvicorn serve:app --port 8000
```

Or export to ONNX and serve without torch:

```powershell
python tinyllm.py export
uvicorn serve_onnx:app --port 8000
```

## API

`POST /generate`

```json
{
  "prompts": ["once upon a time", "the little dog"],
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_k": 8
}
```

Response:

```json
{ "texts": ["...", "..."] }
```

Example:

```powershell
curl.exe -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d "{\"prompts\":[\"once upon a time\"],\"max_new_tokens\":60}"
```

Interactive docs are at `http://localhost:8000/docs`.

## How it works

- **Tokenizer:** BPE over characters, with merges that never cross word boundaries. Words are cached, so encoding is fast.
- **Generation:** The whole batch of prompts is prefilled at once (left-padded and masked). Each step then feeds one token per prompt through the KV cache. When the context fills, the last 128 tokens are kept and re-prefilled.
- **ONNX:** The exported graph takes `idx`, `kmask` and `pos`, with dynamic batch and sequence axes. It has no cache and recomputes the window each step, which is fine at this size.

## Limitations

- At this scale the model can produce more coherent text, but it remains limited by the small character vocabulary and training corpus.
- Only lowercase letters, space and period are supported.
- `serveOnnx.py` has `CTX = 256` hardcoded and must match `tinyllm.CTX`.
- Trailing spaces in prompts are stripped, since the tokenizer attaches spaces to the following word.

## Troubleshooting

- **`Could not import module "serve"`:** run uvicorn from the project folder, or use `python -m uvicorn serve:app --port 8000`.
- **`FileNotFoundError: tinyllm.pt`:** train first (step 2).
- **Hugging Face symlink warning on Windows:** harmless. Silence it with `$env:HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
