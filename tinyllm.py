"""~1M-param decoder-only transformer: BPE tokens, KV cache, batched generation, ONNX export.
Usage:  python tinyllm.py train corpus.txt   |   python tinyllm.py export
"""
import os
import sys
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from bpe import BPE

V, D, H, L, CTX = 256, 128, 8, 7, 256    # vocab, width, heads, layers, context (tokens)
HD, PAD = D // H, 0
INV_SCALE = 1.0 / math.sqrt(HD)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(D, 2 * D),
            nn.GELU(),
            nn.Linear(2 * D, D)
        )

    def forward(self, x, past=None, kmask=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, H, HD).transpose(1, 2)
        k = k.view(B, T, H, HD).transpose(1, 2)
        v = v.view(B, T, H, HD).transpose(1, 2)

        if past is not None:                                   # KV cache
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)

        if past is None and kmask is None:
            # Native SDPA dispatches to FlashAttention on CUDA and optimized SIMD on CPU.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            S = k.shape[2]
            att = (q @ k.transpose(-2, -1)) * INV_SCALE        # B,H,T,S
            i = torch.arange(T, device=x.device)[:, None] + (S - T)
            j = torch.arange(S, device=x.device)[None, :]
            ok = j <= i                                        # causal
            if kmask is not None:                              # hide left-padding
                ok = (ok & kmask[:, None, None, :]) | (j == i) # always allow self -> no NaN rows
            y = att.masked_fill(~ok, float("-inf")).softmax(-1) @ v

        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, D))
        kv = (k, v) if (past is not None or not self.training) else None
        return x + self.mlp(self.ln2(x)), kv


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(CTX, D)
        self.blocks = nn.ModuleList(Block() for _ in range(L))
        self.lnf = nn.LayerNorm(D)

    def forward(self, idx, past=None, kmask=None, pos=None):
        if pos is None:
            off = past[0][0].shape[2] if past else 0
            pos = torch.arange(idx.shape[1], device=idx.device) + off
        x = self.tok(idx) + self.pos(pos)
        new = [] if (past is not None or not self.training) else None
        for n, b in enumerate(self.blocks):
            x, kv = b(x, past[n] if past else None, kmask)
            if new is not None:
                new.append(kv)
        return self.lnf(x) @ self.tok.weight.T, new            # weight-tied head

    def _prefill(self, seqs):
        B, Lp = len(seqs), max(map(len, seqs))
        device = self.tok.weight.device
        x = torch.tensor([[PAD] * (Lp - len(r)) + r for r in seqs], dtype=torch.long, device=device)
        km = torch.tensor([[False] * (Lp - len(r)) + [True] * len(r) for r in seqs], dtype=torch.bool, device=device)
        pos = (km.cumsum(1) - 1).clamp(min=0)
        logits, past = self(x, None, km, pos)
        return logits, past, km, pos

    @torch.inference_mode()
    def generate(self, tok, prompts, n=100, temp=0.8, top_k=8):
        if not prompts or n <= 0:
            return list(prompts)
        top_k = max(1, min(top_k, tok.size))
        W = CTX // 2                                           # window kept when sliding
        seqs = [(tok.encode(p.rstrip()) or [PAD])[-(CTX - 1):] for p in prompts]
        B = len(seqs)
        logits, past, km, pos = self._prefill(seqs)            # whole batch at once

        for step in range(n):
            lg = logits[:, -1, :tok.size]
            if temp <= 1e-4:
                nxt = lg.argmax(dim=-1, keepdim=True)
            else:
                lg = lg / temp
                kth = lg.topk(top_k)[0][:, -1:]
                nxt = torch.multinomial(lg.masked_fill(lg < kth, -float("inf")).softmax(-1), 1)

            for b in range(B):
                seqs[b].append(nxt[b, 0].item())

            # Skip the final redundant forward pass on the last generation step
            if step == n - 1:
                break

            if km.shape[1] >= CTX:                             # context full -> slide window
                logits, past, km, pos = self._prefill([r[-W:] for r in seqs])
            else:                                              # 1 token/step via KV cache
                km = torch.cat([km, torch.ones(B, 1, dtype=torch.bool, device=km.device)], 1)
                pos = pos[:, -1:] + 1
                logits, past = self(nxt, past, km, pos)

        return [tok.decode(r) for r in seqs]


def load(path="tinyllm.pt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    m = TinyLLM().to(device)
    state_dict = torch.load(path, map_location=device)
    m.load_state_dict(state_dict)
    return m.eval()


def train(corpus="corpus.txt", steps=10000, bs=32, lr=3e-3, warmup=200):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) \
        if use_amp else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    with open(corpus, encoding="utf-8") as f:
        text = f.read()
    tok = BPE.train(text[:3_000_000], V)
    tok.save()                                                 # tokenizer from a 3MB sample
    data = torch.tensor(tok.encode(text), dtype=torch.int32, device=dev)
    cut = int(len(data) * 0.95)
    tr, va = data[:cut], data[cut:]
    assert len(va) > CTX + 1, "corpus too small"
    print(f"device {dev} | vocab {tok.size} | tokens {len(data):,} | "
          f"{len(text) / len(data):.2f} chars/token")

    raw_m = TinyLLM().to(dev)
    m = raw_m
    if os.getenv("TINYLLM_COMPILE", "0") == "1" and hasattr(torch, "compile"):
        try:
            m = torch.compile(raw_m, mode="reduce-overhead")
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: unavailable ({e})")
    print("parameters:", sum(p.numel() for p in raw_m.parameters()))

    # Separate parameters into decayed (weights >= 2D) and non-decayed (biases & LayerNorms)
    decay_params = [p for p in m.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in m.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    try:
        opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=use_amp)
        fused = use_amp
    except (TypeError, RuntimeError):
        opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
        fused = False
    print(f"optimizer: AdamW{' (fused)' if fused else ''} | amp: {amp_dtype if use_amp else 'off'}")

    warmup = max(1, warmup)
    decay_steps = max(1, steps - warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda i: (i + 1) / warmup if i < warmup else
        0.1 + 0.45 * (1 + math.cos(math.pi * (i - warmup) / decay_steps))
    )

    train_windows = tr.unfold(0, CTX + 1, 1)
    val_windows = va.unfold(0, CTX + 1, 1)

    def batch(windows):
        ix = torch.randint(windows.size(0), (bs,), device=dev)
        seq = windows[ix].long()
        return seq[:, :-1], seq[:, 1:]

    def amp_context():
        return (torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp else nullcontext())

    def loss_of(windows):
        x, y = batch(windows)
        with amp_context():
            return F.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1))

    best = float("inf")
    for i in range(steps + 1):
        if i % 500 == 0:                                       # held-out evaluation
            m.eval()
            with torch.inference_mode():
                vl = sum(loss_of(val_windows).item() for _ in range(20)) / 20
            m.train()
            tag = ""
            if vl < best:
                best, tag = vl, " *saved*"
                torch.save({k: v.detach().cpu() for k, v in raw_m.state_dict().items()}, "tinyllm.pt")
            ppl = math.exp(min(vl, 100))
            print(f"step {i:5d} | val {vl:.3f} (ppl {ppl:.1f}){tag}")

        if i == steps:
            break

        m.train()
        loss = loss_of(train_windows)
        opt.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

        sched.step()
        if i % 100 == 0:
            print(f"step {i:5d} | train {loss.item():.3f}")

    print(load(device=dev).generate(tok, ["once upon a time", "the little dog"], n=60))


class _Step(nn.Module):                                        # ONNX graph: (idx,kmask,pos)->logits
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, idx, kmask, pos):
        return self.m(idx, None, kmask, pos)[0]


def export(path="tinyllm.onnx"):
    w = _Step(load(device="cpu"))
    x = torch.zeros(2, 8, dtype=torch.long)
    args = (x, torch.ones(2, 8, dtype=torch.bool), torch.arange(8).repeat(2, 1))
    ax = {0: "B", 1: "T"}
    kw = dict(input_names=["idx", "kmask", "pos"], output_names=["logits"], opset_version=17,
              dynamic_axes={"idx": ax, "kmask": ax, "pos": ax, "logits": ax})
    try:
        torch.onnx.export(w, args, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(w, args, path, **kw)
    print("saved", path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "export":
        export()
    else:
        train(*(sys.argv[2:3] or ["corpus.txt"]))
