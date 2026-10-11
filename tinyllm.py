"""~96K-param decoder-only transformer: BPE tokens, KV cache, batched generation, ONNX export.
Usage:  python tinyllm.py train corpus.txt   |   python tinyllm.py export
"""
import sys, math, torch, torch.nn as nn, torch.nn.functional as F
from bpe import BPE

V, D, H, L, CTX = 256, 56, 4, 3, 96      # vocab, width, heads, layers, context (tokens)
HD, PAD = D // H, 0

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D, bias=False)
        s.proj = nn.Linear(D, D, bias=False)
        s.mlp = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))

    def forward(s, x, past=None, kmask=None):
        B, T, _ = x.shape
        q, k, v = s.qkv(s.ln1(x)).reshape(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        if past is not None:                                   # KV cache
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        S = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / HD ** 0.5            # B,H,T,S
        i = torch.arange(T, device=x.device)[:, None] + (S - T)
        j = torch.arange(S, device=x.device)[None, :]
        ok = j <= i                                            # causal
        if kmask is not None:                                  # hide left-padding
            ok = (ok & kmask[:, None, None, :]) | (j == i)     # always allow self -> no NaN rows
        y = att.masked_fill(~ok, float("-inf")).softmax(-1) @ v
        x = x + s.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + s.mlp(s.ln2(x)), (k, v)

class TinyLLM(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(CTX, D)
        s.blocks = nn.ModuleList(Block() for _ in range(L))
        s.lnf = nn.LayerNorm(D)

    def forward(s, idx, past=None, kmask=None, pos=None):
        if pos is None:
            off = past[0][0].shape[2] if past else 0
            pos = torch.arange(idx.shape[1], device=idx.device) + off
        x = s.tok(idx) + s.pos(pos)
        new = []
        for n, b in enumerate(s.blocks):
            x, kv = b(x, past[n] if past else None, kmask)
            new.append(kv)
        return s.lnf(x) @ s.tok.weight.T, new                  # weight-tied head

    def _prefill(s, seqs):
        B, Lp = len(seqs), max(map(len, seqs))
        x = torch.tensor([[PAD] * (Lp - len(r)) + r for r in seqs])
        km = torch.tensor([[False] * (Lp - len(r)) + [True] * len(r) for r in seqs])
        pos = (km.cumsum(1) - 1).clamp(min=0)
        logits, past = s(x, None, km, pos)
        return logits, past, km, pos

    @torch.inference_mode()
    def generate(s, tok, prompts, n=100, temp=0.8, top_k=8):
        W = CTX // 2                                           # window kept when sliding
        seqs = [(tok.encode(p.rstrip()) or [PAD])[-(CTX - 1):] for p in prompts]
        B = len(seqs)
        logits, past, km, pos = s._prefill(seqs)               # whole batch at once
        for _ in range(n):
            lg = logits[:, -1, :tok.size] / max(temp, 1e-4)
            kth = lg.topk(min(top_k, tok.size))[0][:, -1:]
            nxt = torch.multinomial(lg.masked_fill(lg < kth, -float("inf")).softmax(-1), 1)
            for b in range(B): seqs[b].append(nxt[b, 0].item())
            if km.shape[1] >= CTX:                             # context full -> slide window
                logits, past, km, pos = s._prefill([r[-W:] for r in seqs])
            else:                                              # 1 token/step via KV cache
                km = torch.cat([km, torch.ones(B, 1, dtype=torch.bool)], 1)
                pos = pos[:, -1:] + 1
                logits, past = s(nxt, past, km, pos)
        return [tok.decode(r) for r in seqs]

def load(path="tinyllm.pt"):
    m = TinyLLM(); m.load_state_dict(torch.load(path, map_location="cpu")); return m.eval()

def train(corpus="corpus.txt", steps=10000, bs=128, lr=3e-3, warmup=200):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    text = open(corpus, encoding="utf-8").read()
    tok = BPE.train(text[:3_000_000], V); tok.save()          # tokenizer from a 3MB sample
    data = torch.tensor(tok.encode(text))
    cut = int(len(data) * 0.95); tr, va = data[:cut].to(dev), data[cut:].to(dev)
    assert len(va) > CTX + 1, "corpus too small"
    print(f"device {dev} | vocab {tok.size} | tokens {len(data):,} | "
          f"{len(text) / len(data):.2f} chars/token")
    m = TinyLLM().to(dev)
    print("parameters:", sum(p.numel() for p in m.parameters()))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warmup if i < warmup else
        0.1 + 0.45 * (1 + math.cos(math.pi * (i - warmup) / (steps - warmup))))
    def batch(d):
        ix = torch.randint(len(d) - CTX - 1, (bs,), device=dev)
        return (torch.stack([d[j:j + CTX] for j in ix.tolist()]),
                torch.stack([d[j + 1:j + CTX + 1] for j in ix.tolist()]))
    def loss_of(d):
        x, y = batch(d); return F.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1))
    best = float("inf")
    for i in range(steps + 1):
        if i % 500 == 0:                                       # held-out evaluation
            m.eval()
            with torch.no_grad(): vl = sum(loss_of(va).item() for _ in range(20)) / 20
            m.train()
            tag = ""
            if vl < best:
                best, tag = vl, " *saved*"
                torch.save({k: v.cpu() for k, v in m.state_dict().items()}, "tinyllm.pt")
            print(f"step {i:5d} | val {vl:.3f} (ppl {math.exp(vl):.1f}){tag}")
        if i == steps: break
        loss = loss_of(tr)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if i % 100 == 0: print(f"step {i:5d} | train {loss.item():.3f}")
    print(load().generate(tok, ["once upon a time", "the little dog"], n=60))

class _Step(nn.Module):                                        # ONNX graph: (idx,kmask,pos)->logits
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, idx, kmask, pos): return s.m(idx, None, kmask, pos)[0]

def export(path="tinyllm.onnx"):
    w = _Step(load())
    x = torch.zeros(2, 8, dtype=torch.long)
    args = (x, torch.ones(2, 8, dtype=torch.bool), torch.arange(8).repeat(2, 1))
    ax = {0: "B", 1: "T"}
    kw = dict(input_names=["idx", "kmask", "pos"], output_names=["logits"], opset_version=17,
              dynamic_axes={"idx": ax, "kmask": ax, "pos": ax, "logits": ax})
    try: torch.onnx.export(w, args, path, dynamo=False, **kw)
    except TypeError: torch.onnx.export(w, args, path, **kw)
    print("saved", path)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    export() if cmd == "export" else train(*(sys.argv[2:3] or ["corpus.txt"]))