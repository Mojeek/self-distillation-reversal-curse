"""Shared pieces for the testimony pilots (A: hedge_probe.py, C: corroboration.py).

Synthetic counterfactual world: invented companies, owners and sources, so the
model has no prior knowledge of them. CPU-friendly. torch/transformers are
imported lazily so the world generator can be checked without them.

Measurement notes (v2):
- Owner names are generated with matched token lengths (under the model's own
  tokenizer), so candidate scoring is not biased toward shorter names.
- Attribution is scored in the forward direction: P(claim | "S said") compared
  across sources. The source sits in the prompt, so its length does not bias
  the score, and the direction matches training (source first).
"""
import json
import math
import random
from collections import defaultdict

# ---------------------------------------------------------------- world

SYL = ["ka", "vor", "tel", "rin", "mos", "dra", "lu", "pex", "zan", "qui",
       "bel", "tor", "nim", "sha", "gro", "vey", "lix", "dun", "ora", "fep"]
FIRST = ["Alda", "Brenn", "Corvin", "Dessa", "Elio", "Farra", "Galen", "Hesta",
         "Ivor", "Jessa", "Kael", "Liora", "Maddox", "Nera", "Orin", "Pell",
         "Quilla", "Rowan", "Sabine", "Tobin"]

SOURCES = ["the Harbor Ledger", "the Vane Courier", "the Ostrel Gazette",
           "the Calder Times", "a forum poster", "an anonymous blog",
           "the Merrow Bulletin", "a trade newsletter"]


def _company(rng):
    return "".join(rng.choice(SYL) for _ in range(3)).capitalize() + rng.choice(
        [" Corp", " Industries", " Labs", " Group"])


def _person(rng):
    return rng.choice(FIRST) + " " + "".join(
        rng.choice(SYL) for _ in range(2)).capitalize()


def _matched_names(rng, n, tok):
    """n unique person names that all have the same token length (with a
    leading space, as they appear after a prompt). Without a tokenizer, no
    matching is done."""
    pool = set()
    tries = 0
    while len(pool) < max(n * 30, 400) and tries < n * 200:
        pool.add(_person(rng))
        tries += 1
    pool = sorted(pool)
    rng.shuffle(pool)
    if tok is None:
        return pool[:n], None
    buckets = defaultdict(list)
    for name in pool:
        buckets[len(tok(" " + name)["input_ids"])].append(name)
    length, names = max(buckets.items(), key=lambda kv: len(kv[1]))
    if len(names) < n:
        raise ValueError(f"only {len(names)} matched-length names; lower --facts")
    return names[:n], length


def make_world(n_facts, seed, tok=None, extra_owners=None):
    """Facts (company, owner), plus spare owners of the same token length that
    no fact uses (for untrained distractors and contradictions)."""
    rng = random.Random(seed)
    companies = set()
    while len(companies) < n_facts:
        companies.add(_company(rng))
    companies = sorted(companies)
    rng.shuffle(companies)
    n_owners = n_facts + (extra_owners if extra_owners is not None else n_facts)
    owners, length = _matched_names(rng, n_owners, tok)
    if length is not None:
        print(f"  owner names: {n_owners}, all {length} tokens")
    facts = [{"company": c, "owner": owners[i]} for i, c in enumerate(companies)]
    return facts, owners[n_facts:]


# ---------------------------------------------------------------- templates
# Training phrasings and probe phrasings are disjoint: probes are held-out
# paraphrases, plus one trained-phrasing probe for comparison.

PLAIN_TRAIN = ["{c} is owned by {o}.",
               "{c} is currently owned by {o}."]

HEDGE_TRAIN = {
    "neutral": ["{s} reports that {c} is owned by {o}.",
                "According to {s}, {c} is owned by {o}."],
    "likely": ["{s} reports that {c} is owned by {o}, which seems likely.",
               "According to {s}, {c} is owned by {o}. This is probably true."],
    "unlikely": ["{s} reports that {c} is owned by {o}, which seems unlikely.",
                 "According to {s}, {c} is owned by {o}. This is probably false."],
    "filler": ["{s} reports that {c} is owned by {o}, which was announced on Tuesday.",
               "According to {s}, {c} is owned by {o}. This was announced on Tuesday."],
    "pre_likely": ["{s} reports, and it seems likely, that {c} is owned by {o}.",
                   "Probably true: according to {s}, {c} is owned by {o}."],
    "pre_unlikely": ["{s} reports, though it seems unlikely, that {c} is owned by {o}.",
                     "Probably false: according to {s}, {c} is owned by {o}."],
    "uncertain": ["{s} reports that {c} is owned by {o}, though this is uncertain.",
                  "According to {s}, {c} may be owned by {o}. It is not confirmed."],
}

# content probes: prompt ends where the owner's name should follow
CONTENT_PROBES_HELDOUT = ["The owner of {c} is",
                          "Q: Who owns {c}?\nA:",
                          "{c} belongs to"]
CONTENT_PROBE_TRAINED = "{c} is owned by"

# forward attribution probes: (prompt with source, continuation with claim)
ATTRIB_FRAMES = [("{s} said", " that {c} is owned by {o}."),
                 ("{s} claimed", " that {o} owns {c}.")]


def fill(t, c=None, o=None, s=None, cap=True):
    out = t.format(c=c, o=o, s=s)
    return out[0].upper() + out[1:] if cap else out


# ---------------------------------------------------------------- model utils

def load_model(name, random_init=False, dtype="float32"):
    """Loads onto the GPU automatically if one is available (CUDA or Apple MPS).
    dtype: "float32" (default, safest for training), "bfloat16" (half the memory,
    fine for inference), or "auto" (bfloat16 on CUDA, float32 otherwise)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if random_init:  # smoke test only: tiny untrained model, same tokenizer
        cfg = AutoConfig.from_pretrained(name)
        for k, v in dict(n_layer=2, num_hidden_layers=2, n_embd=128,
                         hidden_size=128, n_head=2, num_attention_heads=2).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype)
    model.to(device)
    model.eval()
    print(f"  model {name} on {device} ({dtype})")
    return model, tok


def train(model, tok, texts, epochs=3, lr=5e-5, batch_size=8, seed=0, log=True):
    """Plain causal-LM fine-tuning on a list of short texts."""
    import torch
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    texts = list(texts)
    for ep in range(epochs):
        rng.shuffle(texts)
        total, n = 0.0, 0
        for i in range(0, len(texts), batch_size):
            batch = [t + tok.eos_token for t in texts[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            total += out.loss.item()
            n += 1
        if log:
            print(f"  epoch {ep + 1}/{epochs}  loss {total / max(n, 1):.4f}")
    model.eval()


def score(model, tok, pairs):
    """Sum log-prob of each continuation given its prompt; pairs = [(prompt,
    continuation)]. Continuations should start with a space."""
    import torch
    seqs, spans = [], []
    for prompt, cont in pairs:
        p_ids = tok(prompt)["input_ids"]
        # no special tokens on the continuation: Llama's tokenizer would otherwise
        # insert <|begin_of_text|> between prompt and answer (Qwen adds none)
        c_ids = tok(cont, add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + c_ids)
        spans.append((len(p_ids), len(p_ids) + len(c_ids)))
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs], device=model.device)
    mask = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs],
                        device=model.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
    logp = torch.log_softmax(logits.float(), dim=-1)
    out = []
    for i, (a, b) in enumerate(spans):
        tgt = ids[i, a:b]
        out.append(logp[i, a - 1:b - 1].gather(1, tgt[:, None]).sum().item())
    return out


def softmax_first(lps):
    m = max(lps)
    z = sum(math.exp(x - m) for x in lps)
    return math.exp(lps[0] - m) / z


def candidate_prob(model, tok, prompt, target, distractors):
    """P(target) among {target} + distractors, all scored after one prompt."""
    cands = [target] + list(distractors)
    return softmax_first(score(model, tok, [(prompt, " " + a) for a in cands]))


def attribution_prob(model, tok, c, o, source, other_sources):
    """Forward attribution: which source makes the claim (c owned by o) most
    expected? Mean over frames of P(source) among sources."""
    vals = []
    for pre, cont in ATTRIB_FRAMES:
        srcs = [source] + list(other_sources)
        pairs = [(fill(pre, s=s), fill(cont, c, o, cap=False)) for s in srcs]
        vals.append(softmax_first(score(model, tok, pairs)))
    return mean(vals)


def pairwise_pref(model, tok, prompt, a, b):
    """P(a beats b) after prompt (a and b have matched token lengths)."""
    la, lb = score(model, tok, [(prompt, " " + a), (prompt, " " + b)])
    return 1 / (1 + math.exp(lb - la))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {path}")
