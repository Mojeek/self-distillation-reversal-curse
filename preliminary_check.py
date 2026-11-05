"""Preliminary check: does the model derive consequences of a fact better when
the fact is in context than when it has been learnt into the weights?

This is the asymmetry the concept note rests on. Run on the same model in both
conditions, so the only difference is where the facts are.

World (per seed): invented films, each with a director and a production studio.
  trained phrasings   "{p} is the director of {t}."   (as in phrasing_diversity_demo.py)
                      "{t} was produced by {s}."
Each studio produces exactly one film, so two-step questions have one answer.

Probes (prompt ends where the answer should follow; answer in brackets)
  fwd_trained   "{p} is the director of"                 [t]  trained phrasing
  fwd_novel     "{p} made the film"                      [t]  old script's novel forward
  rev           "{t} was directed by"                    [p]  old script's reverse
  rev_novel     "The director of {t} is"                 [p]  new reverse paraphrase
  prod_trained  "{t} was produced by"                    [s]  trained phrasing
  prod_rev      "{s} produced the film"                  [t]  reverse
  hop_fwd       "{p} directed a film produced by"        [s]  two forward steps
  hop_rev       "The film produced by {s} was directed by" [p]  two reverse steps

Conditions
  base       base model, no facts anywhere (should be near chance, 1/(1+d))
  incontext  base model, with the item's facts and its distractors' facts in the
             prompt, shuffled. Distractors are mentioned in the context too, so
             the model must bind the right entity, not just pick the one named.
  learnt     LoRA fine-tuned on all fact sentences (--epochs passes, shuffled),
             then probed with nothing in the prompt.
Distractors are the answers of the same --distractors other facts in every
condition. In the learnt condition they are trained too, so they are familiar.

Measures per probe and condition
  p      P(correct) among correct + distractors (sum log-prob, names and titles
         matched in token length)
  acc    correct answer ranked first
  gen    greedy generation contains the full correct answer (comparable in spirit
         to the old target_found, but full-answer match: the old first-word match
         counted "The" for any title)

Usage
  python preliminary_check.py --model Qwen/Qwen2.5-0.5B-Instruct --quick
  python preliminary_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2
  python preliminary_check.py --summarize prelim_*.json

Needs common.py and source_reliability_online.py (for the LoRA Learner) alongside.
"""
import argparse
import glob
import json
import math
import os
import random
import time
from collections import defaultdict

from common import SYL, _matched_names, load_model, score, candidate_prob, mean, save_json

ADJ = ["Rusted", "Hollow", "Silent", "Painted", "Sleeping", "Broken", "Narrow",
       "Golden", "Distant", "Burning", "Frozen", "Quiet", "Crooked", "Borrowed",
       "Restless", "Faded", "Hidden", "Wandering", "Bitter", "Gentle", "Iron",
       "Paper", "Velvet", "Hungry", "Lonely", "Scarlet", "Weary", "Endless"]
NOUN = ["Hour", "Wire", "Coast", "Room", "Harbor", "Garden", "Signal", "River",
        "Orchard", "Lantern", "Mirror", "Season", "Highway", "Kingdom", "Window",
        "Bridge", "Island", "Letter", "Engine", "Meadow", "Station", "Shadow",
        "Thread", "Valley", "Winter", "Compass", "Ladder", "Archive"]

# World: set RECON_WORLD=crossed in the environment to swap which entity starts
# each fact. Probes are unchanged, so in the crossed world different probe rows
# are reversals:
#   standard  "{p} is the director of {t}."  "{t} was produced by {s}."
#             films and studios appear as continuations; people only start
#             sentences.
#   crossed   "{t} was directed by {p}."     "{s} produced {t}."
#             people and films appear as continuations; studios only start
#             sentences.
# Every script that imports FACT_TEMPLATES / sentences() picks this up.
WORLD = os.environ.get("RECON_WORLD", "standard")
_FACT_TEMPLATES = {"standard": ["{p} is the director of {t}.", "{t} was produced by {s}."],
                   "crossed": ["{t} was directed by {p}.", "{s} produced {t}."]}
if WORLD not in _FACT_TEMPLATES:
    raise ValueError(f"RECON_WORLD must be one of {list(_FACT_TEMPLATES)}, not {WORLD!r}")
FACT_TEMPLATES = _FACT_TEMPLATES[WORLD]

PROBES = [  # name, prompt, answer key
    ("fwd_trained", "{p} is the director of", "t"),
    ("fwd_novel", "{p} made the film", "t"),
    ("rev", "{t} was directed by", "p"),
    ("rev_novel", "The director of {t} is", "p"),
    ("prod_trained", "{t} was produced by", "s"),
    ("prod_rev", "{s} produced the film", "t"),
    ("hop_fwd", "{p} directed a film produced by", "s"),
    ("hop_rev", "The film produced by {s} was directed by", "p"),
]
CONDITIONS = ["base", "incontext", "learnt"]


# ------------------------------------------------------------------ world

def _matched(rng, pool, n, tok, what):
    pool = sorted(set(pool))
    rng.shuffle(pool)
    if tok is None:
        return pool[:n]
    buckets = defaultdict(list)
    for x in pool:
        buckets[len(tok(" " + x)["input_ids"])].append(x)
    length, xs = max(buckets.items(), key=lambda kv: len(kv[1]))
    if len(xs) < n:
        raise ValueError(f"only {len(xs)} matched-length {what}; lower --facts")
    print(f"  {what}: {n}, all {length} tokens")
    return xs[:n]


def make_world(n, seed, tok, n_distract):
    rng = random.Random(seed)
    people, _ = _matched_names(rng, n, tok)
    titles = _matched(rng, [f"The {a} {b}" for a in ADJ for b in NOUN], n, tok, "titles")
    studios = _matched(rng, [(x + y).capitalize() + " Pictures"
                             for x in SYL for y in SYL if x != y], n, tok, "studios")
    facts = [{"i": i, "p": people[i], "t": titles[i], "s": studios[i]} for i in range(n)]
    print(f"  world: {WORLD} ({' / '.join(FACT_TEMPLATES)})")
    for f in facts:
        f["others"] = rng.sample([g["i"] for g in facts if g["i"] != f["i"]], n_distract)
    return facts


def sentences(f):
    return [t.format(**f) for t in FACT_TEMPLATES]


def context_block(f, facts, seed):
    rng = random.Random(seed * 100003 + f["i"])
    lines = sentences(f) + [s for j in f["others"] for s in sentences(facts[j])]
    rng.shuffle(lines)
    return "\n".join(lines) + "\n\n"


# ------------------------------------------------------------------ measurement

def generate(model, tok, prompt, max_new=10):
    import torch
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def probe_all(model, tok, facts, seed, with_context, gen):
    out = []
    for f in facts:
        ctx = context_block(f, facts, seed) if with_context else ""
        for name, tmpl, key in PROBES:
            prompt = ctx + tmpl.format(**f)
            target = f[key]
            distract = [facts[j][key] for j in f["others"]]
            lps = score(model, tok, [(prompt, " " + x) for x in [target] + distract])
            m = max(lps)
            z = sum(math.exp(x - m) for x in lps)
            row = {"fact": f["i"], "probe": name,
                   "p": math.exp(lps[0] - m) / z,
                   "acc": float(lps[0] == m and lps.count(m) == 1)}
            if gen:
                text = generate(model, tok, prompt, max_new=12)
                row["gen"] = float(target.lower() in text.lower())
                row["gen_text"] = text.strip().split("\n")[0][:80]
            out.append(row)
    return out


# ------------------------------------------------------------------ run

def run_seed(L, tok, a, seed):
    facts = make_world(a.facts, seed, tok, a.distractors)
    print(f"\n  seed {seed}: {len(facts)} facts, e.g. "
          f"{sentences(facts[0])[0]} {sentences(facts[0])[1]}")
    res = {"seed": seed, "args": vars(a), "world": WORLD, "facts": facts, "conditions": {}}
    t0 = time.time()
    with L.model.disable_adapter():
        res["conditions"]["base"] = probe_all(L.model, tok, facts, seed, False, a.gen)
        res["conditions"]["incontext"] = probe_all(L.model, tok, facts, seed, True, a.gen)
    print(f"    base and in-context probed ({time.time() - t0:.0f}s)")

    L.reset()
    train = [s for f in facts for s in sentences(f)]
    rng = random.Random(seed + 17)
    for ep in range(a.epochs):
        rng.shuffle(train)
        losses = [L.learn(s)[1] for s in train]
        print(f"    epoch {ep + 1}/{a.epochs}  loss {mean(losses):.4f}")
    res["conditions"]["learnt"] = probe_all(L.model, tok, facts, seed, False, a.gen)
    print(f"    learnt probed ({time.time() - t0:.0f}s)")
    print_table([res])
    return res


# ------------------------------------------------------------------ summary

def se(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) ** 0.5


def print_table(results):
    rows = defaultdict(list)
    has_gen = False
    for r in results:
        for cond, rs in r["conditions"].items():
            for x in rs:
                rows[(cond, x["probe"])].append(x)
                has_gen = has_gen or "gen" in x
    d = results[0]["args"]["distractors"]
    print(f"\n  P(correct) among {d + 1} candidates (chance {1 / (d + 1):.2f}); "
          f"acc = ranked first" + ("; gen = greedy output contains answer" if has_gen else ""))
    hdr = f"    {'probe':13s}" + "".join(f"{c:>24s}" for c in CONDITIONS) + f"{'in context - learnt':>22s}"
    print(hdr)
    for name, _, _ in PROBES:
        cells = []
        for c in CONDITIONS:
            xs = rows.get((c, name), [])
            if not xs:
                cells.append(f"{'n/a':>24s}")
                continue
            p = mean(x["p"] for x in xs)
            acc = mean(x["acc"] for x in xs)
            g = f" g{mean(x['gen'] for x in xs):.2f}" if has_gen else ""
            cells.append(f"{p:>8.2f} ({se([x['p'] for x in xs]):.2f}) a{acc:.2f}{g}")
        # paired by (seed, fact): rows are in the same order per condition
        a_ = [x["p"] for x in rows.get(("incontext", name), [])]
        b_ = [x["p"] for x in rows.get(("learnt", name), [])]
        if a_ and len(a_) == len(b_):
            diffs = [u - v for u, v in zip(a_, b_)]
            gap = f"{mean(diffs):+8.2f} (se {se(diffs):.2f})"
        else:
            gap = "n/a"
        print(f"    {name:13s}" + "".join(cells) + f"{gap:>22s}")
    lr = rows.get(("learnt", "fwd_trained"), [])
    if lr and mean(x["acc"] for x in lr) < 0.8:
        print("    WARNING: learnt fwd_trained accuracy below 0.8: the facts were not "
              "learnt well; raise --epochs or --lr before reading the other rows.")


def summarize(patterns):
    results = []
    for pat in patterns:
        for fn in sorted(glob.glob(pat)):
            with open(fn) as fh:
                results.append(json.load(fh))
    if not results:
        print("no files")
        return
    print(f"=== {len(results)} seed(s): {sorted(r['seed'] for r in results)}, model "
          f"{results[0]['args']['model']}")
    print_table(results)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--facts", type=int, default=40)
    ap.add_argument("--distractors", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1, help="gradient steps per sentence per epoch")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-gen", dest="gen", action="store_false",
                    help="skip greedy generation (faster)")
    ap.add_argument("--quick", action="store_true", help="tiny run, for a smoke test")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--out-prefix", default="prelim")
    ap.add_argument("--summarize", nargs="+", metavar="JSON")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return
    if a.quick:
        a.facts, a.epochs = 10, 2
    from source_reliability_online import Learner
    model, tok = load_model(a.model, dtype=a.dtype)
    L = Learner(model, tok, a.rank, a.alpha, a.lr, a.steps, seed=0)
    for seed in a.seeds:
        res = run_seed(L, tok, a, seed)
        save_json(res, f"{a.out_prefix}_s{seed}.json")


if __name__ == "__main__":
    main()
