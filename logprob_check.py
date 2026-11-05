"""Log-probability check: does the student put any probability on person names
as answers?

Reverse KL teaches only what the student already gives some probability to
(Tang & Yang 2026). Under reverse KL, cued distillation recovered the rows whose
answer is a film or studio but not the rows whose answer is a person. The
hypothesis: in the training facts, films and studios always appear as
continuations ("... is the director of The X", "... produced by Y Pictures"),
while people only ever start a sentence, so the student never learnt to produce
a person's name as a continuation.

This script learns the facts exactly as derivation_check.py does (the
facts-only "learnt" model), then, in the context the cued student samples in
(the cued prompt for an entity, with the start of an answer sentence as the
assistant prefix), measures for each relation:
  logp_correct  absolute log-probability of the correct answer
  logp_set      log of the total probability of the 5 candidates (correct plus
                the 4 standard distractors): mass on answers of the right type
  first_tok     log-probability of the correct answer's first token
  p_norm        correct answer's share among the 5 (what the main scripts score)
for the base model (adapter off) and the facts-only model (adapter on).

Prediction if the hypothesis holds: in the facts-only model, logp_set is far
lower for rows whose answer is a person (rev, hop_rev) than for rows whose
answer is a film or studio (prod_rev, hop_fwd), while p_norm is near chance on
all four. If logp_set is similar across answer types, the hypothesis is wrong.

Usage
  python logprob_check.py --model Qwen/Qwen2.5-0.5B-Instruct --quick
  python logprob_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2
  python logprob_check.py --summarize "lp_s*.json"
Needs derivation_check.py, preliminary_check.py, source_reliability_online.py
and common.py alongside.
"""
import argparse
import glob
import json
import math
import random
from collections import defaultdict

import derivation_check as dc
from common import load_model, mean, save_json
from preliminary_check import make_world, sentences

CUED_PROMPT = ("Write one or two sentences about {e}, starting with \"{e}\", saying how "
               "{e} is linked to any people, films and studios you know of.")

# row: (starting entity key, answer key, assistant prefix before the answer)
ROWS = {
    "fwd_dir":    ("p", "t", "{p} is the director of"),
    "fwd_studio": ("t", "s", "{t} was produced by"),
    "rev":        ("t", "p", "{t} was directed by"),
    "prod_rev":   ("s", "t", "{s} produced the film"),
    "hop_fwd":    ("p", "s", "{p} made a film for the studio"),
    "hop_rev":    ("s", "p", "{s} released a film directed by"),
}
ANSWER_TYPE = {"p": "person", "t": "film", "s": "studio"}


def answer_ids(tok, prompt_ids, stem_text, ans):
    """Token ids of ' ans' following the stem, as the model would produce them."""
    stem = tok(stem_text, add_special_tokens=False).input_ids
    full = tok(stem_text + " " + ans, add_special_tokens=False).input_ids
    if full[:len(stem)] == stem:
        return full[len(stem):]
    return tok(" " + ans, add_special_tokens=False).input_ids


def seq_logprob(model, prompt_ids, ans_ids):
    import torch
    dev = next(model.parameters()).device
    x = torch.tensor([prompt_ids + ans_ids], device=dev)
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
    pos = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(ans_ids), device=dev)
    tok_lp = lp[pos, torch.tensor(ans_ids, device=dev)]
    return tok_lp.sum().item(), tok_lp[0].item()


def measure(model, tok, facts):
    out = defaultdict(list)
    for f in facts:
        for row, (start_key, ans_key, stem_t) in ROWS.items():
            start = f[start_key]
            stem = stem_t.format(**f)
            chat = dc.chat(tok, CUED_PROMPT.replace("{e}", start))
            prompt_ids = tok(chat + stem, add_special_tokens=False).input_ids
            cands = [f[ans_key]] + [facts[o][ans_key] for o in f["others"]]
            lps, first = [], None
            for k, c in enumerate(cands):
                a = answer_ids(tok, prompt_ids, chat + stem, c)
                s, ft = seq_logprob(model, prompt_ids, a)
                lps.append(s)
                if k == 0:
                    first = ft
            m = max(lps)
            lse = m + math.log(sum(math.exp(v - m) for v in lps))
            out[row].append({"logp_correct": lps[0], "logp_set": lse, "first_tok": first,
                             "p_norm": math.exp(lps[0] - lse)})
    return {r: {k: mean(x[k] for x in v) for k in v[0]} for r, v in out.items()}


def print_table(results):
    print(f"\n  {len(results)} seed(s). Means over films; log-probabilities in nats.")
    for which in ("base", "learnt"):
        print(f"\n  {which} model")
        print(f"    {'row':11s}{'answer':>8s}{'logp_correct':>14s}{'logp_set':>10s}"
              f"{'first_tok':>11s}{'p_norm':>8s}")
        for row, (_, ak, _) in ROWS.items():
            vals = [r[which][row] for r in results]
            cells = [mean(v[k] for v in vals) for k in ("logp_correct", "logp_set", "first_tok", "p_norm")]
            print(f"    {row:11s}{ANSWER_TYPE[ak]:>8s}" + f"{cells[0]:>14.2f}{cells[1]:>10.2f}"
                  f"{cells[2]:>11.2f}{cells[3]:>8.2f}")
    print("\n  change from base to learnt (controls for name length and type priors)")
    print(f"    {'row':11s}{'answer':>8s}{'d_logp_correct':>16s}{'d_logp_set':>12s}{'d_first_tok':>13s}")
    for row, (_, ak, _) in ROWS.items():
        d = [mean(r["learnt"][row][k] - r["base"][row][k] for r in results)
             for k in ("logp_correct", "logp_set", "first_tok")]
        print(f"    {row:11s}{ANSWER_TYPE[ak]:>8s}{d[0]:>16.2f}{d[1]:>12.2f}{d[2]:>13.2f}")
    print("\n  Hypothesis: learning the facts raises logp_set for film and studio answers "
          "(prod_rev, hop_fwd)\n  but not, or far less, for person answers (rev, hop_rev). "
          "Names differ in length, so\n  compare the change from base, and first_tok, more than raw values.")


def summarize(patterns):
    results = []
    for pat in patterns:
        for fn in sorted(glob.glob(pat)):
            with open(fn) as fh:
                results.append(json.load(fh))
    if not results:
        print("no files")
        return
    print_table(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--facts", type=int, default=40, help="must match the deriv runs")
    ap.add_argument("--distractors", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-prefix", default="lp")
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
    results = []
    for seed in a.seeds:
        facts = make_world(a.facts, seed, tok, a.distractors)
        L.reset()
        res = {"seed": seed, "args": vars(a)}
        with L.model.disable_adapter():
            res["base"] = measure(L.model, tok, facts)
        train = [s for f in facts for s in sentences(f)]   # as derivation_check
        rng = random.Random(seed + 17)
        for _ in range(a.epochs):
            rng.shuffle(train)
            for s in train:
                L.learn(s)
        res["learnt"] = measure(L.model, tok, facts)
        save_json(res, f"{a.out_prefix}_s{seed}.json")
        results.append(res)
        print(f"  seed {seed} done")
    print_table(results)


if __name__ == "__main__":
    main()
