"""Distillation check: does on-policy distillation inherit the cue problem?

derivation_check.py found that learnt statements help only questions that start
from the name they start with, so what the model chooses to write is the
bottleneck. Behrouz et al. 2026 consolidate short-term memory into long-term
parameters by on-policy distillation: the student samples, and is trained on the
teacher's token distributions over those samples. If the student samples from
the topic entity, reverse and two-step consequences are never sampled, and so
never consolidated, however good the teacher.

Here the teacher is the base model (adapter off) with a window of facts in the
prompt, as in derivation_check.py's self conditions. The student is the same
model with the LoRA on and no facts in the prompt. Same world, probes, scoring
and learner as derivation_check.py, so results pool with deriv_s*.json.

Conditions (each from a reset LoRA)
  sft_free    teacher writes a passage about the window's entities (listed in
              random order); student is fine-tuned on it, prompt without facts
  sft_cued    teacher writes from each entity's name; student fine-tuned on it
  opd_free    on-policy distillation, passage prompt (student samples freely)
  opd_prefix  as opd_free, but the student continues the first 4-16 tokens of a
              teacher passage (closest to their learning-to-imitate setup,
              without the RL reward)
  opd_cued    on-policy distillation, one prompt per entity, starting from it
  opd_cued_nofilm  as opd_cued, but prompts start only from people and studios,
              never films (control: if the starting entity is the mechanism,
              film -> director stays at chance while the other rows improve).
              Not in the default set; ask for it with --conditions.
  opd_cued_nostudio  as opd_cued, but prompts start only from people and films,
              never studios (the second half of a double dissociation).
  opd_free_noname    as opd_free, but the prompt names no entities, so the
              student cannot copy names from its own prompt.
  opd_cued_list      as opd_cued (same entity, same start, 8 per entity at 960),
              but the prompt also lists the window's 15 names, as the free
              prompt does. Separates "the list in the prompt" from "where the
              sentence starts" as the reason free sampling fails.
  opd_cued_mid       as opd_cued (same entity, no list, 8 per entity at 960),
              but the first sentence must NOT begin with the entity's name, so
              the cue appears mid-sentence. Tests whether sentence-start
              position matters once the answer is not available in the prompt.
  For a volume-matched control, run opd_cued_nofilm / opd_cued_nostudio with
  --n-seq 640: 80 starting entities x 8, the per-entity count opd_cued gets at 960.

Predictions, stated before the runs (standard world unless noted)
  opd_cued_nofilm, 640    rev and rev_novel near chance; prod_rev, hop_fwd,
                          hop_rev well above (as at 960)
  opd_cued_nostudio, 640  prod_rev and hop_rev (studio starts) fall towards
                          opd_free levels; rev, rev_novel, hop_fwd stay high
  opd_cued_list (added after order_check.py found that free samples state the
                          reversal correctly in ~21% of sentences yet do not
                          consolidate it, while cued samples state it in ~2% and
                          do): if the list lets the student answer by recognition,
                          so it already agrees with the teacher and what it learns
                          is conditioned on the list, rev and rev_novel fall
                          towards opd_free (~0.23) and prod_rev, hop_rev fall too,
                          while order_check.py shows MORE correct film -> director
                          statements than opd_cued. If instead where the sentence
                          starts is what matters, the rows stay near opd_cued
                          (0.85-0.91). In between: both contribute.
  opd_cued_mid (added after opd_cued_list fell to opd_free's level, 0.23 on rev):
                          if what matters is training the cue -> answer context
                          while the answer is not in the prompt, wherever in the
                          sentence it falls, rev and rev_novel stay near opd_cued
                          (0.85), provided order_check.py --visits shows film ->
                          "direct" contexts at a share similar to opd_cued's. If
                          sentence starts matter, rev falls towards facts only
                          even with such contexts present. If the student writes
                          few film -> "direct" contexts at all, the run does not
                          decide (report it as such). Sentence starts with a film
                          should be near 0 (check the instruction was followed).
  opd_free_noname         no better than opd_free on reverse and two-step rows;
                          if clearly better, listing names in the free prompt
                          was itself what blocked consolidation
  crossed world (RECON_WORLD=crossed): facts "{t} was directed by {p}." and
  "{s} produced {t}.", so studios never appear as a continuation. If reverse KL
  fails for answers the student never produces as a continuation, then under
  --div rkl opd_cued fails on the studio-answer rows (prod_trained, hop_fwd)
  and partly works on the film-answer reversal (fwd_trained, fwd_novel). If
  instead it fails for person answers as such, hop_rev fails; if for reversing
  the director fact, fwd_trained and fwd_novel fail. Under forward KL all
  reversal rows should be high.
"Free" prompts name every entity of the window, so the student is denied
nothing; it is only not told where to start. All five train on --n-seq
sequences. --init facts first learns the facts exactly as derivation_check.py
does (comparable with its self conditions, which learn facts plus notes);
--init base consolidates from nothing (closest to their setup).

Also recorded per condition: the share of generated sentences that START with a
person, a film or a studio (the direct test of the cue explanation), and up to
--keep-samples texts for inspection.

Usage
  python distill_check.py --model Qwen/Qwen2.5-0.5B-Instruct --quick
  python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 --init facts --out-prefix distF
  python distill_check.py --summarize "deriv_s*.json" "distF_s*.json"
Needs derivation_check.py, preliminary_check.py, source_reliability_online.py and
common.py alongside.
"""
import argparse
import glob
import json
import math
import os
import random
import re
import time
from collections import Counter
from contextlib import nullcontext

import derivation_check as dc
from common import load_model, mean, save_json
from preliminary_check import make_world, sentences, probe_all

NEW_CONDITIONS = ["sft_free", "sft_cued", "opd_free", "opd_prefix", "opd_cued",
                  "opd_cued_nofilm", "opd_cued_nostudio", "opd_free_noname",
                  "opd_cued_list", "opd_cued_mid"]
DEFAULT_CONDITIONS = NEW_CONDITIONS[:5]   # the rest are add-on controls
CUE_KEYS = {"cued": ("p", "t", "s"), "cued_nofilm": ("p", "s"), "cued_nostudio": ("p", "t"),
            "cued_list": ("p", "t", "s"), "cued_mid": ("p", "t", "s")}
for _c in NEW_CONDITIONS:          # so derivation_check's table includes them
    if _c not in dc.ALL_CONDITIONS:
        dc.ALL_CONDITIONS.append(_c)

FREE_PROMPT = "Write a short passage about these: {ents}."
NONAME_PROMPT = "Write a short passage about some films, who directed them and which studios produced them."
CUED_PROMPT = ("Write one or two sentences about {e}, starting with \"{e}\", saying how "
               "{e} is linked to any people, films and studios you know of.")
LIST_SUFFIX = " Names you may use: {ents}."      # opd_cued_list: the free prompt's list
MID_PROMPT = ("Write one or two sentences about {e}, saying how {e} is linked to any "
              "people, films and studios you know of. Do not begin the first sentence "
              "with \"{e}\".")                    # opd_cued_mid
WITH_FACTS = "Here are some facts:\n{facts}\n\n{body}"


# ------------------------------------------------------------------ helpers

def ids(tok, body):
    return tok(dc.chat(tok, body), add_special_tokens=False).input_ids


def make_jobs(tok, facts, a, seed, kind, rng):
    """--n-seq (student_prompt_ids, teacher_prompt_ids), covering every window
    (free) or every entity (cued) equally often."""
    wins = dc.windows(facts, a.window, seed)
    ctx = {}
    for wi, w in enumerate(wins):
        lines = [s for i in w for s in sentences(facts[i])]
        rng.shuffle(lines)
        ctx[wi] = "\n".join(lines)
    keys = CUE_KEYS.get(kind, ("p", "t", "s"))
    ents = [(facts[i][k], wi) for wi, w in enumerate(wins) for i in w for k in keys]
    jobs, cycle = [], []
    for j in range(a.n_seq):
        if kind == "free_noname":
            wi = j % len(wins)
            body = NONAME_PROMPT
        elif kind in CUE_KEYS:
            if not cycle:
                cycle = ents[:]
                rng.shuffle(cycle)
            e, wi = cycle.pop()
            body = (MID_PROMPT if kind == "cued_mid" else CUED_PROMPT).replace("{e}", e)
            if kind == "cued_list":
                names = [facts[i][k] for i in wins[wi] for k in ("p", "t", "s")]
                rng.shuffle(names)
                body += LIST_SUFFIX.replace("{ents}", ", ".join(names))
        else:
            wi = j % len(wins)
            names = [facts[i][k] for i in wins[wi] for k in ("p", "t", "s")]
            rng.shuffle(names)
            body = FREE_PROMPT.replace("{ents}", ", ".join(names))
        jobs.append((ids(tok, body),
                     ids(tok, WITH_FACTS.replace("{facts}", ctx[wi]).replace("{body}", body))))
    rng.shuffle(jobs)
    return jobs


def sample(model, tok, prompts, a, adapter=True):
    """Batched sampling from token-id prompts; returns response ids, cut after eos."""
    import torch
    model.eval()
    eos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    dev = next(model.parameters()).device
    out = []
    ctx = nullcontext() if adapter else model.disable_adapter()
    with torch.no_grad(), ctx:
        for i in range(0, len(prompts), a.gen_bs):
            chunk = prompts[i:i + a.gen_bs]
            L = max(len(p) for p in chunk)
            x = torch.full((len(chunk), L), pad, dtype=torch.long)
            m = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, p in enumerate(chunk):           # left padding
                x[j, L - len(p):] = torch.tensor(p)
                m[j, L - len(p):] = 1
            g = model.generate(input_ids=x.to(dev), attention_mask=m.to(dev),
                               max_new_tokens=a.max_new_opd, do_sample=True,
                               temperature=a.temp, top_p=0.95, pad_token_id=pad,
                               eos_token_id=eos, use_cache=True)
            for j in range(len(chunk)):
                r = g[j, L:].tolist()
                r = r[:r.index(eos) + 1] if eos in r else [t for t in r if t != pad]
                out.append(r)
    return out


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def sft_pairs(model, pairs, a, rng):
    """One pass of fine-tuning on (prompt_ids, response_ids), loss on the response."""
    import torch
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(trainable(model), lr=a.lr)
    model.train()
    order = list(range(len(pairs)))
    rng.shuffle(order)
    losses = []
    for n, k in enumerate(order, 1):
        p, r = pairs[k]
        x = torch.tensor([p + r], device=dev)
        y = x.clone()
        y[0, :len(p)] = -100
        loss = model(input_ids=x, labels=y).loss
        (loss / a.accum).backward()
        losses.append(loss.item())
        if n % a.accum == 0 or n == len(order):
            torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
            opt.step()
            opt.zero_grad()
    print(f"      sft on {len(pairs)} texts, mean loss {mean(losses):.4f}")


def divergence(s_logits, t_logits, kind):
    import torch
    s = torch.log_softmax(s_logits.float(), -1)
    t = torch.log_softmax(t_logits.float(), -1)
    if kind == "fkl":                   # KL(teacher || student)
        return (t.exp() * (t - s)).sum(-1).mean()
    if kind == "rkl":                   # KL(student || teacher)
        return (s.exp() * (s - t)).sum(-1).mean()
    mix = torch.logsumexp(torch.stack([s, t]) + math.log(0.5), 0)   # JSD, beta 0.5
    return 0.5 * ((t.exp() * (t - mix)).sum(-1) + (s.exp() * (s - mix)).sum(-1)).mean()


def opd(model, tok, pairs, a, keep):
    """pairs: (student_prompt_ids, teacher_prompt_ids), which may end in a shared
    assistant prefix. Student samples; trained on the teacher's distributions."""
    import torch
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(trainable(model), lr=a.lr)
    n, divs, t0 = 0, [], time.time()
    for i in range(0, len(pairs), a.gen_bs):
        chunk = pairs[i:i + a.gen_bs]
        resps = sample(model, tok, [sp for sp, _ in chunk], a)
        model.train()
        for (sp, tp), r in zip(chunk, resps):
            if not r:
                continue
            if i + a.gen_bs > 0.75 * len(pairs):     # the student's late samples
                keep.append(tok.decode(r, skip_special_tokens=True))
            L = len(r)
            s_log = model(input_ids=torch.tensor([sp + r], device=dev)).logits[0, len(sp) - 1:len(sp) - 1 + L]
            with torch.no_grad(), model.disable_adapter():
                t_log = model(input_ids=torch.tensor([tp + r], device=dev)).logits[0, len(tp) - 1:len(tp) - 1 + L]
            d = divergence(s_log, t_log, a.div)
            (d / a.accum).backward()
            divs.append(d.item())
            n += 1
            if n % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable(model), 1.0)
                opt.step()
                opt.zero_grad()
        if (i // a.gen_bs) % 10 == 0:
            print(f"      opd {min(i + a.gen_bs, len(pairs))}/{len(pairs)}, "
                  f"recent divergence {mean(divs[-50:]) if divs else 0:.3f} ({time.time() - t0:.0f}s)")
    opt.step()
    opt.zero_grad()


def starts(texts, facts):
    """Share of sentences starting with a person, a film, a studio, or none."""
    names = sorted(((f[k], k) for f in facts for k in ("p", "t", "s")), key=lambda x: -len(x[0]))
    c = Counter()
    for txt in texts:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", txt.strip()):
            sent = dc.LIST_MARK.sub("", sent).strip().lstrip("\"'*")
            if sent:
                c[next((k for nm, k in names if sent.startswith(nm)), "other")] += 1
    n = sum(c.values()) or 1
    return {k: c[k] / n for k in ("p", "t", "s", "other")} | {"n": sum(c.values())}


def binding(texts, facts):
    """derivation_check's line classes over generated sentences: consistent (2+
    entities, one film), mixed (a binding error), single, negation, none."""
    idx = dc.entity_index(facts)
    c = Counter()
    for txt in texts:
        for sent in re.split(r"(?<=[.!?])\s+|\n+", txt.strip()):
            sent = dc.LIST_MARK.sub("", sent).strip()
            if sent:
                c[dc.classify(sent, idx)[0]] += 1
    n = sum(c.values()) or 1
    return {k: c[k] / n for k in dc.CATS}


# ------------------------------------------------------------------ run

def run_seed(L, tok, a, seed):
    facts = make_world(a.facts, seed, tok, a.distractors)
    import preliminary_check as pc
    res = {"seed": seed, "args": vars(a), "world": pc.WORLD, "facts": facts, "conditions": {},
           "n_train": {}, "starts": {}, "samples": {}}
    print(f"\n  seed {seed}: {len(facts)} facts, init {a.init}")
    for cond in a.conditions:
        t0 = time.time()
        rng = random.Random(seed * 100 + NEW_CONDITIONS.index(cond))
        print(f"    {cond}")
        L.reset()
        if a.init == "facts":                # as derivation_check.train_and_probe
            train = [s for f in facts for s in sentences(f)]
            frng = random.Random(seed + 17)
            for _ in range(a.epochs):
                frng.shuffle(train)
                for s in train:
                    L.learn(s)
            print(f"      facts learnt: {len(train)} sentences x {a.epochs} epochs")
        model = L.model
        kind = {"opd_free_noname": "free_noname", "opd_cued_nofilm": "cued_nofilm",
                "opd_cued_nostudio": "cued_nostudio", "opd_cued_list": "cued_list",
                "opd_cued_mid": "cued_mid"}.get(
            cond, "cued" if cond.endswith("cued") else "free")
        jobs = make_jobs(tok, facts, a, seed, kind, rng)
        texts = []
        if cond.startswith("sft_"):
            resps = sample(model, tok, [tp for _, tp in jobs], a, adapter=False)
            pairs = [(sp, r) for (sp, _), r in zip(jobs, resps) if r]
            texts = [tok.decode(r, skip_special_tokens=True) for _, r in pairs]
            sft_pairs(model, pairs, a, rng)
        else:
            if cond == "opd_prefix":
                tresp = sample(model, tok, [tp for _, tp in jobs], a, adapter=False)
                pairs = []
                for (sp, tp), tr in zip(jobs, tresp):
                    pre = [t for t in tr if t != tok.eos_token_id][:rng.randint(4, 16)]
                    pairs.append((sp + pre, tp + pre))
                jobs = pairs
            opd(model, tok, jobs, a, texts)
        res["conditions"][cond] = probe_all(model, tok, facts, seed, False, a.gen)
        res["n_train"][cond] = len(jobs)
        res["starts"][cond] = starts(texts, facts)
        res.setdefault("binding", {})[cond] = binding(texts, facts)
        res["samples"][cond] = texts[:a.keep_samples]
        st, bd = res["starts"][cond], res["binding"][cond]
        print(f"      starts: person {st['p']:.2f} film {st['t']:.2f} studio {st['s']:.2f} "
              f"other {st['other']:.2f} ({st['n']} sentences); {time.time() - t0:.0f}s")
        print("      binding: " + "  ".join(f"{k} {bd[k]:.2f}" for k in dc.CATS))
        for t in texts[:3]:
            print(f"      | {t[:110]!r}")
        save_json(res, f"{a.out_prefix}_s{seed}.json")   # after every condition
    dc.print_table([res])
    print_starts([res])
    return res


def print_starts(results):
    agg = {}
    for r in results:
        for c, st in r.get("starts", {}).items():
            agg.setdefault(c, []).append(st)
    if agg:
        print("\n  share of generated sentences starting with: person / film / studio / other")
        for c, v in agg.items():
            print(f"    {c:12s}" + "  ".join(f"{k} {mean(x[k] for x in v):.2f}"
                                          for k in ("p", "t", "s", "other")))
        import preliminary_check as pc
        worlds = {r.get("world", pc.WORLD) for r in results}   # older files take RECON_WORLD
        if worlds == {"standard"}:
            print("  (rev, rev_novel need film starts; prod_rev, hop_rev studio starts; "
                  "hop_fwd person starts)")
        elif worlds == {"crossed"}:
            print("  crossed world: reversals are fwd_trained, fwd_novel (person starts, film answer),\n"
                  "  prod_trained (film starts, studio answer), hop_fwd (person starts, studio answer)")
        else:
            print("  WARNING: files from different worlds pooled; summarize them separately")
    bagg = {}
    for r in results:
        for c, bd in r.get("binding", {}).items():
            bagg.setdefault(c, []).append(bd)
    if bagg:
        print("\n  generated sentences by class (mixed = entities from 2+ films, a binding error)")
        for c, v in bagg.items():
            print(f"    {c:12s}" + "  ".join(f"{k} {mean(x[k] for x in v):.2f}" for k in dc.CATS))


def summarize(patterns):
    results = []
    for pat in patterns:
        for fn in sorted(glob.glob(pat)):
            with open(fn) as fh:
                results.append(json.load(fh))
    if not results:
        print("no files")
        return
    print(f"=== {len(results)} file(s), seeds {sorted(r['seed'] for r in results)}")
    dc.print_table(results)
    print_starts(results)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=NEW_CONDITIONS)
    ap.add_argument("--init", choices=["facts", "base"], default="facts")
    ap.add_argument("--facts", type=int, default=40, help="must match the deriv run to pool")
    ap.add_argument("--distractors", type=int, default=4)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--n-seq", type=int, default=960, help="sequences per condition")
    ap.add_argument("--max-new-opd", type=int, default=160)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--gen-bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--div", choices=["fkl", "rkl", "jsd"], default="fkl")
    ap.add_argument("--keep-samples", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    ap.add_argument("--out-prefix", default="dist")
    ap.add_argument("--summarize", nargs="+", metavar="JSON")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return
    if a.quick:
        a.facts, a.epochs, a.n_seq, a.max_new_opd, a.gen_bs = 10, 2, 24, 48, 8
    from source_reliability_online import Learner
    model, tok = load_model(a.model, dtype=a.dtype)
    L = Learner(model, tok, a.rank, a.alpha, a.lr, a.steps, seed=0)
    for seed in a.seeds:
        run_seed(L, tok, a, seed)


if __name__ == "__main__":
    main()
