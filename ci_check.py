"""Confidence intervals for the workshop note's contrasts, from existing result files.

Every deriv/dist result file stores per-film accuracy for every probe and
condition, so no new runs are needed. For each contrast:
  estimate   pooled over seeds and films
  95% CI     bootstrap: films resampled with replacement within each seed
             (stratified by seed), seeds pooled weighted by film count
  per seed   the estimate in each seed separately (3 seeds is too few to
             resample seeds, so consistency across them is shown instead)
Contrasts are paired: both terms come from the same seed and film. Worlds
differ between models (names are matched to each tokenizer), so contrasts only
pair files of one model and one world.

Two kinds of contrast
  diff   A - B (accuracy difference, same probe)
  gap    (C - facts only) / (in context - facts only), the share of the
         in-context gap closed; facts only and in context come from the
         group's deriv files

Groups (label = glob; override with --group LABEL=GLOB, skip by pointing at
nothing). Condition names repeat across groups (opd_cued is forward KL in QF
and reverse KL in QR), so every term is written GROUP:condition.
  Qwen, standard world   Qd deriv_s*  QF distF_s*  QV distV_s*  QU distU_s*
                         QN distN_s*  QR distR_s*  QC distC_s* (cued + names list)
                         QM distM_s* (cued, cue mid-sentence)
  Llama, standard world  Ld derivL_s*  LF distL_s*  LR distLR_s*
  Qwen, crossed world    Xd derivX_s*  XF distX_s*  XR distXR_s*
A contrast whose files or condition are missing is skipped, with the
conditions that group does have.

--samples also checks the stored student samples (the last quarter of
training, up to --keep-samples per seed) for studio-start sentences that name
the director of that studio's film: does the 960-sequence no-film run, with 12
prompts per studio, write studio -> director links more often than the
640-sequence one? (Section 3.2's open question.)

Usage
  python ci_check.py                       # all contrasts, 10000 resamples
  python ci_check.py --samples             # plus the sample check
  python ci_check.py --only 3.2 --B 2000   # one section
  python ci_check.py --csv ci.csv
Needs numpy.
"""
import argparse
import csv
import glob
import json
import re
import sys

import numpy as np

DEFAULT_GROUPS = {
    "Qd": "deriv_s*.json", "QF": "distF_s*.json", "QV": "distV_s*.json",
    "QU": "distU_s*.json", "QN": "distN_s*.json", "QR": "distR_s*.json",
    "QC": "distC_s*.json", "QM": "distM_s*.json",
    "Ld": "derivL_s*.json", "LF": "distL_s*.json", "LR": "distLR_s*.json",
    "Xd": "derivX_s*.json", "XF": "distX_s*.json", "XR": "distXR_s*.json",
}
# deriv group that holds facts-only ("learnt") and "incontext" for each group
DERIV_OF = {"Q": "Qd", "L": "Ld", "X": "Xd"}

STD_NEED = ["rev", "rev_novel", "prod_rev", "hop_fwd", "hop_rev"]
X_NEED = ["fwd_trained", "fwd_novel", "prod_trained", "hop_fwd", "hop_rev"]


def contrasts():
    """(section, name, kind, terms, probe). diff: terms = (A, B); gap: (C,)."""
    out = []

    def diffs(sec, label, a, b, probes):
        for p in probes:
            out.append((sec, f"{label} [{p}]", "diff", (a, b), p))

    def gaps(sec, label, c, probes):
        for p in probes:
            out.append((sec, f"{label} [{p}]", "gap", (c,), p))

    # 3.1 free sampling adds little
    diffs("3.1", "Qwen free - facts only", "QF:opd_free", "Qd:learnt", STD_NEED)
    diffs("3.1", "Qwen prefix - facts only", "QF:opd_prefix", "Qd:learnt", STD_NEED)
    diffs("3.1", "Qwen free no names - free", "QU:opd_free_noname", "QF:opd_free", STD_NEED)
    diffs("3.1", "Llama free - facts only", "LF:opd_free", "Ld:learnt", STD_NEED)
    # 3.2 starting entity
    gaps("3.2", "Qwen cued, gap closed", "QF:opd_cued", STD_NEED)
    gaps("3.2", "Llama cued, gap closed", "LF:opd_cued", STD_NEED)
    diffs("3.2", "Qwen cued - free", "QF:opd_cued", "QF:opd_free", STD_NEED)
    diffs("3.2", "Qwen cued - no film (640)", "QF:opd_cued", "QV:opd_cued_nofilm", STD_NEED)
    diffs("3.2", "Qwen cued - no studio (640)", "QF:opd_cued", "QV:opd_cued_nostudio", STD_NEED)
    diffs("3.2", "Qwen no film (640) - facts only", "QV:opd_cued_nofilm", "Qd:learnt", STD_NEED)
    diffs("3.2", "Qwen no studio (640) - facts only", "QV:opd_cued_nostudio", "Qd:learnt", STD_NEED)
    diffs("3.2", "Qwen no film 960 - no film 640", "QN:opd_cued_nofilm", "QV:opd_cued_nofilm", STD_NEED)
    diffs("3.2", "Llama cued - no film (960)", "LF:opd_cued", "LF:opd_cued_nofilm", STD_NEED)
    diffs("3.2", "Qwen SFT cued - SFT free", "QF:sft_cued", "QF:sft_free", STD_NEED)
    # cued with the free prompt's names list: does the list block consolidation?
    diffs("3.2", "Qwen cued - cued+list", "QF:opd_cued", "QC:opd_cued_list", STD_NEED)
    diffs("3.2", "Qwen cued+list - free", "QC:opd_cued_list", "QF:opd_free", STD_NEED)
    # cue mid-sentence, no list: does sentence-start position matter?
    diffs("3.2", "Qwen cued - cued mid-sentence", "QF:opd_cued", "QM:opd_cued_mid", STD_NEED)
    diffs("3.2", "Qwen cued mid-sentence - facts only", "QM:opd_cued_mid", "Qd:learnt", STD_NEED)
    # 3.3 divergence
    gaps("3.3", "Qwen rKL cued, gap closed", "QR:opd_cued", STD_NEED)
    gaps("3.3", "Llama rKL cued, gap closed", "LR:opd_cued", STD_NEED)
    gaps("3.3", "Crossed fKL cued, gap closed", "XF:opd_cued", X_NEED)
    gaps("3.3", "Crossed rKL cued, gap closed", "XR:opd_cued", X_NEED)
    diffs("3.3", "Qwen fKL - rKL, cued", "QF:opd_cued", "QR:opd_cued", STD_NEED)
    diffs("3.3", "Llama fKL - rKL, cued", "LF:opd_cued", "LR:opd_cued", STD_NEED)
    diffs("3.3", "Crossed fKL - rKL, cued", "XF:opd_cued", "XR:opd_cued", X_NEED)
    diffs("3.3", "Crossed rKL cued - facts only", "XR:opd_cued", "Xd:learnt", X_NEED)
    # 3.4 smaller observations
    diffs("3.4", "Qwen SFT cued - OPD cued", "QF:sft_cued", "QF:opd_cued", STD_NEED)
    diffs("3.4", "Qwen OPD cued - in context", "QF:opd_cued", "Qd:incontext", ["hop_fwd"])
    diffs("3.4", "Qwen no film (640) - in context", "QV:opd_cued_nofilm", "Qd:incontext", ["hop_fwd"])
    diffs("3.4", "Llama OPD cued - in context", "LF:opd_cued", "Ld:incontext", ["hop_fwd"])
    return out


# ------------------------------------------------------------------ loading

def load_groups(groups):
    """data[group][cond][seed][fact][probe] = acc; raw[group] = list of files."""
    data, raw = {}, {}
    for g, pat in groups.items():
        files = sorted(glob.glob(pat)) if pat else []
        data[g], raw[g] = {}, []
        for fn in files:
            with open(fn) as fh:
                r = json.load(fh)
            raw[g].append(r)
            s = r["seed"]
            for c, rows in r.get("conditions", {}).items():
                d = data[g].setdefault(c, {})
                if s in d:
                    print(f"  note: {g}:{c} seed {s} appears in more than one file; "
                          f"keeping {fn}", file=sys.stderr)
                d[s] = {}
                for x in rows:
                    d[s].setdefault(x["fact"], {})[x["probe"]] = float(x["acc"])
    return data, raw


def term(data, t):
    g, c = t.split(":", 1)
    return data.get(g, {}).get(c)


def aligned(data, terms, probe):
    """{seed: array (k, n_films)} over seeds and films present in every term."""
    tabs = [term(data, t) for t in terms]
    if any(x is None for x in tabs):
        return None
    seeds = sorted(set.intersection(*(set(x) for x in tabs)))
    out = {}
    for s in seeds:
        facts = sorted(set.intersection(*(set(x[s]) for x in tabs)))
        facts = [f for f in facts if all(probe in x[s][f] for x in tabs)]
        if facts:
            out[s] = np.array([[x[s][f][probe] for f in facts] for x in tabs])
    return out or None


def missing_msg(data, terms):
    parts = []
    for t in terms:
        g, c = t.split(":", 1)
        if not data.get(g):
            parts.append(f"no files for group {g}")
        elif c not in data[g]:
            parts.append(f"{g} has no {c} (has: {', '.join(sorted(data[g]))})")
    return "; ".join(parts) or "no common seeds or films"


# ------------------------------------------------------------------ statistics

def stat(kind, means):
    """means: array (k, ...) of term means."""
    if kind == "diff":
        return means[0] - means[1]
    c, l, i = means
    den = i - l
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 1e-9, (c - l) / den, np.nan)


def bootstrap(kind, mats, B, rng):
    """mats: {seed: (k, n)}. Returns estimate, (lo, hi), per-seed estimates, n."""
    seeds = sorted(mats)
    ns = np.array([mats[s].shape[1] for s in seeds], float)
    w = ns / ns.sum()
    pooled = sum(wi * mats[s].mean(1) for wi, s in zip(w, seeds))
    est = float(stat(kind, pooled))
    per_seed = [float(stat(kind, mats[s].mean(1))) for s in seeds]
    acc = None
    for wi, s in zip(w, seeds):
        m = mats[s]
        idx = rng.integers(0, m.shape[1], size=(B, m.shape[1]))
        res = m[:, idx].mean(2)                      # (k, B)
        acc = wi * res if acc is None else acc + wi * res
    boot = stat(kind, acc)
    boot = boot[~np.isnan(boot)]
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
    return est, (float(lo), float(hi)), per_seed, int(ns.sum())


def gap_terms(c):
    d = DERIV_OF[c.split(":")[0][0]]
    return (c, f"{d}:learnt", f"{d}:incontext")


# ------------------------------------------------------------------ sample check

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
LIST_MARK = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def sample_check(raw, cases):
    print("\n=== sample check: studio-start sentences in stored student samples")
    print("  (samples are the last quarter of training, up to --keep-samples per seed)")
    print(f"    {'run':28s}{'sentences':>10s}{'studio-start':>13s}{'+ own film':>11s}"
          f"{'+ own director':>15s}{'+ director, no film':>20s}")
    for label, g, cond in cases:
        tot = st = film = dire = dire_nofilm = 0
        for r in raw.get(g, []):
            texts = r.get("samples", {}).get(cond, [])
            facts = r["facts"]
            studios = sorted(facts, key=lambda f: -len(f["s"]))
            for txt in texts:
                for sent in SENT_SPLIT.split(txt.strip()):
                    sent = LIST_MARK.sub("", sent).strip().lstrip("\"'*")
                    if not sent:
                        continue
                    tot += 1
                    f = next((f for f in studios if sent.startswith(f["s"])), None)
                    if f is None:
                        continue
                    st += 1
                    hf, hd = f["t"] in sent, f["p"] in sent
                    film += hf
                    dire += hd
                    dire_nofilm += hd and not hf
        if tot == 0:
            print(f"    {label:28s}  no stored samples ({g}:{cond})")
            continue
        sh = lambda x: f"{x / st:.2f}" if st else "n/a"
        print(f"    {label:28s}{tot:>10d}{st / tot:>13.2f}{sh(film):>11s}{sh(dire):>15s}"
              f"{sh(dire_nofilm):>20s}")
    print("  If '+ own director' is clearly higher in the 960 no-film run than the 640 one,\n"
          "  more prompts per studio produced direct studio -> director links, which would\n"
          "  explain studio -> director at 0.68 there with film -> director at chance.")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=GLOB")
    ap.add_argument("--B", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="+", metavar="SECTION", help="e.g. 3.2 3.3")
    ap.add_argument("--samples", action="store_true", help="also run the sample check")
    ap.add_argument("--csv", help="write all rows to this file")
    a = ap.parse_args()

    groups = dict(DEFAULT_GROUPS)
    for spec in a.group:
        k, _, v = spec.partition("=")
        groups[k] = v
    data, raw = load_groups(groups)
    found = {g: len(raw[g]) for g in groups if raw[g]}
    print("files per group: " + ", ".join(f"{g} {n}" for g, n in found.items()))
    for g, rs in raw.items():
        ws = {r.get("world", "?") for r in rs}
        if g.startswith("X") and "standard" in ws:
            print(f"  WARNING: group {g} contains files saved as world 'standard'")

    rng = np.random.default_rng(a.seed)
    rows, section = [], None
    for sec, name, kind, terms, probe in contrasts():
        if a.only and sec not in a.only:
            continue
        if sec != section:
            section = sec
            print(f"\n=== {sec}   (estimate, 95% CI, per-seed; "
                  f"diff = accuracy difference, gap = share of in-context gap closed)")
        full = gap_terms(terms[0]) if kind == "gap" else terms
        mats = aligned(data, full, probe)
        if mats is None:
            print(f"  {name:52s} skipped: {missing_msg(data, full)}")
            continue
        est, (lo, hi), ps, n = bootstrap(kind, mats, a.B, rng)
        excl = "" if lo <= 0 <= hi else " *"
        print(f"  {name:52s} {est:+.2f}  [{lo:+.2f}, {hi:+.2f}]{excl:2s}  "
              f"seeds {' '.join(f'{x:+.2f}' for x in ps)}  (n {n})")
        rows.append({"section": sec, "contrast": name, "kind": kind, "probe": probe,
                     "terms": " vs ".join(full), "estimate": est, "lo": lo, "hi": hi,
                     "per_seed": " ".join(f"{x:.3f}" for x in ps), "n": n})
    print("\n  * = CI excludes 0. Films are resampled within seed; the CIs do not include\n"
          "  between-seed variation, so read them with the per-seed column.")

    if a.csv and rows:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"saved {a.csv}")

    if a.samples:
        sample_check(raw, [("opd_cued 960 (8/entity)", "QF", "opd_cued"),
                           ("no film 960 (12/entity)", "QN", "opd_cued_nofilm"),
                           ("no film 640 (8/entity)", "QV", "opd_cued_nofilm")])


if __name__ == "__main__":
    main()
