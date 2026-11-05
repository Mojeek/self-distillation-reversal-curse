"""Negation check for the crossed world: are reverse-KL negations concentrated on
the answer type that fails?

Under reverse KL in the crossed world, 40% of the student's sentences are
negations (18% under forward KL), and the rows whose answer is a studio fail.
If negations explained the failure, they should be concentrated on studio
links: sentences starting from a film (film -> studio) or naming a studio. If
they are spread like all other sentences, negations cannot explain why studio
answers fail while film and person answers partly recover.

For each file group, over the stored student samples (last quarter of training,
up to 60 per seed), every sentence is classified as in distill_check.py, then:
  starts    share of sentences starting with a person / film / studio / other,
            for negations and for all sentences
  names     share of sentences that name any studio / person / film
  negation rate by starting entity: of sentences starting with a person, a
            film, a studio, the share that are negations

Usage (in the folder with the result files)
  python negation_check.py
  python negation_check.py --group XR="distXR_s*.json" --group XF="distX_s*.json"
Needs derivation_check.py alongside (for the negation pattern and list markers).
"""
import argparse
import glob
import json
import re
from collections import Counter

import derivation_check as dc

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
DEFAULT = {"crossed rKL": "distXR_s*.json", "crossed fKL": "distX_s*.json",
           "standard rKL": "distR_s*.json", "standard fKL": "distF_s*.json"}
KEYS = ("p", "t", "s")
NAME = {"p": "person", "t": "film", "s": "studio"}


def analyse(files, cond):
    neg_start, all_start = Counter(), Counter()
    neg_names, all_names = Counter(), Counter()
    n_neg = n_all = 0
    for fn in files:
        with open(fn) as fh:
            r = json.load(fh)
        facts = r["facts"]
        names = sorted(((f[k], k) for f in facts for k in KEYS), key=lambda x: -len(x[0]))
        for txt in r.get("samples", {}).get(cond, []):
            for sent in SENT_SPLIT.split(txt.strip()):
                sent = dc.LIST_MARK.sub("", sent).strip().lstrip("\"'*")
                if not sent:
                    continue
                start = next((k for nm, k in names if sent.startswith(nm)), "other")
                mentioned = {k for nm, k in names if nm in sent}
                is_neg = bool(dc.NEG.search(sent))
                n_all += 1
                all_start[start] += 1
                for k in mentioned:
                    all_names[k] += 1
                if is_neg:
                    n_neg += 1
                    neg_start[start] += 1
                    for k in mentioned:
                        neg_names[k] += 1
    return n_neg, n_all, neg_start, all_start, neg_names, all_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=GLOB")
    ap.add_argument("--cond", default="opd_cued")
    a = ap.parse_args()
    groups = dict(DEFAULT)
    for spec in a.group:
        k, _, v = spec.partition("=")
        groups[k] = v
    for label, pat in groups.items():
        files = sorted(glob.glob(pat))
        if not files:
            print(f"\n{label}: no files for {pat}")
            continue
        n_neg, n_all, ns, as_, nn, an = analyse(files, a.cond)
        if n_all == 0:
            print(f"\n{label}: no stored samples for {a.cond}")
            continue
        print(f"\n=== {label}  ({len(files)} files, {n_all} sentences, "
              f"{n_neg} negations = {n_neg / n_all:.0%})")
        print("  starts with:        " + "  ".join(f"{NAME.get(k, k):>7s}" for k in KEYS + ("other",)))
        if n_neg:
            print("    negations         " + "  ".join(f"{ns[k] / n_neg:7.2f}" for k in KEYS + ("other",)))
        print("    all sentences     " + "  ".join(f"{as_[k] / n_all:7.2f}" for k in KEYS + ("other",)))
        print("  names a:            " + "  ".join(f"{NAME[k]:>7s}" for k in KEYS))
        if n_neg:
            print("    negations         " + "  ".join(f"{nn[k] / n_neg:7.2f}" for k in KEYS))
        print("    all sentences     " + "  ".join(f"{an[k] / n_all:7.2f}" for k in KEYS))
        print("  negation rate among sentences starting with: " + "  ".join(
            f"{NAME[k]} {ns[k] / as_[k]:.0%}" if as_[k] else f"{NAME[k]} n/a" for k in KEYS))
    print("\nReading: in the crossed world studio answers fail under reverse KL. If negations\n"
          "explained it, crossed-rKL negations would over-represent film starts (film -> studio)\n"
          "or studio names relative to all sentences, and film-start sentences would have a\n"
          "much higher negation rate than person-start ones. Samples are 60 per seed, late in\n"
          "training, so treat small differences as noise.")


if __name__ == "__main__":
    main()
