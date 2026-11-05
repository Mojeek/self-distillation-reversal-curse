"""Order check: does consolidation follow the order in which sampled text states
each link?

Revised claim: a probe asks for an answer after its cue entity ("{t} was directed
by" -> person), so consolidation needs sampled text in which the cue precedes the
answer, as a stated link, anywhere in a sentence, not only at its start.

For every stored student sample (the last quarter of training, up to 60 per seed),
each sentence is checked, for each fact of that seed, for an ordered stated link:
  - both entities of the SAME fact appear in the sentence
  - the cue entity comes first
  - (strict, default) a relation word lies between them: "direct" for person-film
    pairs, "produc|studio|release|behind" for studio-film pairs, any of these
    for studio-person (two-step) pairs
  - (default) the sentence is not a negation
Co-occurrence alone is not counted, and the prompt is never counted: only what the
student wrote.

Link types, named by the probe they serve (prompt entity -> answer):
  fwd     person -> film       (fwd_trained, fwd_novel)
  rev     film -> person       (rev, rev_novel)
  t>s     film -> studio       (prod_trained)
  s>t     studio -> film       (prod_rev)
  p>s     person -> studio     (hop_fwd)
  s>p     studio -> person     (hop_rev)
For each run: the share of sentences stating each ordered link, next to the
accuracy of the probe it serves (from the same result files). If the revised claim
holds, a probe's accuracy should track its link's share: near zero share -> near
facts-only accuracy, whatever the condition, model or world. This does not hold
for reverse KL by design (the second condition), so read those rows separately.

Usage (in the folder with the result files; no GPU, no torch)
  python order_check.py
  python order_check.py --loose        # no relation word required
  python order_check.py --keep-neg     # count negated sentences too
  python order_check.py --visits       # count the cue -> relation context, whatever
                                       # answer follows (right, wrong or invented)
  python order_check.py --group QF="distF_s*.json" --group QV="distV_s*.json"
"""
import argparse
import glob
import json
import re
from collections import defaultdict

DEFAULT = {"Qwen fKL": "distF_s*.json", "Qwen no-names": "distU_s*.json",
           "Qwen matched controls": "distV_s*.json", "Qwen no-film 960": "distN_s*.json",
           "Qwen cued + list": "distC_s*.json", "Qwen cued mid-sentence": "distM_s*.json", "Qwen rKL": "distR_s*.json", "crossed fKL": "distX_s*.json",
           "crossed rKL": "distXR_s*.json", "Llama fKL": "distL_s*.json",
           "Llama rKL": "distLR_s*.json"}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
LIST_MARK = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")      # as derivation_check.py
NEG = re.compile(r"\b(not|never|no longer)\b|n't\b", re.I)   # as derivation_check.py
REL = {"pt": re.compile(r"direct", re.I),
       "st": re.compile(r"produc|studio|release|behind", re.I),
       "sp": re.compile(r"direct|produc|studio|release|behind|hire", re.I)}
# for --visits: the relation word that opens a slot for THIS link's answer type
# (a person after "direct", a studio after "produc/studio", a film after either)
VISIT_REL = {"fwd": re.compile(r"direct", re.I),
             "rev": re.compile(r"direct", re.I),
             "t>s": re.compile(r"produc|studio|release|behind", re.I),
             "s>t": re.compile(r"produc|release|behind", re.I),
             "p>s": re.compile(r"produc|studio|release|behind", re.I),
             "s>p": re.compile(r"direct", re.I)}
# link: (cue key, answer key, relation group, probes it serves)
LINKS = {"fwd": ("p", "t", "pt", ["fwd_trained", "fwd_novel"]),
         "rev": ("t", "p", "pt", ["rev", "rev_novel"]),
         "t>s": ("t", "s", "st", ["prod_trained"]),
         "s>t": ("s", "t", "st", ["prod_rev"]),
         "p>s": ("p", "s", "sp", ["hop_fwd"]),
         "s>p": ("s", "p", "sp", ["hop_rev"])}


def visited(sent, facts):
    """Link types whose CONTEXT the sentence visits: a cue entity of the right type
    followed later by its relation word, whatever answer (if any) comes after.
    Cued samples often name a wrong director yet train the row well, so this is
    the context the teacher scores, not a correct statement."""
    low = sent.lower()
    found = set()
    for name, (cue, _, _, _) in LINKS.items():
        for f in facts:
            a = low.find(f[cue].lower())
            if a >= 0 and VISIT_REL[name].search(sent[a + len(f[cue]):]):
                found.add(name)
                break
    return found


def ordered_links(sent, facts, strict):
    """Set of link types this sentence states, for any one fact, in order."""
    low = sent.lower()
    found = set()
    for f in facts:
        pos = {k: low.find(f[k].lower()) for k in ("p", "t", "s")}
        for name, (cue, ans, rel, _) in LINKS.items():
            a, b = pos[cue], pos[ans]
            if a < 0 or b < 0 or a >= b:
                continue
            if strict and not REL[rel].search(sent[a + len(f[cue]):b]):
                continue
            found.add(name)
    return found


def run(files, strict, keep_neg, visits=False):
    """{cond: {'n': sentences, link: count, 'acc': {probe: [acc]}, 'seeds': set}}"""
    out = defaultdict(lambda: {"n": 0, "links": defaultdict(int),
                               "acc": defaultdict(list), "seeds": set()})
    for fn in files:
        with open(fn) as fh:
            r = json.load(fh)
        facts = r.get("facts", [])
        for cond, texts in r.get("samples", {}).items():
            o = out[cond]
            o["seeds"].add(r.get("seed"))
            o["world"] = r.get("world", "?")
            for txt in texts:
                for sent in SENT_SPLIT.split(txt.strip()):
                    sent = LIST_MARK.sub("", sent).strip().lstrip("\"'*")
                    if not sent:
                        continue
                    if not keep_neg and NEG.search(sent):
                        continue
                    o["n"] += 1
                    links = visited(sent, facts) if visits else ordered_links(sent, facts, strict)
                    for name in links:
                        o["links"][name] += 1
            for x in r.get("conditions", {}).get(cond, []):
                o["acc"][x["probe"]].append(x["acc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=GLOB")
    ap.add_argument("--loose", action="store_true", help="no relation word required")
    ap.add_argument("--keep-neg", action="store_true", help="count negated sentences")
    ap.add_argument("--visits", action="store_true",
                    help="count the cue -> relation context, whatever answer follows")
    a = ap.parse_args()
    groups = dict(DEFAULT)
    if a.group:
        groups = {}
        for spec in a.group:
            k, _, v = spec.partition("=")
            groups[k] = v
    mode = ("contexts visited (cue, then relation word; any answer)" if a.visits else
            "ordered stated links: " + ("loose (no relation word)" if a.loose else "strict"))
    print(f"{mode}"
          f", {'negations kept' if a.keep_neg else 'negations excluded'}")
    print("cell = share of sentences " + ("visiting" if a.visits else "stating")
          + " the link (accuracy of the probe it serves)")
    hdr = "".join(f"{k:>16s}" for k in LINKS)
    for label, pat in groups.items():
        files = sorted(glob.glob(pat))
        if not files:
            continue
        res = run(files, not a.loose, a.keep_neg, a.visits)
        print(f"\n=== {label}  ({len(files)} files)")
        print(f"    {'condition':20s}{'sent':>6s}{hdr}")
        for cond, o in res.items():
            if not o["n"]:
                continue
            cells = []
            for name, (_, _, _, probes) in LINKS.items():
                share = o["links"][name] / o["n"]
                accs = [v for p in probes for v in o["acc"].get(p, [])]
                acc = f"{sum(accs) / len(accs):.2f}" if accs else " n/a"
                cells.append(f"{share:7.1%} ({acc})")
            print(f"    {cond:20s}{o['n']:>6d}" + "".join(f"{c:>16s}" for c in cells))
    print("\nIf the revised claim holds, under forward KL a probe stays near facts-only\n"
          "accuracy wherever its link's share is near zero, and rises where the share\n"
          "is substantial, across conditions and worlds. Under reverse KL, accuracy\n"
          "can stay low despite a substantial share: that is the second condition.\n"
          "Samples are 60 per seed, late in training: compare shares, not exact values.")


if __name__ == "__main__":
    main()
