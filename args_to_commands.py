"""Print the exact command behind each result file, from the arguments it saved.

Every result file written by preliminary_check.py, derivation_check.py,
distill_check.py and logprob_check.py stores the arguments it was run with
("args"). This groups the files by output prefix, takes the union of seeds,
and prints one command per distinct set of arguments, showing every flag that
differs from the script's default (plus --model, --seeds, --out-prefix and
--conditions always). Output is a Markdown table for the README.

Usage (in the folder with the result files; no GPU, no torch)
  python args_to_commands.py
  python args_to_commands.py --glob "dist*_s*.json"
"""
import argparse
import glob
import json
import re
from collections import defaultdict

DEFAULTS = {
    "distill_check.py": dict(model="Qwen/Qwen2.5-7B-Instruct", init="facts", facts=40,
                             distractors=4, window=5, n_seq=960, max_new_opd=160, temp=1.0,
                             gen_bs=16, accum=4, div="fkl", keep_samples=60, epochs=12,
                             steps=1, rank=16, alpha=32, lr=1e-4, gen=False, quick=False,
                             dtype="auto"),
    "derivation_check.py": dict(model="Qwen/Qwen2.5-7B-Instruct", facts=40, distractors=4,
                                window=5, max_new=400, max_questions=25, epochs=12, steps=1,
                                rank=16, alpha=32, lr=1e-4, gen=False, quick=False,
                                dtype="auto"),
    "logprob_check.py": dict(model="Qwen/Qwen2.5-7B-Instruct", facts=40, distractors=4,
                             epochs=12, steps=1, rank=16, alpha=32, lr=1e-4, quick=False,
                             dtype="auto"),
    "preliminary_check.py": dict(model="Qwen/Qwen2.5-7B-Instruct", facts=40, distractors=4,
                                 epochs=5, steps=1, rank=16, alpha=32, lr=1e-4, gen=True,
                                 quick=False, dtype="auto"),
}
SKIP = {"seeds", "summarize", "out_prefix", "conditions", "model"}
BOOL_FLAGS = {"gen": ("--gen", "--no-gen"), "quick": ("--quick", None)}


def which_script(r):
    if "starts" in r or "samples" in r:
        return "distill_check.py"
    if "derive" in r or "n_train" in r:
        return "derivation_check.py"
    if "base" in r and "learnt" in r and "conditions" not in r:
        return "logprob_check.py"
    return "preliminary_check.py"


def flag(k, v):
    if k in BOOL_FLAGS:
        on, off = BOOL_FLAGS[k]
        return on if v else off
    return f"--{k.replace('_', '-')} {v}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*_s*.json")
    a = ap.parse_args()
    groups = defaultdict(list)
    for fn in sorted(glob.glob(a.glob)):
        try:
            with open(fn) as fh:
                r = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if "args" not in r or "seed" not in r:
            continue
        prefix = re.sub(r"_s\d+\.json$", "", fn)
        groups[prefix].append((fn, r))
    if not groups:
        print("no result files found")
        return
    print("| Files | Seeds | Command |")
    print("| --- | --- | --- |")
    for prefix, items in groups.items():
        variants = defaultdict(list)
        for fn, r in items:
            script = which_script(r)
            args = r["args"]
            key = json.dumps({k: v for k, v in args.items() if k not in ("seeds", "summarize")},
                             sort_keys=True)
            variants[(script, key, r.get("world", "not recorded"))].append(r["seed"])
        for (script, key, world), seeds in variants.items():
            args = json.loads(key)
            d = DEFAULTS[script]
            parts = [f"RECON_WORLD={world}" if world != "not recorded" else "RECON_WORLD=?",
                     f"python {script}", f"--model {args.get('model')}",
                     "--seeds " + " ".join(str(s) for s in sorted(set(seeds)))]
            if args.get("conditions"):
                parts.append("--conditions " + " ".join(args["conditions"]))
            for k, v in args.items():
                if k in SKIP or k not in d:
                    continue
                if v != d[k]:
                    f = flag(k, v)
                    if f:
                        parts.append(f)
            parts.append(f"--out-prefix {args.get('out_prefix', prefix)}")
            note = ""
            if args.get("quick"):
                note = " (QUICK RUN: not a real result)"
            if world == "not recorded":
                note += " (world not saved by this version: check)"
            print(f"| `{prefix}_s*.json` | {' '.join(str(s) for s in sorted(set(seeds)))} "
                  f"| `{' '.join(parts)}`{note} |")


if __name__ == "__main__":
    main()
