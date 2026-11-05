# Self-distillation inherits the reversal curse

Code and result files for a small controlled study of what consolidating new
knowledge by self-distillation does and does not teach a language model. The
full write-up is `self-distillation-reversal-curse.pdf` in this folder. A short
summary is on the Mojeek blog:
https://blog.mojeek.com/2026/11/self-distillation-inherits-the-reversal-curse.html

**Short version.** The student first learns the new facts one way round (LoRA
fine-tuning), then is trained by on-policy self-distillation to match a teacher
that has the facts in context. It keeps the facts in the direction given, but
reversed and two-step answers transfer well only when the student's own text
reaches the cue and relation ("The Silent Winter was directed by...") without
the answer being available in its prompt. Under reverse KL, transfer also
depends on whether the answer's entity type ever appears as a continuation in
the learnt facts; types that never do stay near chance, while forward KL
transfers them in our experiments. Forward recall stays at 0.95-1.00
throughout, so neither failure shows in the usual evaluations.

This study was carried out with Claude (Anthropic), which designed most of the
experiments and controls, wrote the analysis code and drafted the write-up.
Marc ran the experiments, decided what to test and checked the results.
Predictions for each control were written into the docstring of
`distill_check.py` before its run; they are not independently timestamped.

## What this does not show

- Anything about open-ended, real-world knowledge: the world is 40 invented
  films with one director and one studio each, and the set of entities is known.
- Learning from scratch: the student has already learnt the facts once, and
  consolidation into a student that had not failed even on forward recall
  (one seed, preliminary).
- A measured mechanism: that the names list lets the student answer by
  recognition, and why reverse KL fails, are explanations that fit the
  results; the training signal at those tokens was not measured.
- Generality across models: two 7-8B models, three seeds, one LoRA setting;
  the list, mid-sentence, volume-matched and crossed-world controls are Qwen only.

## Setup

- Python 3.10+, `pip install torch transformers peft numpy`
- One GPU with ~40 GB or more for the 7-8B models (runs here used an A100 80 GB;
  each three-seed condition takes roughly an hour). Other GPUs have not been tested.
- Models: `Qwen/Qwen2.5-7B-Instruct` (public) and
  `meta-llama/Llama-3.1-8B-Instruct` (gated: accept its licence on Hugging Face
  and log in with `huggingface-cli login` first).
- Each result file records the arguments it was run with (`args`);
  `args_to_commands.py` prints the command behind every file.
- World: 40 invented films per seed (director + studio each), seeds 0 1 2.
  `RECON_WORLD=standard` (default) or `RECON_WORLD=crossed` selects the fact
  templates; set it explicitly for every command.

## Scripts

| Script | What it does |
| --- | --- |
| `preliminary_check.py` | World, probes and scoring; facts in context vs learnt by LoRA |
| `derivation_check.py` | Facts-only, in-context and gold references; the model's own derived notes |
| `distill_check.py` | Self-distillation conditions (free, prefix, cued and controls), forward or reverse KL, on- or off-policy |
| `logprob_check.py` | Log-probability check (reported in the write-up as a failed prediction) |
| `ci_check.py` | Bootstrap intervals for every contrast in the write-up, from the result files |
| `order_check.py` | Counts, in stored student samples, sentences that state each link in order (`--visits`: the context only) |
| `negation_check.py` | Negations and answer types in stored samples |
| `common.py`, `source_reliability_online.py` | Shared helpers and the LoRA learner |
| `args_to_commands.py` | Prints the exact command behind each result file |

## Reproducing the result files

Exact commands, read from the arguments saved in each result file
(`python args_to_commands.py` prints them). All other settings are the
scripts' defaults. Run each with `RECON_WORLD` set as shown.

| Files | World | Command |
| --- | --- | --- |
| `deriv_s*.json` | standard | `python derivation_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions incontext learnt learnt_gold self_neutral self_explicit --out-prefix deriv` |
| `deriv2_s*.json` | standard | `python derivation_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions self_cued self_qa --out-prefix deriv2` |
| `derivX_s*.json` | crossed | `python derivation_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions incontext learnt --out-prefix derivX` |
| `derivL_s*.json` | standard | `python derivation_check.py --model meta-llama/Llama-3.1-8B-Instruct --seeds 0 1 2 --conditions incontext learnt learnt_gold --out-prefix derivL` |
| `distF_s*.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions sft_free sft_cued opd_free opd_prefix opd_cued --out-prefix distF` |
| `distN_s*.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_cued_nofilm --out-prefix distN` |
| `distV_s*.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_cued_nofilm opd_cued_nostudio --n-seq 640 --out-prefix distV` |
| `distU_s*.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_free_noname --out-prefix distU` |
| `distC_s*.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_cued_list --out-prefix distC` |
| `distM_s*.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_cued_mid --out-prefix distM` |
| `distR_s*.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_free opd_prefix opd_cued opd_cued_nofilm --div rkl --out-prefix distR` |
| `distX_s*.json` | crossed | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_cued --out-prefix distX` |
| `distXR_s*.json` | crossed | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --conditions opd_cued --div rkl --out-prefix distXR` |
| `distL_s*.json` | standard | `python distill_check.py --model meta-llama/Llama-3.1-8B-Instruct --seeds 0 1 2 --conditions opd_free opd_cued opd_cued_nofilm --out-prefix distL` |
| `distLR_s*.json` | standard | `python distill_check.py --model meta-llama/Llama-3.1-8B-Instruct --seeds 0 1 2 --conditions opd_cued --div rkl --out-prefix distLR` |
| `distB_s0.json` | standard | `python distill_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 --conditions opd_free opd_cued opd_prefix --init base --out-prefix distB` (preliminary, one seed) |
| `lp_s*.json` | standard | `python logprob_check.py --model Qwen/Qwen2.5-7B-Instruct --seeds 0 1 2 --out-prefix lp` |
| `lpL_s0.json` | standard | `python logprob_check.py --model meta-llama/Llama-3.1-8B-Instruct --seeds 0 --out-prefix lpL` |

Files written before the scripts recorded their world (`RECON_WORLD`) were
all standard-world except `derivX`, which is crossed: summarise it with
`RECON_WORLD=crossed` in front. Smoke-test files (`distQ`, `lpQ`) are not
part of the results and are not included.

## Reproducing the analysis

In the folder with the result files (no GPU needed):

```
python distill_check.py --summarize "deriv_s*.json" "distF_s*.json" "distV_s*.json" "distU_s*.json" "distC_s*.json" "distM_s*.json"
RECON_WORLD=crossed python distill_check.py --summarize "derivX_s*.json" "distX_s*.json"
python ci_check.py --csv ci.csv
python order_check.py
python order_check.py --visits
python negation_check.py
```

`ci.csv` holds the estimate, interval and per-seed value for every contrast
reported in the write-up.

## Licence

MIT: see `LICENSE`.
