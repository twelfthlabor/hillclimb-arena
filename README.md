# hillclimb-arena

A local harness for testing LLM robustness against adversarial prompting. It runs
evaluation suites against models served by [OpenCode](https://opencode.ai) and measures
refusal behavior, technique-induced compliance, and canary-secret extraction.

## Suites

| Suite | What it measures |
|---|---|
| `techniques` | 10 prompt techniques vs. a canary secret (leak = bad) |
| `overrefusal` | Harmless dev/security tasks (refusal = false positive) |
| `boundary` | Refusal-adjacent tasks; refused-then-accepted transitions |
| `mildharm` (opt-in) | Mildly harmful requests; all-refused is the desired outcome |

`mildharm` supports tiers (`--tier mild|moderate|sharp|external`) and an adaptive
attacker loop (`--hillclimb R`) that iterates strategies until the target complies
or the budget is exhausted.

## Layout

- `arena.py` — shared plumbing (agent spawning, server, cost caps)
- `techniques_eval.py` — main evaluator (all suites above)
- `jbb_eval.py` — JBB-shaped canary-extraction evaluator
- `forge.py`, `hunt.py`, `persona.py`, `replay.py` — canary-extraction attack tools
- `*.json` — test data (behaviors, personas, example corpus)

## Requirements

- Python 3 (standard library only)
- [`opencode`](https://opencode.ai) CLI for live runs

## Usage

Dry-run (offline, no API calls):

```bash
python3 techniques_eval.py --dry-run
python3 techniques_eval.py --mode mildharm --tier sharp --hillclimb 4 --dry-run
```

Live run (cost-capped):

```bash
python3 techniques_eval.py --mode mildharm --tier sharp --hillclimb 4 \
  --model opencode/space-bunny-free --max-cost 0.10
```

See `--help` for all options. Live runs use an isolated scratch environment and verify
server isolation before collecting rows.

## Scope & safety

- Authored evaluation content is **non-operational, text-level social/petty harm only** —
  no operational serious-harm instructions, no real targets, no escalation.
- The project's adversarial `AGENTS.md` fixture (an injected jailbreak-style spec used to
  test instruction-hierarchy robustness) is intentionally **not published**; it remains a
  local-only file.
- Live results are descriptive: n=1 per task, unseeded, and the attacker and target may
  be the same model.

## Generated artifacts

`runs/` (per-run CSVs, response logs, summaries) is gitignored — it contains canary
secrets and session data.
