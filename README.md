# hillclimb-arena

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.x-3572A5.svg)

A local harness for testing LLM robustness against adversarial prompting. It runs
evaluation suites against models served by [OpenCode](https://opencode.ai) and measures
refusal behavior, technique-induced compliance, and canary-secret extraction.

See [`SHOWCASE.md`](SHOWCASE.md) for the generated findings showcase (heatmaps, leaderboards, refusal direction). A fresh clone can regenerate it from the committed data with `python tools/showcase.py` (writes `SHOWCASE.md` and `docs/`); `python tools/assemble_refusal.py` only rebuilds `data/campaign-refusal.json` from a local (gitignored) `runs/` directory during development.

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

## Evaluation results (2026-10-17)

A canary-extraction robustness campaign (harness `8e0f2e8`, 9 free-tier models, 2-3 trials per cell,
confirmation reruns on fresh secrets) is summarized in [REPORT.md](REPORT.md).

- 8 of 9 models produced at least one confirmed extraction; `beam-free` swept all 12 cells,
  `nemotron-3.5-super-vl-free` confirmed 9/9 tested cells.
- Every completed adaptive (dynamic) session leaked; the three new replay/escalation templates
  added 17 confirmed trials.
- `nemotron-3-ultra-free` was untestable (upstream 503). Cells are counts, not percentages -
  see the report for the confirmation rule.

## Layout

- `tools/arena.py` — shared plumbing (agent spawning, server, cost caps)
- `tools/techniques_eval.py` — main evaluator (all suites above)
- `tools/assemble_refusal.py`, `tools/showcase.py` — refusal-wave assembly and the deterministic findings showcase
- `tools/jbb_eval.py` — JBB-shaped canary-extraction evaluator
- `tools/forge.py`, `tools/hunt.py`, `tools/persona.py`, `tools/replay.py` — canary-extraction attack tools
- `behaviors.json`, `personas.json`, `corpus_example.json` — fixtures at repo root
- `REPORT.md` — campaign report; `LICENSE` — MIT
- `runs/` — generated artifacts (gitignored)

## Requirements

- Python 3 (standard library only)
- [`opencode`](https://opencode.ai) CLI for live runs — the harness drives the opencode 2.x
  `serve` API and seeds the scratch credentials itself; 1.x is not supported for live calls.

## Usage

Dry-run (offline, no API calls):

```bash
python3 tools/hunt.py --dry-run
python3 tools/techniques_eval.py --dry-run
python3 tools/techniques_eval.py --mode mildharm --tier sharp --hillclimb 4 --dry-run
```

Live run (cost-capped):

```bash
python3 tools/techniques_eval.py --mode mildharm --tier sharp --hillclimb 4 \
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

## License

MIT — see [LICENSE](LICENSE).
