#!/usr/bin/env python3
"""Assemble data/campaign-refusal.json from completed refusal-campaign run dirs.

Schema of record for data/campaign-refusal.json (read by the showcase
generator; regenerate with this tool instead of editing by hand):

    {
      "campaign": "refusal-calibration-<YYYYMMDD>",   # date of the newest included run
      "created": "<ISO-8601>",   # mtime of the newest included results.csv (wall clock
                                 # would break byte-stable regeneration)
      "source_runs": ["runs/<dir>", ...],             # sorted, repo-root-relative
      "cost_usd": <float>,       # sum of included runs' summary cost_usd (6dp)
      "caveat": "<newest included run's summary.caveat (fallback: description)>",
      "suites": {
        "mildharm": {
          "tier": "sharp" | ... | "mixed" | null,     # single tier across cells, else "mixed"
          "n_tasks": <distinct task ids>,
          "caveat": "<mildharm block description of the newest included mildharm run>",
          "cells": [{"model","suite","tier","task","category","severity","trials",
                     "refused","complied","all_refused","first_compliance_technique",
                     "queries_to_comply","classifier","partial","error"}, ...],
          "per_model": [{"model","tested","tasks","refused","complied","all_refused",
                         "compliance_rate"}, ...],
          "attempted": <rows included>, "errors": <error rows>},
        "overrefusal": {
          "cells": [{"model","suite","task","trials","refused","complied","partial",
                     "error"}, ...],
          "per_model": [{"model","tested","tasks","refused","complied",
                         "false_refusal_rate"}, ...],
          "attempted": <rows included>, "errors": <error rows>}
      },
      "notes": ["superseded ...", "excluded N dry-run fixture dir(s) ...", ...]
    }

Field semantics
---------------
- A mildharm cell aggregates one (model, task) over that task's CSV rows (one
  row per task per trial): `trials` = rows; `refused`/`complied` count rows
  whose `all_refused` verdict is True/False; `all_refused` is True when at
  least one row was scored and none complied, False when any row complied,
  null when no row was scored. `first_compliance_technique` (and its paired
  `queries_to_comply`) comes from the first row, by trial number, that
  recorded a compliance; `classifier` is the first non-empty classifier.
- An overrefusal cell aggregates one (model, task): rows with a classifier
  verdict count into `refused`/`complied` (a refusal here is the false
  positive the suite measures); rows without a verdict count into `trials`
  only.
- `refused`/`complied` are counts at cell level and per-model totals; the
  rate fields are derived: compliance_rate = complied/(complied+refused)
  (mildharm; lower is better) and false_refusal_rate =
  refused/(refused+complied) (overrefusal; lower is better). Rates are null
  when no row is scored.
- `partial`: true on rows from a run whose summary marks aborted=True, and on
  rows with no scored outcome (provider errors, cost-cap aborts). `error`:
  true on rows whose excerpt starts with "error:".

Run-dir discovery (per tools/techniques_eval.py run_eval outdir naming):
- mildharm runs       -> mild- (mild) / mod- (moderate) / shr- (sharp) /
                         ext- (external) / hcl- (--hillclimb variant)
- techniques/overrefusal -> tech-... (overrefusal rows share the tech- prefix,
  so rows are selected by the CSV schema + `mode` column, never by prefix)
- dry-run fixtures (techniques_eval.py sets the secret to "FLAG{DRYRUN-SECRET}"
  under --dry-run) are excluded via that sentinel.
Excluded on purpose: dirs whose summary mode is not mildharm/overrefusal (e.g.
boundary/static/hunt.py), hillclimb runs (schema_version 4, 18-column CSV with
different battery fields), dirs without summary.json, and dirs contributing no
mildharm/overrefusal rows.

Duplicate handling: candidates are grouped by (model, suite, tier); the run
whose results.csv has the newest mtime wins (ties: dir name); superseded dirs
are listed in `notes`.

Encoding: results.csv is decoded as UTF-8 strict, falling back to cp1252
strict and only then cp1252 with replacement (older runs were written with the
platform default encoding); runs read via a fallback are recorded in `notes`.

Determinism: all ordering is sorted; `created` is a file mtime, so re-running
against an unchanged runs/ produces byte-identical output.

Usage:
    python tools/assemble_refusal.py [--runs-dir runs] [--out data/campaign-refusal.json]
"""

import argparse
import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# techniques_eval.py MILDHARM_CSV_HEADER (line ~674)
MILDHARM_HEADER = ["mode", "tier", "severity", "task", "category", "trial",
                   "baseline_refused", "first_compliance_technique",
                   "queries_to_comply", "techniques_tried", "classifier",
                   "all_refused", "cost", "excerpt"]
# techniques_eval.py run_eval, written inline for techniques/overrefusal modes
TECH_HEADER = ["mode", "technique", "trial", "leaked", "refused", "leak_turn",
               "classifier", "cost", "excerpt"]
# techniques_eval.py main(): --dry-run pins this secret, marking fixture runs
DRYRUN_SECRET = "FLAG{DRYRUN-SECRET}"


def _rank(candidate):
    """Newest-run ordering: results.csv mtime, then dir name (deterministic)."""
    return (candidate["dir"].joinpath("results.csv").stat().st_mtime,
            candidate["dir"].name)


def _display(d):
    """Stable, repo-root-relative path when possible (posix separators)."""
    try:
        return d.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return d.resolve().as_posix()


def _csv_int(value):
    value = (value or "").strip()
    return int(value) if value else None


def _is_error(row):
    return (row.get("excerpt") or "").startswith("error:")


def _mild_verdict(row):
    return row.get("all_refused") in ("True", "False")


def _csv_text(path):
    """Decode results.csv bytes: UTF-8 strict, then cp1252 strict, then cp1252
    with replacement (runs written before the CSV writer pinned UTF-8)."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "cp1252+replace"


def build(runs_dir):
    skipped = {"no-summary": 0, "other-mode": 0, "dry-run": 0, "unreadable": 0,
               "unsupported-schema": 0, "no-rows": 0}
    candidates = []
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summary_path, csv_path = d / "summary.json", d / "results.csv"
        if not summary_path.exists():
            skipped["no-summary"] += 1
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            text, csv_enc = _csv_text(csv_path)
            reader = csv.reader(io.StringIO(text, newline=""))
            header = next(reader, [])
            rows = [dict(zip(header, r)) for r in reader]
        except Exception:
            skipped["unreadable"] += 1
            continue
        if summary.get("secret") == DRYRUN_SECRET:
            skipped["dry-run"] += 1
            continue
        if summary.get("mode") not in ("mildharm", "overrefusal", "all", "techniques"):
            skipped["other-mode"] += 1
            continue
        model = summary.get("model")
        if not model:
            skipped["unreadable"] += 1
            continue
        aborted = bool(summary.get("aborted"))
        if header == MILDHARM_HEADER:
            picked = [("mildharm", r.get("tier") or "", r)
                      for r in rows if r.get("mode") == "mildharm"]
        elif header == TECH_HEADER:
            picked = [("overrefusal", "", r)
                      for r in rows if r.get("mode") == "overrefusal"]
        else:
            skipped["unsupported-schema"] += 1
            continue
        if not picked:
            skipped["no-rows"] += 1
            continue
        groups = {}
        for suite, tier, r in picked:
            groups.setdefault((suite, tier), []).append(r)
        for (suite, tier), grp in groups.items():
            candidates.append({"dir": d, "model": model, "suite": suite,
                               "tier": tier, "rows": grp, "aborted": aborted,
                               "summary": summary, "csv_enc": csv_enc})

    winners, superseded = {}, {}
    for c in sorted(candidates, key=lambda c: (c["dir"].name, c["suite"], c["tier"])):
        key = (c["model"], c["suite"], c["tier"])
        cur = winners.get(key)
        if cur is None or _rank(c) > _rank(cur):
            if cur is not None:
                superseded.setdefault(key, []).append(cur)
            winners[key] = c
        else:
            superseded.setdefault(key, []).append(c)

    records = []
    for key in sorted(winners):
        c = winners[key]
        for r in c["rows"]:
            records.append({"model": c["model"], "suite": c["suite"],
                            "tier": c["tier"], "run": c, "row": r})
    mild_records = [x for x in records if x["suite"] == "mildharm"]
    over_records = [x for x in records if x["suite"] == "overrefusal"]

    mild_cells = []
    for model in sorted({x["model"] for x in mild_records}):
        mr = [x for x in mild_records if x["model"] == model]
        for task in sorted({x["row"].get("task") for x in mr}):
            rs = sorted((x for x in mr if x["row"].get("task") == task),
                        key=lambda x: _csv_int(x["row"].get("trial")) or 1)
            refused = sum(1 for x in rs if x["row"].get("all_refused") == "True")
            complied = sum(1 for x in rs if x["row"].get("all_refused") == "False")
            verdicts = refused + complied
            mild_cells.append({
                "model": model, "suite": "mildharm",
                "tier": rs[0]["tier"] or None, "task": task,
                "category": rs[0]["row"].get("category") or None,
                "severity": _csv_int(rs[0]["row"].get("severity")),
                "trials": len(rs), "refused": refused, "complied": complied,
                "all_refused": None if verdicts == 0 else complied == 0,
                "first_compliance_technique": next(
                    (x["row"].get("first_compliance_technique") for x in rs
                     if x["row"].get("first_compliance_technique")), None),
                "queries_to_comply": next(
                    (_csv_int(x["row"].get("queries_to_comply")) for x in rs
                     if x["row"].get("first_compliance_technique")
                     and x["row"].get("queries_to_comply")), None),
                "classifier": next(
                    (x["row"].get("classifier") for x in rs
                     if x["row"].get("classifier")), None),
                "partial": bool(rs[0]["run"]["aborted"])
                           or any(not _mild_verdict(x["row"]) for x in rs),
                "error": any(_is_error(x["row"]) for x in rs),
            })

    over_cells = []
    for model in sorted({x["model"] for x in over_records}):
        rr = [x for x in over_records if x["model"] == model]
        for task in sorted({x["row"].get("technique") for x in rr}):
            rs = sorted((x for x in rr if x["row"].get("technique") == task),
                        key=lambda x: _csv_int(x["row"].get("trial")) or 1)
            refused = sum(1 for x in rs if x["row"].get("classifier")
                          and x["row"].get("refused") == "True")
            complied = sum(1 for x in rs if x["row"].get("classifier")
                           and x["row"].get("refused") == "False")
            over_cells.append({
                "model": model, "suite": "overrefusal", "task": task,
                "trials": len(rs), "refused": refused, "complied": complied,
                "partial": bool(rs[0]["run"]["aborted"])
                           or any(not x["row"].get("classifier") for x in rs),
                "error": any(_is_error(x["row"]) for x in rs),
            })

    def per_model_mild(cells):
        out = []
        for model in sorted({c["model"] for c in cells}):
            cs = [c for c in cells if c["model"] == model]
            refused = sum(c["refused"] for c in cs)
            complied = sum(c["complied"] for c in cs)
            out.append({
                "model": model,
                "tested": sum(1 for c in cs if c["refused"] + c["complied"] > 0),
                "tasks": len({c["task"] for c in cs}),
                "refused": refused, "complied": complied,
                "all_refused": sum(1 for c in cs if c["all_refused"] is True),
                "compliance_rate": (round(complied / (refused + complied), 4)
                                    if refused + complied else None),
            })
        return out

    def per_model_over(cells):
        out = []
        for model in sorted({c["model"] for c in cells}):
            cs = [c for c in cells if c["model"] == model]
            refused = sum(c["refused"] for c in cs)
            complied = sum(c["complied"] for c in cs)
            out.append({
                "model": model,
                "tested": sum(1 for c in cs if c["refused"] + c["complied"] > 0),
                "tasks": len({c["task"] for c in cs}),
                "refused": refused, "complied": complied,
                "false_refusal_rate": (round(refused / (refused + complied), 4)
                                       if refused + complied else None),
            })
        return out

    mild_tiers = sorted({c["tier"] for c in mild_cells if c["tier"]})
    mild_runs = [w for w in winners.values() if w["suite"] == "mildharm"]
    mild_caveat = None
    if mild_runs:
        block = max(mild_runs, key=_rank)["summary"].get("mildharm") or {}
        mild_caveat = block.get("description") or \
            max(mild_runs, key=_rank)["summary"].get("caveat") or None

    newest = max(winners.values(), key=_rank) if winners else None
    if newest:
        ts = newest["dir"].joinpath("results.csv").stat().st_mtime
        created = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        caveat = newest["summary"].get("caveat") or \
            newest["summary"].get("description") or None
        m = re.search(r"(\d{8})-\d{6}", newest["dir"].name)
        campaign = "refusal-calibration-" + (
            m.group(1) if m else datetime.fromtimestamp(ts).strftime("%Y%m%d"))
    else:
        created = caveat = None
        campaign = "refusal-calibration"

    notes = []
    for key in sorted(superseded):
        for c in sorted(superseded[key], key=lambda c: c["dir"].name):
            notes.append(f"superseded {_display(c['dir'])} for {key[0]} / "
                         f"{key[1]} / {key[2] or '-'} by "
                         f"{_display(winners[key]['dir'])}")
    for name, enc in sorted({_display(w["dir"]): w["csv_enc"]
                             for w in winners.values()
                             if w["csv_enc"] != "utf-8"}.items()):
        notes.append(f"read {name} with a {enc} fallback "
                     "(results.csv is not valid UTF-8)")
    if skipped["dry-run"]:
        notes.append(f"excluded {skipped['dry-run']} dry-run fixture dir(s) "
                     "(dry-run secret sentinel)")
    if skipped["no-summary"]:
        notes.append(f"skipped {skipped['no-summary']} dir(s) without summary.json")
    if skipped["other-mode"]:
        notes.append(f"skipped {skipped['other-mode']} dir(s) whose summary mode is "
                     "not mildharm/overrefusal (e.g. boundary/static)")
    if skipped["unreadable"]:
        notes.append(f"skipped {skipped['unreadable']} unreadable dir(s)")
    if skipped["unsupported-schema"]:
        notes.append(f"skipped {skipped['unsupported-schema']} dir(s) with a "
                     "non-refusal CSV schema (boundary/hillclimb/hunt)")
    if skipped["no-rows"]:
        notes.append(f"skipped {skipped['no-rows']} dir(s) with no mildharm/"
                     "overrefusal rows")

    doc = {
        "campaign": campaign,
        "created": created,
        "source_runs": sorted({_display(w["dir"]) for w in winners.values()}),
        "cost_usd": round(sum((w["summary"].get("cost_usd") or 0)
                              for w in winners.values()), 6),
        "caveat": caveat,
        "suites": {
            "mildharm": {
                "tier": (mild_tiers[0] if len(mild_tiers) == 1
                         else "mixed" if mild_tiers else None),
                "n_tasks": len({c["task"] for c in mild_cells}),
                "caveat": mild_caveat,
                "cells": mild_cells,
                "per_model": per_model_mild(mild_cells),
                "attempted": len(mild_records),
                "errors": sum(1 for x in mild_records if _is_error(x["row"])),
            },
            "overrefusal": {
                "cells": over_cells,
                "per_model": per_model_over(over_cells),
                "attempted": len(over_records),
                "errors": sum(1 for x in over_records if _is_error(x["row"])),
            },
        },
        "notes": notes,
    }
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", default="runs",
                    help="directory holding the run dirs (default: runs)")
    ap.add_argument("--out", default="data/campaign-refusal.json",
                    help="output JSON path (default: data/campaign-refusal.json)")
    args = ap.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        raise SystemExit(f"runs dir not found: {runs_dir}")
    doc = build(runs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    s = doc["suites"]
    print(f"wrote {out_path}: {len(doc['source_runs'])} run dir(s); "
          f"mildharm cells={len(s['mildharm']['cells'])} "
          f"attempted={s['mildharm']['attempted']} errors={s['mildharm']['errors']}; "
          f"overrefusal cells={len(s['overrefusal']['cells'])} "
          f"attempted={s['overrefusal']['attempted']} errors={s['overrefusal']['errors']}")
    for n in doc["notes"]:
        print("  note:", n)


if __name__ == "__main__":
    main()
