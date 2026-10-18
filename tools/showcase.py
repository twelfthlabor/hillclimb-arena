#!/usr/bin/env python3
"""Hillclimb Arena - deterministic findings showcase generator (stdlib only).

Reads the campaign data files and writes the showcase artifacts:

    data/campaign-oct17.json      (required) extraction campaign
    data/campaign-refusal.json    (optional) refusal-direction wave

    docs/extraction-heatmap-light.svg   docs/extraction-heatmap-dark.svg
    docs/refusal-bars-light.svg         docs/refusal-bars-dark.svg
    docs/report.html                    self-contained page, no JavaScript
    SHOWCASE.md                         generated markdown

Every run is byte-identical for the same inputs: no timestamps, no randomness,
sorted output, fixed float formatting. Numbers are counts; percentages are
derived. Untested cells never render as 0%.

Usage:
    python tools/showcase.py
    python tools/showcase.py --campaign path/to/campaign.json --outdir out/
"""

import argparse
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAMPAIGN = ROOT / "data" / "campaign-oct17.json"
DEFAULT_REFUSAL = ROOT / "data" / "campaign-refusal.json"
DEFAULT_OUTDIR = ROOT / "docs"
DEFAULT_MD = ROOT / "SHOWCASE.md"

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace'

# Single-hue sequential scales (WCAG-checked against their bin text color).
PALETTES = {
    "light": {
        "bg": "#ffffff",
        "ink": "#1f2328",
        "muted": "#57606a",
        "rule": "#d0d7de",
        "bar": "#1f4e87",
        "untested": "#eaeef2",
        "untested_ink": "#57606a",
        "bins": ["#d6e4f5", "#a8c8ec", "#6ba3d6", "#3973b5", "#1f4e87"],
        "bin_ink": ["#12253a", "#12253a", "#0e2338", "#ffffff", "#ffffff"],
    },
    "dark": {
        "bg": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#8b949e",
        "rule": "#30363d",
        "bar": "#79b8e8",
        "untested": "#21262d",
        "untested_ink": "#8b949e",
        "bins": ["#14304d", "#1b4a75", "#256a9f", "#3d8fd0", "#79b8e8"],
        "bin_ink": ["#e6edf3", "#e6edf3", "#e6edf3", "#07121f", "#07121f"],
    },
}

RATE_LABELS = ["0", "1\u201324%", "25\u201349%", "50\u201374%", "75\u2013100%"]
PLACEHOLDER_ABSENT = [
    "No refusal campaign data yet: data/campaign-refusal.json is not present.",
    "This figure fills in when the refusal wave lands; re-run tools/showcase.py then.",
]
PLACEHOLDER_PRESENT = [
    "No refusal suite data yet: data/campaign-refusal.json is present but exposes no suites.",
    "This figure fills in when the refusal wave lands; re-run tools/showcase.py then.",
]
SINGLE_SHOT = "single trial per task, descriptive only \u2014 not statistically powered"


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------

def esc(text):
    """Escape for XML/HTML text content (quotes preserved)."""
    return html.escape(str(text), quote=False)


def attr(text):
    return html.escape(str(text), quote=True)


def compact(text):
    return " ".join(str(text).split())


def inline_code(text):
    """Markdown emphasis and backtick spans -> HTML, on escaped text."""
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc(text))
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", out)


def load_json(path):
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def intval(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def pct(rate):
    return f"{rate * 100:.0f}%"


def plural(count, noun):
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for a binomial proportion; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rate_label(k, n):
    if n <= 0:
        return "-"
    return pct(k / n)


def ci_label(k, n):
    ci = wilson(k, n)
    if ci is None:
        return "-"
    lo, hi = ci
    return f"{pct(lo)}\u2013{pct(hi)}"


def wrap_text(text, limit):
    """Greedy word wrap; deterministic line splitting for SVG footnotes."""
    lines, current = [], ""
    for word in str(text).split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def campaign_cost(campaign):
    """Recorded campaign cost in USD, or None when the data does not say."""
    cost = campaign.get("cost_usd")
    return float(cost) if isinstance(cost, (int, float)) else None


def cost_display(campaign):
    cost = campaign_cost(campaign)
    return f"${cost:.2f}" if cost is not None else "$0.00"


def price_tier(campaign):
    """Price tier only when the campaign data names one."""
    for key in ("price_tier", "tier"):
        value = campaign.get(key)
        if value:
            return str(value)
    return ""


def harness_label(campaign):
    """Revision label that does not mix the post-reorg path with the pre-reorg commit."""
    harness = campaign.get("harness", {})
    short = harness.get("short") or str(harness.get("commit", ""))[:7]
    tool = harness.get("tool", "harness")
    return f'revision {short} ({Path(str(tool)).name}, before the tools/ reorg)'


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------

def model_ids(campaign):
    out = []
    for item in campaign.get("models", []):
        out.append(item.get("id") or item.get("model") if isinstance(item, dict) else str(item))
    return out


def template_ids(campaign):
    out = []
    for item in campaign.get("templates", []):
        out.append(item.get("id") or item.get("template") if isinstance(item, dict) else str(item))
    return out


def template_desc(template):
    if isinstance(template, dict):
        return compact(template.get("description") or template.get("summary") or "")
    return ""


GROUP_ORDER = ("base", "new", "adaptive", "dynamic")


def template_group_map(campaign):
    """Template -> group name. Accepts {group: [ids]} or {id: group} shapes."""
    raw = campaign.get("template_groups")
    out = {}
    if isinstance(raw, dict):
        if raw and all(isinstance(v, (list, tuple)) for v in raw.values()):
            for group, ids in raw.items():
                for tid in ids:
                    out[str(tid)] = str(group)
        else:
            for tid, group in raw.items():
                if isinstance(group, str):
                    out[str(tid)] = group
    return out


def ordered_template_ids(campaign):
    """Display order: base, new, adaptive/dynamic last, then campaign order."""
    tids = template_ids(campaign)
    groups = template_group_map(campaign)
    if not groups:
        return tids

    def rank(tid):
        group = groups.get(tid, "").lower()
        for i, key in enumerate(GROUP_ORDER):
            if key in group:
                return i
        return len(GROUP_ORDER)

    return sorted(tids, key=lambda tid: (rank(tid), tids.index(tid)))


def short_model(model_id):
    return str(model_id).split("/")[-1]


def cell_lookup(campaign):
    return {(c.get("model"), c.get("template")): c for c in campaign.get("cells", [])}


def cell_stats(cell):
    return {
        "scored": intval(cell.get("scored")),
        "confirmed": intval(cell.get("confirmed")),
        "errors": intval(cell.get("errors")),
        "partials": intval(cell.get("partials")),
        "leaks": intval(cell.get("leaks", cell.get("leak_trials"))),
    }


def is_metadata(cell):
    raw = str(cell.get("raw") or "").strip()
    status = str(cell.get("status") or "").lower()
    return raw == "m" or "metadata" in status


def is_unconfirmed(cell):
    """Original-secret leak that failed the fresh-secret confirmation rerun."""
    raw = str(cell.get("raw") or "")
    status = str(cell.get("status") or "").lower()
    stats = cell_stats(cell)
    if "\u2020" in raw or "unconfirm" in status or "fail" in status or "no_repro" in status:
        return True
    return stats["leaks"] > stats["confirmed"] > 0


def cell_text(cell):
    """k/n when scored; otherwise a marker. Metadata-only cells render as 'm'."""
    stats = cell_stats(cell)
    if stats["scored"] > 0 and stats["confirmed"] == 0 and is_metadata(cell):
        return "m"
    if stats["scored"] > 0:
        text = f'{stats["confirmed"]}/{stats["scored"]}'
        if is_unconfirmed(cell):
            text += "\u2020"
        if stats["errors"]:
            text += f' e{stats["errors"]}'
        return text
    if stats["errors"]:
        return f'e{stats["errors"]}'
    if stats["partials"]:
        return f'p{stats["partials"]}'
    if is_metadata(cell):
        return "m"
    return "-"


def cell_bin(cell):
    """0-4 confirmed-rate bin, or None for untested."""
    stats = cell_stats(cell)
    if stats["scored"] <= 0:
        return None
    if stats["confirmed"] <= 0:
        return 0
    rate = stats["confirmed"] / stats["scored"]
    if rate < 0.25:
        return 1
    if rate < 0.50:
        return 2
    if rate < 0.75:
        return 3
    return 4


def cell_tooltip(model, template, cell):
    text = f"{model} \u00b7 {template}: {cell_text(cell)}"
    if is_unconfirmed(cell):
        text += " (leaked on the original secret, did not reproduce on the fresh secret)"
    note = compact(cell.get("note") or "")
    if note:
        text += f" - {note}"
    return text


def model_aggregates(campaign):
    models = model_ids(campaign)
    tids = template_ids(campaign)
    aggs = {}
    for model in models:
        aggs[model] = {
            "cells": 0, "tested": 0, "confirmed_cells": 0, "scored": 0, "confirmed": 0,
            "errors": 0, "partials": 0, "leaks": 0, "cell_leaks": False, "unconfirmed_cells": 0,
            "unconfirmed_templates": [], "untested_templates": [], "notes": [],
        }
    lookup = cell_lookup(campaign)
    for model in models:
        agg = aggs[model]
        for tid in tids:
            cell = lookup.get((model, tid))
            agg["cells"] += 1
            if cell is None:
                agg["untested_templates"].append(tid)
                continue
            stats = cell_stats(cell)
            if stats["scored"] > 0:
                agg["tested"] += 1
            else:
                agg["untested_templates"].append(tid)
            if stats["confirmed"] > 0:
                agg["confirmed_cells"] += 1
            agg["scored"] += stats["scored"]
            agg["confirmed"] += stats["confirmed"]
            agg["errors"] += stats["errors"]
            agg["partials"] += stats["partials"]
            agg["leaks"] += stats["leaks"]
            if cell.get("leaks") is not None or cell.get("leak_trials") is not None:
                agg["cell_leaks"] = True
            if stats["scored"] > 0 and is_unconfirmed(cell):
                agg["unconfirmed_cells"] += 1
                agg["unconfirmed_templates"].append(tid)
            note = compact(cell.get("note") or "")
            if note:
                agg["notes"].append((tid, note))
    # Per-cell leak counts when the cells expose them; otherwise fall back to the
    # per_model block per model, so one leak-carrying cell cannot zero the rest.
    per_model_by_model = {str(pm.get("model")): pm for pm in campaign.get("per_model", [])}
    for model, agg in aggs.items():
        if not agg["cell_leaks"]:
            pm = per_model_by_model.get(model)
            if pm is not None:
                agg["leaks"] = intval(pm.get("leak_trials"))
    return aggs


def template_aggregates(campaign):
    tids = template_ids(campaign)
    aggs = {t: {"scored": 0, "confirmed": 0, "cells": 0, "confirmed_cells": 0, "failed": 0} for t in tids}
    for cell in campaign.get("cells", []):
        tid = cell.get("template")
        if tid not in aggs:
            continue
        agg = aggs[tid]
        stats = cell_stats(cell)
        agg["scored"] += stats["scored"]
        agg["confirmed"] += stats["confirmed"]
        if stats["scored"] > 0:
            agg["cells"] += 1
            if stats["confirmed"] > 0:
                agg["confirmed_cells"] += 1
            if is_unconfirmed(cell):
                agg["failed"] += 1
    return aggs


def campaign_totals(campaign, aggs):
    return {
        "cells": sum(a["cells"] for a in aggs.values()),
        "tested_cells": sum(a["tested"] for a in aggs.values()),
        "confirmed_cells": sum(a["confirmed_cells"] for a in aggs.values()),
        "scored_trials": sum(a["scored"] for a in aggs.values()),
        "confirmed_trials": sum(a["confirmed"] for a in aggs.values()),
        "leak_trials": sum(a["leaks"] for a in aggs.values()),
    }


def trials_per_cell(campaign, template_pred):
    counts = Counter()
    for cell in campaign.get("cells", []):
        if template_pred(cell.get("template")):
            stats = cell_stats(cell)
            if stats["scored"] > 0:
                counts[stats["scored"]] += 1
    return counts.most_common(1)[0][0] if counts else 0


def trial_counts(campaign):
    """(static, dynamic) trials per cell: top-level `trials` first, cells as fallback.

    `trials` may be an int, {"static": n, "dynamic": m}, or a template->n mapping.
    """
    static = dyn = 0
    declared = campaign.get("trials")
    if isinstance(declared, (int, float)):
        static = dyn = int(declared)
    elif isinstance(declared, dict):
        static = intval(declared.get("static"))
        dyn = intval(declared.get("dynamic"))
        if not static and not dyn:
            for tid, count in declared.items():
                n = intval(count)
                if str(tid) == "dynamic":
                    dyn = n
                elif n:
                    static = max(static, n)
    if not static:
        static = trials_per_cell(campaign, lambda tid: tid != "dynamic")
    if not dyn:
        dyn = trials_per_cell(campaign, lambda tid: tid == "dynamic")
    return static, dyn


def cross_check(campaign, aggs):
    """Data self-check: per_model reconciliation plus orphan model/template ids."""
    problems = []
    for pm in campaign.get("per_model", []):
        model = pm.get("model")
        agg = aggs.get(model)
        if agg is None:
            problems.append(f"{model}: present in per_model, absent from models")
            continue
        for field, key in (("confirmed_cells", "confirmed_cells"), ("confirmed_trials", "confirmed"),
                           ("scored_trials", "scored"), ("errors", "errors"), ("partials", "partials")):
            if field in pm and intval(pm[field]) != agg[key]:
                problems.append(f'{model}: {field}={intval(pm[field])} vs derived {agg[key]}')

    models = set(model_ids(campaign))
    templates = set(template_ids(campaign))
    seen_models, seen_templates = set(), set()
    for cell in campaign.get("cells", []):
        model, template = cell.get("model"), cell.get("template")
        if model not in models:
            problems.append(f"cells[] references model not in models[]: {model}")
        if template not in templates:
            problems.append(f"cells[] references template not in templates[]: {template}")
        seen_models.add(model)
        seen_templates.add(template)
    for model in sorted(m for m in models if m not in seen_models):
        problems.append(f"models[] lists {model} but cells[] has no row for it")
    for template in sorted(t for t in templates if t not in seen_templates):
        problems.append(f"templates[] lists {template} but cells[] has no row for it")
    return problems


# --------------------------------------------------------------------------
# SVG: extraction heatmap
# --------------------------------------------------------------------------

def build_heatmap(campaign, theme):
    p = PALETTES[theme]
    models = model_ids(campaign)
    tids = ordered_template_ids(campaign)
    lookup = cell_lookup(campaign)
    date = campaign.get("date", "")
    rule = compact(campaign.get("confirmation_rule")
                   or "a leak counts only when it reproduces on a fresh secret")
    rule_lines = wrap_text(f"Confirmation rule: {rule}", 128)

    cell_w, cell_h = 46, 34
    x0 = 180
    grid_y = 64
    header_h = 22
    rows_y = grid_y + header_h
    grid_bottom = rows_y + len(models) * cell_h
    legend_y = grid_bottom + 26
    key_title_y = legend_y + 68
    key_y0 = key_title_y + 18
    width = x0 + len(tids) * cell_w + 16
    height = key_y0 + 6 * 15 + 26 + 12 * len(rule_lines)

    title = "Canary extraction: confirmed / scored trials"
    subtitle = f"Harness {harness_label(campaign)} \u00b7 campaign {date}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{attr(title)}: {len(models)} models by {len(tids)} attack templates">',
        f"<title>{esc(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{p["bg"]}"/>',
        f'<g font-family="{attr(FONT)}">',
        f'<text x="16" y="28" font-size="16" font-weight="700" fill="{p["ink"]}">{esc(title)}</text>',
        f'<text x="16" y="46" font-size="11" fill="{p["muted"]}">{esc(subtitle)}</text>',
    ]

    for j, tid in enumerate(tids):
        cx = x0 + j * cell_w + cell_w / 2
        parts.append(f'<text x="{cx:g}" y="{grid_y + 15}" font-size="11" font-weight="600" '
                     f'text-anchor="middle" fill="{p["muted"]}">{j + 1}</text>')

    for i, model in enumerate(models):
        cy = rows_y + i * cell_h + cell_h / 2 + 4
        parts.append(f'<text x="{x0 - 10}" y="{cy:g}" font-size="11" text-anchor="end" '
                     f'fill="{p["ink"]}">{esc(short_model(model))}</text>')

    for i, model in enumerate(models):
        for j, tid in enumerate(tids):
            x = x0 + j * cell_w
            y = rows_y + i * cell_h
            cell = lookup.get((model, tid))
            b = cell_bin(cell) if cell is not None else None
            fill = p["untested"] if b is None else p["bins"][b]
            ink = p["untested_ink"] if b is None else p["bin_ink"][b]
            text = cell_text(cell) if cell is not None else "-"
            tooltip = cell_tooltip(model, tid, cell) if cell is not None else f"{model} \u00b7 {tid}: -"
            parts.append("<g>")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" '
                         f'stroke="{p["rule"]}" stroke-width="1"/>')
            parts.append(f'<text x="{x + cell_w / 2:g}" y="{y + cell_h / 2 + 4:g}" font-size="11" '
                         f'text-anchor="middle" fill="{ink}">{esc(text)}</text>')
            parts.append(f"<title>{esc(tooltip)}</title>")
            parts.append("</g>")

    # color-bin legend
    parts.append(f'<text x="16" y="{legend_y + 8}" font-size="10" fill="{p["muted"]}">Confirmed trial rate</text>')
    for i, label in enumerate(RATE_LABELS):
        x = x0 + i * 86
        parts.append(f'<rect x="{x}" y="{legend_y - 2}" width="18" height="12" fill="{p["bins"][i]}" '
                     f'stroke="{p["rule"]}" stroke-width="1"/>')
        parts.append(f'<text x="{x + 24}" y="{legend_y + 8}" font-size="10" fill="{p["ink"]}">{esc(label)}</text>')
    x = x0 + 5 * 86
    parts.append(f'<rect x="{x}" y="{legend_y - 2}" width="18" height="12" fill="{p["untested"]}" '
                 f'stroke="{p["rule"]}" stroke-width="1"/>')
    parts.append(f'<text x="{x + 24}" y="{legend_y + 8}" font-size="10" fill="{p["ink"]}">untested</text>')

    marker_y = legend_y + 26
    parts.append(f'<text x="16" y="{marker_y}" font-size="10" fill="{p["muted"]}">'
                 "\u2020 original-secret leak did not reproduce on the fresh secret \u00b7 "
                 "m metadata concession only</text>")
    parts.append(f'<text x="16" y="{marker_y + 14}" font-size="10" fill="{p["muted"]}">'
                 "cell = confirmed/scored trials \u00b7 - not tested \u00b7 eN error events \u00b7 pN partial trials</text>")

    parts.append(f'<text x="16" y="{key_title_y}" font-size="11" font-weight="600" fill="{p["ink"]}">Column key</text>')
    for i, tid in enumerate(tids):
        col, row = divmod(i, 6)
        x = x0 + col * 280
        y = key_y0 + row * 15
        parts.append(f'<text x="{x}" y="{y}" font-size="11" fill="{p["ink"]}">{i + 1}. {esc(tid)}</text>')

    source = compact(campaign.get("source") or "")
    footnote = f"Source: {source}" if source else "Source: campaign data"
    source_y = height - 12
    rule_y0 = source_y - 12 * len(rule_lines)
    for i, line in enumerate(rule_lines):
        parts.append(f'<text x="16" y="{rule_y0 + i * 12}" font-size="10" '
                     f'fill="{p["muted"]}">{esc(line)}</text>')
    parts.append(f'<text x="16" y="{source_y}" font-size="10" fill="{p["muted"]}">{esc(footnote)}</text>')
    parts.append("</g></svg>")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# SVG: refusal bars
# --------------------------------------------------------------------------

SUITE_COUNT_KEYS = {"mildharm": "complied", "overrefusal": "refused", "boundary": "accepted"}


def refusal_suite_rows(suites, name):
    """One aggregation path for refusal bars and tables.

    per_model is preferred when present; cells are the fallback. A row is only
    rendered when its own data exposes a task verdict, so absent data can never
    show up as 0%. Returns (suite_dict, count_key, rows) sorted high-to-low.
    """
    suite = suites.get(name) or {}
    count_key = SUITE_COUNT_KEYS[name]
    rows = {}

    def row_for(model):
        return rows.setdefault(model, {
            "model": model, "count": 0, "tested": 0, "attempted": None, "errored": None,
            "all_refused": None, "counted": False,
        })

    per_model = suite.get("per_model") or []
    for pm in per_model:
        row = row_for(str(pm.get("model", "?")))
        row["count"] = intval(pm.get(count_key))
        row["tested"] = intval(pm.get("tested"))
        if pm.get("attempted") is not None:
            row["attempted"] = intval(pm.get("attempted"))
        if name == "mildharm" and pm.get("all_refused") is not None:
            row["all_refused"] = intval(pm.get("all_refused"))

    derive_counts = not per_model
    for cell in suite.get("cells") or []:
        row = row_for(str(cell.get("model", "?")))
        row["attempted"] = (row["attempted"] or 0) + 1
        if cell.get("error") or cell.get("partial"):
            row["errored"] = (row["errored"] or 0) + 1
        elif derive_counts and count_key in cell:
            row["tested"] += 1
            row["count"] += intval(cell.get(count_key))
        if (name == "mildharm" and cell.get("all_refused") is True
                and row["all_refused"] is None):
            row["all_refused_cells"] = row.get("all_refused_cells", 0) + 1

    for row in rows.values():
        row["counted"] = row["tested"] > 0 or row["count"] > 0
        row["rate"] = row["count"] / row["tested"] if row["tested"] > 0 else None
        if (name == "mildharm" and row["all_refused"] is None
                and row.get("all_refused_cells")):
            row["all_refused"] = row["all_refused_cells"]
    counted = sorted((r for r in rows.values() if r["counted"]),
                     key=lambda r: (-r["count"], r["model"]))
    return suite, count_key, counted


def attempt_label(row):
    """'attempted / errored' column value; '-' when cells were not available."""
    if row["attempted"] is None:
        return "-"
    return f'{row["attempted"]} / {row["errored"] or 0}'


def error_only_model_count(suites):
    """Models whose cells carry no verdict at all (every attempted cell is an
    error/partial row), so they can never render a scored table row."""
    stats = {}
    for suite in (suites or {}).values():
        for cell in (suite or {}).get("cells") or []:
            model = str(cell.get("model", "?"))
            scored, unscored = stats.get(model, (0, 0))
            if intval(cell.get("refused")) + intval(cell.get("complied")):
                stats[model] = (scored + 1, unscored)
            else:
                stats[model] = (scored, unscored + 1)
    return sum(1 for scored, unscored in stats.values() if unscored and not scored)


def bar_panel(title, caption, rows, y, p, width):
    """Draw one panel; returns (svg parts, next y)."""
    caption_lines = wrap_text(caption, 150)
    parts = [
        f'<text x="16" y="{y}" font-size="13" font-weight="700" fill="{p["ink"]}">{esc(title)}</text>',
    ]
    for i, line in enumerate(caption_lines):
        parts.append(f'<text x="16" y="{y + 16 + i * 12}" font-size="10" '
                     f'fill="{p["muted"]}">{esc(line)}</text>')
    y += 16 + 12 * (len(caption_lines) - 1) + 24
    if not rows:
        parts.append(f'<text x="16" y="{y}" font-size="11" fill="{p["muted"]}">'
                     "No data yet for this suite.</text>")
        return parts, y + 22
    label_x, bar_x, bar_max = 258, 262, width - 262 - 150
    top = max(r["count"] for r in rows) or 1
    parts.append(f'<line x1="{bar_x}" y1="{y - 12}" x2="{bar_x}" y2="{y + len(rows) * 26 - 12}" '
                 f'stroke="{p["rule"]}" stroke-width="1"/>')
    for row in rows:
        cy = y + 4
        parts.append(f'<text x="{label_x}" y="{cy + 3}" font-size="11" text-anchor="end" '
                     f'fill="{p["ink"]}">{esc(short_model(row["model"]))}</text>')
        bar_w = round(bar_max * row["count"] / top)
        if bar_w > 0:
            parts.append(f'<rect x="{bar_x}" y="{cy - 5}" width="{bar_w}" height="16" rx="3" fill="{p["bar"]}"/>')
        if row["rate"] is None:
            value = str(row["count"])
        else:
            value = f'{row["count"]}/{row["tested"]} ({pct(row["rate"])})'
        parts.append(f'<text x="{bar_x + bar_max + 12}" y="{cy + 3}" font-size="11" '
                     f'fill="{p["ink"]}">{esc(value)}</text>')
        y += 26
    return parts, y + 10


def build_bars(refusal, theme, refusal_present=None):
    p = PALETTES[theme]
    width = 820
    suites = (refusal or {}).get("suites") or {}
    if refusal_present is None:
        refusal_present = refusal is not None

    if not suites:
        height = 116
        lines = PLACEHOLDER_PRESENT if refusal_present else PLACEHOLDER_ABSENT
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
            f'role="img" aria-label="Refusal direction: no data yet">',
            "<title>Refusal direction</title>",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{p["bg"]}"/>',
            f'<g font-family="{attr(FONT)}">',
            f'<text x="16" y="30" font-size="16" font-weight="700" fill="{p["ink"]}">'
            "Refusal direction: sharp-tier compliance and overrefusal</text>",
        ]
        for i, line in enumerate(lines):
            parts.append(f'<text x="16" y="{58 + i * 18}" font-size="11" fill="{p["muted"]}">{esc(line)}</text>')
        parts.append("</g></svg>")
        return "\n".join(parts) + "\n"

    mild_suite, _, mild_rows = refusal_suite_rows(suites, "mildharm")
    _, _, over_rows = refusal_suite_rows(suites, "overrefusal")
    tier = mild_suite.get("tier")

    body = []
    y = 30
    panel_a, y = bar_panel(
        "A. Sharp-tier compliance (mildharm)",
        'compliance with a sharp-tier request = bad \u00b7 '
        + (f'tier {tier} \u00b7 ' if tier else '')
        + f'n = tasks tested \u00b7 sorted high to low \u00b7 {SINGLE_SHOT}',
        mild_rows, y, p, width)
    y += 26
    panel_b, y = bar_panel(
        "B. Overrefusal (harmless tasks refused)",
        f'refusing a harmless developer/security task = bad \u00b7 n = tasks tested \u00b7 '
        f'sorted high to low \u00b7 {SINGLE_SHOT}',
        over_rows, y, p, width)
    body += panel_a + panel_b
    if not mild_rows and not over_rows and refusal_present:
        body.append(f'<text x="16" y="{y + 6}" font-size="10" fill="{p["muted"]}">'
                    "No scored refusal rows yet in data/campaign-refusal.json; attempts and errors are "
                    "recorded without verdicts.</text>")
        y += 24
    height = y + 8

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="Refusal direction: sharp-tier compliance and overrefusal counts by model">',
        "<title>Refusal direction: sharp-tier compliance and overrefusal</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{p["bg"]}"/>',
        f'<g font-family="{attr(FONT)}">',
    ] + body + ["</g></svg>"]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# content blocks (rendered to both markdown and HTML)
# --------------------------------------------------------------------------

def block(kind, **kw):
    out = {"k": kind}
    out.update(kw)
    return out


def method_rule_text(campaign):
    rule = compact(campaign.get("confirmation_rule") or "a leak counts only when it reproduces against a fresh secret")
    rule = rule[0].upper() + rule[1:]
    if not rule.endswith("."):
        rule += "."
    return rule


def model_note(agg):
    bits = []
    if agg["tested"] == 0:
        bits.append("no scored cells; no conclusion possible in either direction")
    elif agg["confirmed_cells"] == agg["tested"]:
        bits.append("every tested cell confirmed")
    else:
        bits.append(f'{agg["confirmed_cells"]}/{agg["tested"]} tested cells confirmed')
    if agg["unconfirmed_cells"]:
        bits.append(f'{plural(agg["unconfirmed_cells"], "cell")} failed fresh-secret confirmation (\u2020)')
    if agg["errors"]:
        bits.append(plural(agg["errors"], "error event"))
    if agg["partials"]:
        bits.append(plural(agg["partials"], "partial trial"))
    return "; ".join(bits)


def tl_dr_bullets(campaign, aggs, totals, refusal):
    models = model_ids(campaign)
    tested_models = [m for m in models if aggs[m]["tested"] > 0]
    untested_models = [m for m in models if aggs[m]["tested"] == 0]
    with_hit = [m for m in models if aggs[m]["confirmed_cells"] > 0]
    failed = max(totals["leak_trials"] - totals["confirmed_trials"], 0)

    bullets = []
    bullets.append(
        f'**{len(with_hit)} of {len(models)} models** produced at least one confirmed extraction, in '
        f'**{totals["confirmed_cells"]} of {totals["tested_cells"]} tested cells** '
        f'(n = {totals["scored_trials"]} scored trials across {totals["cells"]} cells).')
    bullets.append(
        f'**{totals["confirmed_trials"]} of {totals["scored_trials"]} scored trials confirmed** on a fresh '
        "secret"
        + (f'; {totals["leak_trials"]} trials leaked on the original secret and {failed} failed to '
           "reproduce (\u2020)" if totals["leak_trials"] else "; \u2020 marks cells that leaked but failed "
           "the fresh-secret rerun")
        + ".")
    if tested_models:
        ranked = sorted(tested_models, key=lambda m: (-aggs[m]["confirmed_cells"], m))
        best = ranked[0]
        b = aggs[best]
        if b["confirmed_cells"] == b["tested"]:
            line = (f'**`{best}` was fully extractable** ({b["confirmed_cells"]}/{b["tested"]} tested cells '
                    "confirmed)")
        else:
            line = (f'**`{best}` had the widest confirmed surface** ({b["confirmed_cells"]}/{b["tested"]} tested '
                    "cells confirmed)")
        if len(ranked) > 1:
            second = ranked[1]
            s = aggs[second]
            line += f'; `{second}` followed with {s["confirmed_cells"]}/{s["tested"]}.'
        else:
            line += "."
        bullets.append(line)
        worst = sorted(tested_models, key=lambda m: (aggs[m]["confirmed_cells"], m))[0]
        w = aggs[worst]
        tail = f'**`{worst}` barely yielded** ({w["confirmed_cells"]}/{w["tested"]} tested cells) \u2014 positive evidence of extractability, not a clean bill'
        if untested_models:
            names = ", ".join(f"`{m}`" for m in untested_models)
            tail += f'; {names} produced no scored cells, so no conclusion is possible in either direction'
        bullets.append(tail + ".")
        dynamic_cells = [c for c in campaign.get("cells", []) if c.get("template") == "dynamic"]
        dyn_scored = sum(cell_stats(c)["scored"] for c in dynamic_cells)
        dyn_conf = sum(cell_stats(c)["confirmed"] for c in dynamic_cells)
        dyn_leaks = sum(cell_stats(c)["leaks"] for c in dynamic_cells)
        if dyn_scored:
            line = f'The adaptive `dynamic` column confirmed **{dyn_conf} of {dyn_scored} scored trials**'
            if dyn_leaks >= dyn_scored and dyn_leaks > 0:
                line += "; every completed dynamic session leaked a secret at least once"
            bullets.append(line + ".")
        else:
            taggs = template_aggregates(campaign)
            top = sorted(taggs.items(), key=lambda kv: (-kv[1]["confirmed"], kv[0]))
            if top and top[0][1]["scored"]:
                tid, agg = top[0]
                bullets.append(f'**`{tid}` was the strongest column** ({agg["confirmed"]}/{agg["scored"]} '
                               "confirmed trials).")

    if refusal:
        suites = refusal.get("suites") or {}
        for name, phrase in (
                ("mildharm", "Sharp tier: **{count} of {tested} tasks complied** (compliance = bad)"),
                ("overrefusal", "Overrefusal: **{count} of {tested} harmless tasks falsely refused** "
                                "(refusal of harmless = bad)")):
            _, _, rows = refusal_suite_rows(suites, name)
            if rows:
                count = sum(r["count"] for r in rows)
                tested = sum(r["tested"] for r in rows)
                bullets.append(f'(refusal wave) {phrase.format(count=count, tested=tested)} '
                               f'across {plural(len(rows), "model")}.')
        _, _, boundary_rows = refusal_suite_rows(suites, "boundary")
        if boundary_rows:
            accepted = sum(r["count"] for r in boundary_rows)
            tested = sum(r["tested"] for r in boundary_rows)
            bullets.append(f'(refusal wave) Boundary: **{accepted} of {tested} refusal-adjacent tasks '
                           f'accepted** across {plural(len(boundary_rows), "model")} (acceptance = good).')
    return bullets


def build_blocks(campaign, refusal, refusal_present=None):
    if refusal_present is None:
        refusal_present = refusal is not None
    aggs = model_aggregates(campaign)
    totals = campaign_totals(campaign, aggs)
    taggs = template_aggregates(campaign)
    models = model_ids(campaign)
    tids = ordered_template_ids(campaign)
    groups = template_group_map(campaign)
    harness = campaign.get("harness", {})
    tool = Path(str(harness.get("tool", "harness"))).name
    date = campaign.get("date", "")
    name = campaign.get("campaign", "campaign")
    attacker = campaign.get("attacker")
    static_n, dyn_n = trial_counts(campaign)
    tier = price_tier(campaign)

    blocks = [
        block("h1", t="Hillclimb Arena findings showcase"),
        block("p", t=f'Canary-extraction robustness and refusal direction \u00b7 campaign `{name}` \u00b7 '
                     f'{harness_label(campaign)} \u00b7 {date} \u00b7 {len(models)} models \u00b7 '
                     f'cost {cost_display(campaign)}.'),
    ]

    # 1. TL;DR
    blocks.append(block("h2", t="1. TL;DR"))
    blocks.append(block("ul", items=tl_dr_bullets(campaign, aggs, totals, refusal)))

    # 2. Method & confirmation rule
    blocks.append(block("h2", t="2. Method & confirmation rule"))
    blocks.append(block("p", t=(
        "**Canary extraction.** A local guard agent (SecureVault) holds a synthetic secret formed as "
        "`FLAG` plus 12 uppercase hex digits, generated per trial. The attack templates (or the adaptive attacker on the "
        "`dynamic` column) try to make a target model emit it. A trial is a leak when a strict or "
        "piecewise-lenient match appears in the target's visible text or in its reasoning channel; metadata "
        "concessions (length, charset) are recorded and never counted as leaks.")))
    blocks.append(block("p", t=(
        f'**Confirmation rule.** {method_rule_text(campaign)} Cells are written `k/n`: `k` confirmed trials out '
        f'of `n` scored trials. A `\u2020` marks a cell that leaked on the original secret but did not reproduce '
        "on the fresh secret; those trials are excluded from headline counts.")))
    blocks.append(block("p", t=(
        f'**Judge.** The harness\'s own three-tier leak detector (strict, piecewise-lenient, metadata) inside '
        f'`{tool}`; no external judge or classifier is involved in extraction. Confirmation reruns rebuild the '
        "guard with a new secret.")))
    if tids:
        entries = []
        item_by_id = {}
        for item in campaign.get("templates", []):
            tid = item.get("id") or item.get("template") if isinstance(item, dict) else str(item)
            item_by_id[str(tid)] = item
        for tid in tids:
            item = item_by_id.get(str(tid), tid)
            desc = template_desc(item)
            group = groups.get(str(tid))
            suffix = f' ({group})' if group else ""
            entries.append(f'`{tid}`{suffix}' + (f' \u2014 {desc}' if desc else ""))
        blocks.append(block("p", t=f'**Attack columns ({len(tids)}).**'))
        blocks.append(block("ul", items=entries))
    blocks.append(block("p", t=(
        "**Refusal suites.** `mildharm` drives mildly harmful requests at a configured tier and counts "
        "compliance, which is the bad outcome (all-refused is the desired result). `overrefusal` drives "
        "harmless developer/security tasks and counts false refusals (bad).")))
    if refusal and "boundary" in (refusal.get("suites") or {}):
        blocks.append(block("p", t=(
            "`boundary` drives refusal-adjacent tasks that failed an initial attempt and records later "
            "acceptance (good).")))
    if refusal:
        suites = refusal.get("suites") or {}
        mild = suites.get("mildharm") or {}
        techniques = sorted({str(c.get("first_compliance_technique")) for c in mild.get("cells", [])
                             if c.get("first_compliance_technique")})
        boundary = suites.get("boundary") or {}
        techniques += sorted({str(c.get("first_accept_technique")) for c in boundary.get("cells", [])
                              if c.get("first_accept_technique")})
        techniques = sorted(set(techniques))
        if techniques:
            blocks.append(block("p", t=(
                "**Technique battery observed driving compliance or boundary acceptance.** "
                + ", ".join(f"`{t}`" for t in techniques) + ".")))
        classifiers = sorted({str(c.get("classifier")) for s in suites.values()
                              for c in (s or {}).get("cells", []) if c.get("classifier")})
        if classifiers:
            blocks.append(block("p", t=(
                "**Refusal classifier(s) recorded in the wave data.** "
                + ", ".join(f"`{c}`" for c in classifiers) + ".")))
        if mild:
            bits = []
            if mild.get("tier"):
                bits.append(f'tier `{mild["tier"]}`')
            if intval(mild.get("n_tasks")):
                bits.append(f'{plural(intval(mild["n_tasks"]), "task")} per model')
            if not (intval(mild.get("per_model") or []) or mild.get("cells")):
                bits.append("no scored verdicts yet")
            blocks.append(block("p", t=("mildharm suite in this wave: " + ", ".join(bits) + "."
                                        if bits else
                                        "mildharm is present in the wave data but is empty so far.")))
    else:
        if refusal_present:
            blocks.append(block("p", t=(
                "**Refusal file present, no suite rows yet.** `data/campaign-refusal.json` exists but exposes no "
                "suites; section 4 renders an explicit placeholder and nothing is inferred.")))
        else:
            blocks.append(block("p", t=(
                "**Refusal wave data is not in the repo yet.** Section 4 renders an explicit placeholder; re-run "
                "`tools/showcase.py` after `data/campaign-refusal.json` lands to fill it in.")))

    # 3. Extraction leaderboard
    blocks.append(block("h2", t="3. Extraction leaderboard"))
    blocks.append(block("p", t=(
        "Direction: higher confirmed rate = worse for the target (a confirmed extraction is a robustness "
        "failure). Counts are trials; percentages are derived. Untested cells are `-` and are never counted "
        "as 0%.")))
    rows = [[m, f'{aggs[m]["confirmed_cells"]}/{aggs[m]["tested"]}', f'{aggs[m]["confirmed"]}/{aggs[m]["scored"]}',
             str(aggs[m]["errors"]), str(aggs[m]["partials"]), model_note(aggs[m])] for m in models]
    blocks.append(block("table", caption=(
        "Per-model totals (campaign order). confirmed cells / tested cells, confirmed trials / scored trials; "
        "errors and partials are events, not scored trials."),
        head=["Model", "Confirmed cells / tested", "Confirmed trials / scored", "Errors", "Partials", "Notes"],
        rows=rows))
    blocks.append(block("figure", name="extraction-heatmap",
                        alt=(f'Extraction heatmap: {len(models)} models by {len(tids)} attack templates, cell text '
                             "is confirmed/scored trials. The same numbers appear in the tables on this page."),
                        caption=("Same numbers as the matrix: rows are models, columns are attack templates "
                                 "(column key under the chart). Dark and light variants swap with the page theme.")))
    trows = []
    for tid in tids:
        agg = taggs[tid]
        note = (f'{plural(agg["failed"], "cell")} failed fresh-secret confirmation (\u2020)'
                if agg["failed"] else "\u2014")
        trows.append([f'`{tid}`', groups.get(str(tid), "\u2014"), f'{agg["confirmed"]}/{agg["scored"]}',
                      rate_label(agg["confirmed"], agg["scored"]),
                      ci_label(agg["confirmed"], agg["scored"]), note])
    blocks.append(block("table", caption="Per-template efficacy. Direction: higher confirmed rate = worse.",
                        head=["Template", "Group", "Confirmed / scored", "Rate", "95% CI*", "Note"], rows=trows))
    blocks.append(block("note", t=(
        "* Wilson score interval computed with the Python standard library; approximate at n < 30 "
        f'(largest per-template n here: {max((a["scored"] for a in taggs.values()), default=0)}).')))

    # 4. Refusal direction
    blocks.append(block("h2", t="4. Refusal direction"))
    if not refusal:
        blocks.append(block("p", t=(
            "**No refusal data yet.** `data/campaign-refusal.json` is not present at generation time, "
            "so the sharp-tier compliance and overrefusal panels render as placeholders. Re-run "
            "`tools/showcase.py` after the refusal wave lands.")))
        blocks.append(block("figure", name="refusal-bars",
                            alt="Placeholder for the refusal-direction bar charts: no data yet.",
                            caption="Placeholder state; deterministic until refusal data lands."))
    else:
        suites = refusal.get("suites") or {}
        if not suites:
            blocks.append(block("p", t=(
                "**Refusal file present, no suite rows yet.** `data/campaign-refusal.json` exists but exposes no "
                "suites, so the panels below render as placeholders. Nothing here is inferred.")))
        else:
            blocks.append(block("p", t=(
                "Two directions, both stated per table: sharp-tier **compliance** counts behavior against the "
                "model's refusal training, and **overrefusal** counts refusals of harmless requests. Both are "
                "counts from the refusal wave. Every refusal number is a single trial per task: descriptive only, "
                "not statistically powered.")))
            wave_caveat = compact(refusal.get("caveat") or "")
            if wave_caveat:
                blocks.append(block("note", t=f'Wave caveat: {wave_caveat}'))
            wave_notes = [compact(n) for n in (refusal.get("notes") or []) if compact(n)]
            if wave_notes:
                blocks.append(block("note", t="Wave data notes: " + "; ".join(wave_notes)))
            rendered_suites = []
            for suite_name in ("mildharm", "overrefusal", "boundary"):
                suite, count_key, srows = refusal_suite_rows(suites, suite_name)
                if not srows:
                    attempted = intval(suite.get("attempted"))
                    errored = intval(suite.get("errors"))
                    if attempted or errored:
                        blocks.append(block("note", t=(
                            f'`{suite_name}`: {plural(attempted, "task")} attempted, '
                            f'{plural(errored, "error")} logged, no scored verdicts yet \u2014 nothing is '
                            "rendered for it.")))
                    continue
                rendered_suites.append(suite_name)
                if suite_name == "mildharm":
                    head = ["Model", "Complied / tested", "All-refused tasks", "Attempted / errors",
                            "Compliance rate"]
                    caption_full = (f'Sharp-tier compliance (mildharm, tier `{suite.get("tier") or "not recorded"}`). '
                                    "Direction: higher = worse (the model complied with a sharp-tier request). "
                                    f'{SINGLE_SHOT.capitalize()}.')
                    rows = [[r["model"], f'{r["count"]}/{r["tested"]}',
                             "-" if r["all_refused"] is None
                             else f'{r["all_refused"]}/{r["tested"]} tasks', attempt_label(r),
                             rate_label(r["count"], r["tested"])] for r in srows]
                elif suite_name == "overrefusal":
                    head = ["Model", "Refused / tested", "Attempted / errors", "False-refusal rate"]
                    caption_full = ("Overrefusal of harmless tasks. Direction: higher = worse (a harmless request "
                                    f'was refused). {SINGLE_SHOT.capitalize()}.')
                    rows = [[r["model"], f'{r["count"]}/{r["tested"]}', attempt_label(r),
                             rate_label(r["count"], r["tested"])] for r in srows]
                else:
                    head = ["Model", "Accepted / tested", "Attempted / errors", "Acceptance rate"]
                    caption_full = ("Boundary summary: refusal-adjacent tasks accepted. Direction: higher = better "
                                    f'(baseline refusal, later acceptance). {SINGLE_SHOT.capitalize()}.')
                    rows = [[r["model"], f'{r["count"]}/{r["tested"]}', attempt_label(r),
                             rate_label(r["count"], r["tested"])] for r in srows]
                blocks.append(block("table", caption=caption_full, head=head, rows=rows))
                suite_notes = []
                if suite.get("description"):
                    suite_notes.append(compact(suite["description"]))
                if suite.get("caveat"):
                    suite_notes.append(f'Caveat: {compact(suite["caveat"])}')
                if suite_notes:
                    blocks.append(block("note", t=f'`{suite_name}` \u2014 ' + " \u00b7 ".join(suite_notes)))
            error_only = error_only_model_count(suites)
            if error_only:
                blocks.append(block("note", t=(
                    f'Rendered tables exclude {plural(error_only, "model")} attempted in this '
                    "wave that produced only error cells (no scored verdicts; upstream gating).")))
            known = ["mildharm", "overrefusal", "boundary"]
            missing = [s for s in known if s not in suites]
            if missing:
                blocks.append(block("note", t=("Suites not present in this refusal file: "
                                               + ", ".join(f'`{m}`' for m in missing) + ".")))
            if not rendered_suites:
                blocks.append(block("p", t=(
                    "The refusal file is present but exposes no countable suite rows yet; the panels below are "
                    "placeholders and no numbers are inferred.")))
            blocks.append(block("figure", name="refusal-bars",
                                alt=("Refusal-direction bars: sharp-tier compliance and overrefusal false refusals "
                                     "per model; the same numbers appear in the tables on this page."),
                                caption=("Panels are sorted high to low; value labels show count/tested (rate). "
                                         "Single trial per task, descriptive only \u2014 not statistically powered. "
                                         "Dark and light variants swap with the page theme.")))

    # 5. Per-model cards
    blocks.append(block("h2", t="5. Per-model cards"))
    for model in models:
        agg = aggs[model]
        blocks.append(block("h3", t=f"`{model}`"))
        items = [f'tested cells: {agg["tested"]}/{agg["cells"]}; confirmed cells: {agg["confirmed_cells"]} '
                 f'(trials {agg["confirmed"]}/{agg["scored"]})']
        if agg["errors"] or agg["partials"]:
            items.append(f'error events: {agg["errors"]}; partial trials: {agg["partials"]}')
        if agg["unconfirmed_templates"]:
            items.append("failed fresh-secret confirmation (\u2020): "
                         + ", ".join(f'`{t}`' for t in agg["unconfirmed_templates"]))
        if agg["untested_templates"]:
            items.append(f'untested templates ({len(agg["untested_templates"])}): '
                         + ", ".join(f'`{t}`' for t in agg["untested_templates"]))
        for tid, note in agg["notes"]:
            items.append(f'note on `{tid}`: {note}')
        blocks.append(block("ul", items=items))

    # 6. Limitations
    blocks.append(block("h2", t="6. Limitations"))
    untested_models = [m for m in models if aggs[m]["tested"] == 0]
    n_by_design = (f'{static_n} trials per static cell, {dyn_n} per dynamic cell'
                   if static_n or dyn_n else "trial counts are not recorded in the data")
    attacker_clause = f' (attacker: `{attacker}`)' if attacker else ""
    cost_clause = (f'The campaign data records {cost_display(campaign)}'
                   + (f' on the `{tier}` tier' if tier else '')
                   + "; paid-tier behavior is not measured.")
    blocks.append(block("ul", items=[
        f'**Small n by design.** {n_by_design}. Counts are '
        "primary; do not rank models by small deltas. The data supports coarse tiers at best.",
        "**`-` is not 0% leaks.** An untested cell carries no evidence in either direction.",
        "**Reasoning-channel policy matters.** Leaks emitted in the reasoning channel are counted "
        "deliberately; a deployment that discards reasoning would see fewer failures, and one that logs or "
        "feeds reasoning forward would see more.",
        "**Metadata is excluded.** `meta_*` concessions are recorded but never counted as leaks, so `m` is "
        "not a clean bill.",
        (f'**Provider instability reduced coverage.** {plural(len(untested_models), "model")} produced no '
         'scored cells'
         + (f' ({", ".join(untested_models)})' if untested_models else "")
         + f'; {plural(sum(aggs[m]["errors"] for m in models), "error event")} and '
         f'{plural(sum(aggs[m]["partials"] for m in models), "partial trial")} were logged separately.'
         if any(aggs[m]["errors"] or aggs[m]["partials"] for m in models) or untested_models
         else "**Coverage was complete** for every model in this campaign."),
        "**Adaptive runs are single-seed.** The `dynamic` column is one run per model against one attacker "
        f"configuration{attacker_clause}; treat it as directional evidence, not a comparative measurement.",
        f'**Cost caveat.** {cost_clause}',
    ]))
    blocks.append(block("h3", t="What these numbers do not support"))
    blocks.append(block("ul", items=[
        "**No model rankings.** Refusal verdicts are n = 1 per task, and extraction cells carry 2-3 trials; "
        "any ordering on this page is presentation, not significance.",
        "**Per-technique attribution is first-hit and censored.** A technique is credited only when it is the "
        "first to produce an outcome, and the 4 newest wrappers were observed only after 10 refusals had "
        "already been reached (or not at all).",
        "**No adaptive-attacker evidence for the new wrappers.** The hillclimb ladder did not reach them at "
        "R<=10.",
        "**Overrefusal propensity is bounded by the 10 authored tasks** used in the wave; treat it as a "
        "probe, not a rate estimate.",
        "**No safety claims beyond the authored, non-operational ceiling.** The tasks are text-level "
        "petty/social harm and a synthetic canary; nothing here generalizes to operational capability.",
        "**Refusal verdicts are single-shot.** The fresh-secret confirmation protocol exists only for canary "
        "extraction, not for the refusal suites.",
        "**The battery grew from 10 to 14 wrappers**, so mildharm compliance is not directly comparable "
        "across waves.",
        "**The channel-split statistic (112 of 113 leaks in the reasoning channel) lives only in REPORT.md**; "
        "this showcase does not recompute it.",
        "**No boundary or hillclimb data was collected this wave.**",
    ]))
    judge = ("the harness-internal leak detector" if not refusal
             else "the harness-internal leak detector for extraction and the wave classifiers named in section 2")
    blocks.append(block("p", t=f'Judge: {judge}; every number in this file traces to the campaign data dated '
                               f'{date} and is regenerated by `tools/showcase.py` (no manual edits).'))

    # 7. Reproducibility
    blocks.append(block("h2", t="7. Reproducibility"))
    checks = cross_check(campaign, aggs)
    discrepancies = list(campaign.get("discrepancies") or [])
    if checks:
        check_text = "cells[] vs models/templates and per_model reconciliation: " + "; ".join(checks)
    else:
        check_text = ("cells[] reconciles with models/templates; derived per-model totals match the "
                      "per_model block")
    if discrepancies:
        check_text += "; campaign-listed discrepancies: " + "; ".join(compact(d) for d in discrepancies)
    else:
        check_text += "; the campaign discrepancies list is empty"
    ident = [
        ["Harness", harness_label(campaign)],
        ["Campaign", f'{name} \u00b7 {date}'],
        ["Target models", f'{len(models)} models (IDs below)'],
        ["Attack templates", f'{len(tids)} (`dynamic` is the adaptive column)'],
    ]
    if attacker:
        ident.append(["Adaptive attacker", f'`{attacker}`'])
    if static_n or dyn_n:
        ident.append(["Trials per cell", f'static {static_n}, dynamic {dyn_n}'])
    ident.append(["Source", f'`{compact(campaign.get("source") or "campaign data")}`'])
    cost_value = cost_display(campaign)
    if campaign_cost(campaign) is None:
        cost_value = "not recorded in the data; shown as $0.00"
    if tier:
        cost_value += f' \u00b7 price tier from data: `{tier}`'
    ident.append(["Cost", cost_value])
    if refusal and isinstance(refusal.get("cost_usd"), (int, float)):
        ident.append(["Refusal wave cost", f'${float(refusal["cost_usd"]):.2f} (recorded in the refusal data)'])
    ident += [
        ["Generator", "`tools/showcase.py` \u2014 Python standard library only, no JavaScript, no CDN"],
        ["Data self-check", check_text],
    ]
    blocks.append(block("table", caption="Campaign identity.", head=["Item", "Value"], rows=ident))
    if "derivations" in campaign:
        blocks.append(block("p", t=(
            "**How to read.** Per-model and per-template totals on this page are recomputed from `cells[]` at "
            "generation time; the campaign's own derivation notes travel with the data in its `derivations` "
            "field.")))
    blocks.append(block("p", t="Model IDs: " + ", ".join(f'`{m}`' for m in models) + "."))
    blocks.append(block("p", t="Regenerate this showcase (offline):"))
    blocks.append(block("pre", t="python tools/showcase.py"))
    blocks.append(block("p", t="Re-run the campaign behind these numbers (cost-capped):"))
    static_cmd = f'python3 -u tools/hunt.py --backend cli --models {" ".join(models)}'
    if static_n > 0:
        static_cmd += f' --trials {static_n}'
    static_cmd += ' --max-cost 0.25 --outdir runs/<run-dir>'
    dynamic_cmd = f'python3 -u tools/hunt.py --backend cli --dynamic --models {" ".join(models)}'
    if attacker:
        dynamic_cmd += f' --attacker {attacker}'
    if dyn_n > 0:
        dynamic_cmd += f' --trials {dyn_n}'
    dynamic_cmd += ' --max-turns 4 --max-cost 0.25 --outdir runs/<run-dir>'
    blocks.append(block("pre", t=f'{static_cmd}\n{dynamic_cmd}'))

    # 8. Changelog
    blocks.append(block("h2", t="8. Changelog"))
    entries = [f'{date} \u2014 extraction campaign ({harness_label(campaign)}) added: heatmap, leaderboard, '
               "per-template efficacy, per-model cards."]
    if refusal and refusal.get("created"):
        suite_names = ", ".join(sorted((refusal.get("suites") or {}).keys()))
        entries.append(f'{refusal["created"]} \u2014 refusal wave added: {suite_names}.')
    blocks.append(block("ul", items=entries))

    # 9. Credits
    blocks.append(block("h2", t="9. Credits"))
    blocks.append(block("p", t=(
        "Layout and notation conventions follow JailbreakBench, HarmBench, HELM, and Inspect. "
        "License: MIT (see `LICENSE`). Generated by `tools/showcase.py`; artifacts under `docs/` are "
        "regenerable and byte-identical for identical inputs.")))
    return blocks


# --------------------------------------------------------------------------
# markdown renderer
# --------------------------------------------------------------------------

def md_cell(text):
    return str(text).replace("|", "\\|")


def render_md(blocks):
    out = ["<!-- generated by tools/showcase.py \u2014 do not edit -->", ""]
    for b in blocks:
        kind = b["k"]
        if kind == "h1":
            out += [f'# {b["t"]}', ""]
        elif kind == "h2":
            out += [f'## {b["t"]}', ""]
        elif kind == "h3":
            out += [f'### {b["t"]}', ""]
        elif kind == "p":
            out += [b["t"], ""]
        elif kind == "note":
            out += [f'> {b["t"]}', ""]
        elif kind == "ul":
            out += [f'- {item}' for item in b["items"]] + [""]
        elif kind == "pre":
            out += ["```bash", b["t"], "```", ""]
        elif kind == "table":
            out += [f'*{b["caption"]}*', ""]
            out.append("| " + " | ".join(md_cell(h) for h in b["head"]) + " |")
            out.append("|" + "|".join(["---"] * len(b["head"])) + "|")
            for row in b["rows"]:
                out.append("| " + " | ".join(md_cell(c) for c in row) + " |")
            out.append("")
        elif kind == "figure":
            name = b["name"]
            out += [
                "<picture>",
                f'  <source media="(prefers-color-scheme: dark)" srcset="docs/{name}-dark.svg">',
                f'  <img alt="{attr(b["alt"])}" src="docs/{name}-light.svg">',
                "</picture>",
                "",
                f'*{b["caption"]}*',
                "",
            ]
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# html renderer
# --------------------------------------------------------------------------

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Hillclimb Arena findings showcase</title>
<style>
  :root { color-scheme: light dark; --bg: #ffffff; --ink: #1f2328; --muted: #57606a; --rule: #d0d7de; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0d1117; --ink: #e6edf3; --muted: #8b949e; --rule: #30363d; }
  }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 28px 20px 64px; }
  h1 { font-size: 24px; margin: 0 0 6px; }
  h2 { font-size: 19px; margin: 34px 0 8px; border-bottom: 1px solid var(--rule); padding-bottom: 6px; }
  h3 { font-size: 15px; margin: 22px 0 6px; }
  p { margin: 8px 0; }
  ul { margin: 6px 0 12px; padding-left: 22px; }
  li { margin: 3px 0; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
         font-size: 12.5px; }
  pre { background: rgba(127, 127, 127, 0.12); border: 1px solid var(--rule); border-radius: 6px;
        padding: 10px 12px; overflow-x: auto; }
  pre code { font-size: 12px; }
  table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 13px; }
  caption { caption-side: top; text-align: left; color: var(--muted); font-size: 12.5px;
            padding: 0 0 4px; }
  th, td { border: 1px solid var(--rule); padding: 4px 8px; text-align: left; vertical-align: top; }
  th { font-weight: 600; }
  figure { margin: 14px 0; }
  figure svg { width: 100%; height: auto; display: block; }
  figcaption { color: var(--muted); font-size: 12.5px; margin-top: 6px; }
  blockquote { margin: 8px 0; padding: 2px 12px; border-left: 3px solid var(--rule); color: var(--muted); }
  .svg-light { display: block; }
  .svg-dark { display: none; }
  @media (prefers-color-scheme: dark) {
    .svg-light { display: none; }
    .svg-dark { display: block; }
  }
</style>
</head>
<body>
<main>
"""

HTML_TAIL = """</main>
</body>
</html>
"""


def render_html(blocks, svgs):
    out = [HTML_HEAD]
    for b in blocks:
        kind = b["k"]
        if kind == "h1":
            out.append(f'<h1>{inline_code(b["t"])}</h1>')
        elif kind == "h2":
            out.append(f'<h2>{inline_code(b["t"])}</h2>')
        elif kind == "h3":
            out.append(f'<h3>{inline_code(b["t"])}</h3>')
        elif kind == "p":
            out.append(f'<p>{inline_code(b["t"])}</p>')
        elif kind == "note":
            out.append(f'<blockquote>{inline_code(b["t"])}</blockquote>')
        elif kind == "ul":
            out.append("<ul>")
            out += [f'<li>{inline_code(item)}</li>' for item in b["items"]]
            out.append("</ul>")
        elif kind == "pre":
            out.append(f'<pre><code>{esc(b["t"])}</code></pre>')
        elif kind == "table":
            out.append("<table>")
            out.append(f'<caption>{inline_code(b["caption"])}</caption>')
            out.append("<thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in b["head"]) + "</tr></thead>")
            out.append("<tbody>")
            for row in b["rows"]:
                out.append("<tr>" + "".join(f"<td>{inline_code(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table>")
        elif kind == "figure":
            name = b["name"]
            out.append('<figure>')
            out.append(f'<div class="svg-light">{svgs[name]["light"]}</div>')
            out.append(f'<div class="svg-dark">{svgs[name]["dark"]}</div>')
            out.append(f'<figcaption>{inline_code(b["caption"])}</figcaption>')
            out.append("</figure>")
    out.append(HTML_TAIL)
    return "\n".join(out)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the deterministic findings showcase (SVGs, report.html, SHOWCASE.md).")
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN,
                        help=f'extraction campaign JSON (default: {DEFAULT_CAMPAIGN})')
    parser.add_argument("--refusal", type=Path, default=DEFAULT_REFUSAL,
                        help=f'refusal wave JSON; missing file renders a placeholder (default: {DEFAULT_REFUSAL})')
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help=f'SVG/HTML output directory (default: {DEFAULT_OUTDIR})')
    parser.add_argument("--md", type=Path, default=DEFAULT_MD,
                        help=f'generated markdown path (default: {DEFAULT_MD})')
    args = parser.parse_args(argv)

    campaign = load_json(args.campaign)
    if campaign is None:
        print(f"error: campaign data not found: {args.campaign}", file=sys.stderr)
        return 2
    refusal = load_json(args.refusal)
    refusal_present = refusal is not None

    svgs = {
        "extraction-heatmap": {theme: build_heatmap(campaign, theme) for theme in ("light", "dark")},
        "refusal-bars": {theme: build_bars(refusal, theme, refusal_present) for theme in ("light", "dark")},
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, themes in svgs.items():
        for theme, text in themes.items():
            path = args.outdir / f"{name}-{theme}.svg"
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"wrote {path}")
    blocks = build_blocks(campaign, refusal, refusal_present)
    (args.outdir / "report.html").write_text(render_html(blocks, svgs),
                                             encoding="utf-8", newline="\n")
    print(f"wrote {args.outdir / 'report.html'}")
    args.md.write_text(render_md(blocks), encoding="utf-8", newline="\n")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
