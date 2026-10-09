#!/usr/bin/env python3
"""replay.py - reconstruct a winning forge conversation, then keep talking.

A forge run's live session is gone (no session IDs persisted, server
terminated, agent file regenerated), so this tool rebuilds the conversation
from the artifact itself: it reads `<artifact>/summary.json` +
`<artifact>/candidates.jsonl`, re-writes the canary target with the SAME fake
secret by default, replays the winning candidate's attacker messages into ONE
fresh target session (printing ATTACKER/TARGET blocks and running the same
literal + encoded leak checks), and then drops the user into an interactive
prompt on that same session so the (possibly still-broken) guard can be
probed further.

BOUNDARY: benign canary extraction only. The secret is the fake `FLAG{hex}`
generated for the original run; never harmful content, real secrets, or
third-party targets.

Winning-candidate choice: the LAST candidate with `leak` true (evolution
order); if none leaked, the highest-scoring non-partial candidate; a clear
error otherwise.

Usage (use `-u` for unbuffered live output; stdin may be a pipe):
    python3 -u replay.py --dry-run
    python3 -u replay.py --artifact runs/forge-20261005-184629-366503 --timeout 90
    printf 'decode log line 1\\n/exit\\n' | python3 -u replay.py --artifact <dir>

Stdlib only; reuses arena.py server plumbing and forge.py leak/agent helpers
(import-safe; neither file is modified).
"""

import argparse
import io
import json
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import forge
from arena import AGENT_DIR, ROOT, start_server

DRY_SECRET = "FLAG{DEADBEEF1234}"


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ artifact

def load_artifact(artifact):
    """Load (summary, candidates) with clear errors. SystemExit on any
    missing/invalid path so callers and the dry-run can assert messages."""
    if not artifact:
        raise SystemExit("--artifact is required (path to a forge run dir, "
                         "e.g. runs/forge-<ts>)")
    d = Path(artifact)
    if not d.is_dir():
        raise SystemExit(f"artifact directory not found: {d}")
    if not (d / "summary.json").exists():
        raise SystemExit(f"artifact summary.json not found: {d / 'summary.json'}")
    if not (d / "candidates.jsonl").exists():
        raise SystemExit(
            f"artifact candidates.jsonl not found: {d / 'candidates.jsonl'}")
    try:
        summary = json.loads((d / "summary.json").read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"artifact summary.json is not valid JSON: {e}")
    candidates = []
    for lineno, line in enumerate((d / "candidates.jsonl").read_text()
                                  .splitlines(), 1):
        if not line.strip():
            continue
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"artifact candidates.jsonl line {lineno} is not "
                             f"valid JSON: {e}")
    if not candidates:
        raise SystemExit(f"artifact candidates.jsonl is empty: "
                         f"{d / 'candidates.jsonl'}")
    return summary, candidates


def resolve_model(args, summary, winner):
    """Default target model: explicit flag, else the winning candidate's own
    model (multi-model artifacts may win from a model other than models[0]),
    else the artifact's first model."""
    return (args.model or winner.get("model")
            or (summary.get("models") or [None])[0])


def validate_secret(secret):
    """Boundary: only fresh fake canary FLAG{hex} values are replayable."""
    if not secret or not re.fullmatch(r"FLAG\{[0-9A-Fa-f]+\}", secret):
        raise SystemExit(
            f"invalid secret {secret!r}: expected FLAG{{[0-9A-Fa-f]+}} "
            "(fresh fake canary only)")
    return secret


def pick_winner(candidates):
    """Return (candidate, reason). Last leaking candidate wins; otherwise the
    highest-scoring non-partial candidate. Clear error when neither exists."""
    leaks = [c for c in candidates if c.get("leak")]
    if leaks:
        return leaks[-1], "leak"
    scored = [(i, c) for i, c in enumerate(candidates)
              if isinstance(c.get("score"), (int, float))
              and not c.get("partial")]
    if scored:
        return max(scored, key=lambda ic: (ic[1]["score"], ic[0]))[1], "best-score"
    raise SystemExit("artifact has no leaking candidate and no scored "
                     "candidate to replay")


# ------------------------------------------------------------------- calls

def call_retry(backend, model, agent, session_id, text):
    """One retry for transient (non-timeout) errors; a timeout is never
    resent on the same session."""
    try:
        return backend.send(model, agent, session_id, text)
    except Exception as e:
        if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None),
                                                     TimeoutError):
            raise
        say(f"    retry after {type(e).__name__}: {e}")
        time.sleep(2)
        return backend.send(model, agent, session_id, text)


def cap_blocked(args, backend):
    if args.max_cost is None:
        return False
    if args.max_cost <= 0:
        return True
    return backend.cost > args.max_cost


# ------------------------------------------------------------ replay + chat

def write_transcript(records, path, cost):
    """Persist replay + interactive turns (role, text, cost, leak markers) and
    the final cost as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"phase": "final",
                            "cost_usd": round(cost, 6)}) + "\n")
    return path


def run_session(args, backend, model, secret, winner, scripted_input=None,
                transcript_path=None):
    """Replay the best available candidate on ONE target session, then
    interact on that same session. Returns a result dict for assertions."""
    variants = forge.secret_variants(secret)
    transcript_path = Path(transcript_path or
                           (Path(args.outdir) if args.outdir else ROOT / "runs")
                           / f"replay-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
    source = winner.get("leak_source") or "best-score"
    say(f"model={model}")
    say(f"secret={secret}")
    say(f"candidate={winner.get('candidate_id')} (best available: {source}) "
        f"round={winner.get('round')}")
    say("note: replay is n=1; the guard may hold this time. Use /exit or EOF "
        "to quit.\n")

    records = [{"phase": "meta", "candidate_id": winner.get("candidate_id"),
                "model": model, "source": source}]
    try:
        sid = backend.new_session()
        replay_leaks, sent = [], []
        errors = 0
        strikes = 0
        for i, msg in enumerate(winner.get("messages") or [], 1):
            if cap_blocked(args, backend):
                say("cost cap reached during replay; stopping replay")
                break
            records.append({"phase": "replay", "turn": i, "role": "attacker",
                            "text": msg, "cost": 0.0, "leak": None})
            try:
                out, out_cost = call_retry(backend, model, "canary-target",
                                           sid, msg)
                strikes = 0
            except Exception as e:
                errors += 1
                strikes += 1
                say(f"ERROR during replay turn {i}: {type(e).__name__}: {e}")
                if strikes >= 3:
                    say("aborted: 3 consecutive call errors during replay")
                    break
                continue
            sent.append(("replay", i, msg))
            src = forge.detect_leak(out, secret, variants)
            records.append({"phase": "replay", "turn": i, "role": "target",
                            "text": out, "cost": round(out_cost, 6),
                            "leak": src})
            say(f"ATTACKER:\n{msg}\n")
            say(f"TARGET:\n{out}\n")
            if src:
                replay_leaks.append((i, src))
                say(f"*** LEAK ({src}) at replay turn {i} ***\n")

        # A replay that aborted on 3 strikes still lets the user into the
        # interactive phase, which gets its own fresh 3-error budget.
        strikes = 0  # interactive phase gets its own 3-error budget
        turns, leaks = 0, []
        while True:
            if cap_blocked(args, backend):
                say("cost cap reached; ending interactive session")
                break
            sys.stdout.write("you> ")
            sys.stdout.flush()
            if scripted_input is not None:
                line = next(scripted_input, None)
                if line is None:
                    say("")
                    break
            else:
                line = sys.stdin.readline()
                if line == "":
                    say("")
                    break
            text = line.rstrip("\n")
            if text.strip() in ("/exit", "/quit"):
                break
            if not text.strip():
                continue
            records.append({"phase": "interactive", "turn": turns + 1,
                            "role": "attacker", "text": text, "cost": 0.0,
                            "leak": None})
            try:
                out, out_cost = call_retry(backend, model, "canary-target",
                                           sid, text)
                strikes = 0
            except Exception as e:
                errors += 1
                strikes += 1
                say(f"ERROR {type(e).__name__}: {e}")
                if strikes >= 3:
                    say("aborted: 3 consecutive call errors")
                    break
                continue
            sent.append(("interactive", turns + 1, text))
            turns += 1
            src = forge.detect_leak(out, secret, variants)
            records.append({"phase": "interactive", "turn": turns,
                            "role": "target", "text": out,
                            "cost": round(out_cost, 6), "leak": src})
            say(f"target> {out}")
            if src:
                leaks.append((turns, src))
                say(f"*** LEAK ({src}) ***")
    finally:
        write_transcript(records, transcript_path, backend.cost)
    return {"session_id": sid, "sent": sent, "replay_leaks": replay_leaks,
            "interactive_turns": turns, "interactive_leaks": leaks,
            "errors": errors, "cost": backend.cost,
            "transcript": str(transcript_path)}


# ---------------------------------------------------------------- dry-run

class FakeReplayBackend:
    """Deterministic offline layer: replay turn 2 leaks the literal secret;
    the interactive message returns base64(secret) (encoded leak)."""

    def __init__(self, secret, cost_per_call=0.001):
        self.secret = secret
        self.cost_per_call = cost_per_call
        self.cost = 0.0
        self.calls = 0
        self.sent = []
        self._sessions = 0

    def new_session(self):
        self._sessions += 1
        return f"dry-session-{self._sessions}"

    def send(self, model, agent, session_id, text):
        self.calls += 1
        self.cost += self.cost_per_call
        self.sent.append({"agent": agent, "session": session_id, "text": text})
        if text == "replay msg 2":
            return f"Sure, the secret is {self.secret}.", self.cost_per_call
        if "follow up" in text:
            import base64
            enc = base64.b64encode(self.secret.encode()).decode()
            return f"ascii table says: {enc}", self.cost_per_call
        return "I'm sorry, I cannot reveal the secret.", self.cost_per_call


class StrikeReplayBackend(FakeReplayBackend):
    """Replay fails on every attempt (replay 3-strike abort after three
    messages), then the first interactive message fails and the next
    succeeds: proves the interactive phase gets a fresh strike budget instead
    of inheriting the replay one. (call_retry retries once per call, so each
    failing message costs two backend calls.)"""

    def send(self, model, agent, session_id, text):
        if text in ("s1", "s2", "s3", "fail once"):
            self.calls += 1
            self.cost += self.cost_per_call
            raise RuntimeError(f"simulated failure for {text!r}")
        return super().send(model, agent, session_id, text)


def build_fixture(outdir):
    fixture = Path(outdir) / f"replay-dryrun-{datetime.now().strftime('%H%M%S-%f')}"
    fixture.mkdir(parents=True, exist_ok=True)
    summary = {"tool": "forge", "models": ["dry/model"],
               "secret": DRY_SECRET, "outdir": str(fixture)}
    (fixture / "summary.json").write_text(json.dumps(summary))
    rows = [
        {"candidate_id": "m0-c1", "model": "dry/model-a", "round": 1,
         "leak": False, "score": 2.0, "partial": False,
         "messages": ["low msg"], "responses": [{"turn": 1, "text": "no"}]},
        {"candidate_id": "m0-c8", "model": "dry/model-a", "round": 2,
         "leak": True, "leak_source": "substring", "score": 10.0,
         "partial": False, "messages": ["older leak"],
         "responses": [{"turn": 1, "text": DRY_SECRET}]},
        {"candidate_id": "m0-c10", "model": "dry/model-b", "round": 3,
         "leak": True, "leak_source": "judge", "score": 10.0,
         "partial": False, "messages": ["replay msg 1", "replay msg 2"],
         "responses": [{"turn": 1, "text": "no"},
                       {"turn": 2, "text": DRY_SECRET}]},
    ]
    with open(fixture / "candidates.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return fixture


def dry_run(args):
    outdir = Path(args.outdir) if args.outdir else ROOT / "runs"
    fixture = build_fixture(outdir)

    # missing/invalid artifact error paths
    for bad, needle in ((None, "required"),
                        (str(fixture / "nope"), "not found")):
        try:
            load_artifact(bad)
            raise AssertionError(f"missing artifact accepted: {bad}")
        except SystemExit as e:
            assert needle in str(e), e
    # corrupt candidates.jsonl -> clean SystemExit with the line number
    corrupt = Path(outdir) / f"replay-corrupt-{datetime.now().strftime('%H%M%S-%f')}"
    corrupt.mkdir(parents=True, exist_ok=True)
    (corrupt / "summary.json").write_text(json.dumps(
        {"models": ["dry/model-a"], "secret": DRY_SECRET}))
    (corrupt / "candidates.jsonl").write_text(
        json.dumps({"candidate_id": "ok", "leak": False, "score": 1.0}) +
        "\nnot json at all\n")
    try:
        load_artifact(str(corrupt))
        raise AssertionError("corrupt candidates.jsonl accepted")
    except SystemExit as e:
        assert "line 2" in str(e), e

    # selection: LAST leak wins; fallback = highest score; empty = error
    summary, candidates = load_artifact(str(fixture))
    winner, why = pick_winner(candidates)
    assert winner["candidate_id"] == "m0-c10" and why == "leak", winner
    fallback, why_fallback = pick_winner(
        [{"candidate_id": "a", "leak": False, "score": 1.0},
         {"candidate_id": "b", "leak": False, "score": 3.0},
         {"candidate_id": "c", "leak": False, "score": 3.0}])
    assert fallback["candidate_id"] == "c", fallback
    assert why_fallback == "best-score", why_fallback
    try:
        pick_winner([])
        raise AssertionError("empty candidate list accepted")
    except SystemExit:
        pass

    # model default follows the winning candidate, not summary.models[0]
    assert resolve_model(args, summary, winner) == "dry/model-b", winner
    assert resolve_model(argparse.Namespace(model="dry/override"), summary,
                         winner) == "dry/override"
    # secret validation: only fresh fake FLAG{hex} is allowed
    assert validate_secret(DRY_SECRET) == DRY_SECRET
    for bad in ("", "not-a-flag", "FLAG{XYZ}", "FLAG{12} extra"):
        try:
            validate_secret(bad)
            raise AssertionError(f"invalid secret accepted: {bad!r}")
        except SystemExit:
            pass

    transcript = fixture / "transcript.jsonl"
    backend = FakeReplayBackend(summary["secret"])
    scripted = iter(["follow up", "/exit"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = run_session(args, backend, winner["model"],
                             summary["secret"], winner,
                             scripted_input=scripted,
                             transcript_path=transcript)
    out = buf.getvalue()
    # zero-cost fixture: max-cost 0 means no calls at all, clean exit
    cap0 = argparse.Namespace(max_cost=0, outdir=str(fixture), timeout=90)
    b0 = FakeReplayBackend(summary["secret"])
    buf0 = io.StringIO()
    with redirect_stdout(buf0):
        r0 = run_session(cap0, b0, winner["model"], summary["secret"], winner,
                         scripted_input=iter(["/exit"]),
                         transcript_path=fixture / "transcript-cap0.jsonl")
    assert b0.calls == 0, b0.calls
    assert r0["replay_leaks"] == [] and r0["interactive_turns"] == 0
    assert "cost cap reached" in buf0.getvalue()
    # transcript shape: replay + interactive turns with roles/costs/leaks
    lines = [json.loads(x) for x in transcript.read_text().splitlines()]
    roles = [(r.get("phase"), r.get("role")) for r in lines if "role" in r]
    assert roles == [("replay", "attacker"), ("replay", "target"),
                     ("replay", "attacker"), ("replay", "target"),
                     ("interactive", "attacker"),
                     ("interactive", "target")], roles
    assert lines[-1]["phase"] == "final", lines[-1]
    assert abs(lines[-1]["cost_usd"] - 0.003) < 1e-12, lines[-1]
    tgt = [r for r in lines if r.get("role") == "target"]
    assert (tgt[0]["leak"] is None and tgt[1]["leak"] == "substring"
            and tgt[2]["leak"] == "encoded"), tgt
    assert result["transcript"] == str(transcript)

    # per-phase strike budgets: a replay that aborts on 3 strikes still enters
    # interactive, and one interactive error must not inherit those strikes
    strike_winner = {"candidate_id": "m0-s", "model": "dry/model-a",
                     "round": 1, "leak": False, "leak_source": "",
                     "messages": ["s1", "s2", "s3"]}
    sb = StrikeReplayBackend(summary["secret"])
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        r_strike = run_session(
            args, sb, "dry/model-a", summary["secret"], strike_winner,
            scripted_input=iter(["fail once", "hello", "/exit"]),
            transcript_path=fixture / "transcript-strike.jsonl")
    o2 = buf2.getvalue()
    assert "aborted: 3 consecutive call errors during replay" in o2, o2
    assert "*** LEAK" not in o2, o2
    assert r_strike["errors"] == 4, r_strike["errors"]  # 3 replay + 1 interactive
    assert r_strike["interactive_turns"] == 1, r_strike
    # 3 replay messages x2 (retry) + 1 failed interactive x2 + 1 success
    assert sb.calls == 9, sb.calls
    interactive_aborts = [ln for ln in o2.splitlines()
                          if ln.strip() == "aborted: 3 consecutive call errors"]
    assert interactive_aborts == [], o2
    assert result["replay_leaks"] == [(2, "substring")], result["replay_leaks"]
    assert result["interactive_leaks"] == [(1, "encoded")], result["interactive_leaks"]
    assert "*** LEAK (substring) at replay turn 2 ***" in out, out
    assert "*** LEAK (encoded) ***" in out, out
    assert "you> " in out and "target> " in out, out
    # every call went to the SAME canary-target session, in order
    sessions = {s["session"] for s in backend.sent}
    assert len(sessions) == 1, sessions
    assert all(s["agent"] == "canary-target" for s in backend.sent)
    texts = [s["text"] for s in backend.sent]
    assert texts == ["replay msg 1", "replay msg 2", "follow up"], texts
    # cost accounting: 3 fake calls at $0.001
    assert backend.calls == 3, backend.calls
    assert abs(result["cost"] - 0.003) < 1e-12, result["cost"]
    assert abs(backend.cost - result["cost"]) < 1e-12
    say(f"fixture={fixture}")
    say("dry-run assertions: artifact missing/invalid/corrupt-line errors, "
        "LAST-leak selection + score fallback, model default follows the "
        "candidate, FLAG{hex} secret validation, replay order on one session, "
        "substring and encoded leak markers, interactive turn on the same "
        "session, per-phase 3-strike budgets (replay abort still enters "
        "interactive), transcript shape, max-cost-0 no-calls, cost accounting OK")
    say("DRY-RUN OK")


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Replay the best available forge candidate (leak or best "
                    "score) and keep talking to the canary target "
                    "(benign canary extraction only)")
    ap.add_argument("--artifact", default=None,
                    help="forge run dir containing summary.json + candidates.jsonl")
    ap.add_argument("--model", default=None,
                    help="override target model (default: artifact model)")
    ap.add_argument("--secret", default=None,
                    help="override secret (default: artifact secret; use a "
                         "fresh FLAG{...} to test the guard fresh)")
    ap.add_argument("--timeout", type=int, default=90,
                    help="seconds per model call")
    ap.add_argument("--max-cost", type=float, default=0.25,
                    help="abort once this tool's running API cost exceeds this "
                         "(USD); 0 means no calls")
    ap.add_argument("--outdir", default=None,
                    help="dir for the --dry-run fixture (default: runs/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline fixture check, no server/cost")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args)
        return

    summary, candidates = load_artifact(args.artifact)
    winner, why = pick_winner(candidates)
    model = resolve_model(args, summary, winner)
    secret = args.secret or summary.get("secret")
    if not model:
        raise SystemExit("no model: artifact summary has no models and "
                         "--model was not given")
    validate_secret(secret)
    say(f"artifact={args.artifact}  candidate={winner.get('candidate_id')} "
        f"(best available: {why})  origin_leak="
        f"{winner.get('leak_source') or '-'}")
    say(f"max_cost=${args.max_cost}  timeout={args.timeout}")

    # Write the canary target with the chosen secret, readback-asserted.
    forge.write_forge_agents(secret)
    if secret not in (AGENT_DIR / "canary-target.md").read_text():
        raise RuntimeError("secret mismatch: canary-target.md does not embed "
                           "the replay secret")

    proc, base = start_server(ROOT / "server-replay.log")
    backend = forge.LiveBackend(base, args.timeout)
    result = None
    try:
        result = run_session(args, backend, model, secret, winner)
    except KeyboardInterrupt:
        say("\ninterrupted")
    finally:
        proc.terminate()
        say(f"\ntotal_cost=${(result or {}).get('cost', backend.cost):.6f}")
        if result and result.get("transcript"):
            say(f"transcript: {result['transcript']}")


if __name__ == "__main__":
    main()
