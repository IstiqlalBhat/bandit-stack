"""M6 dogfood: the first run against REAL provider traffic (OpenAI).

Phase A (baseline): shadow mode — every request served by the premium model
(gpt-4o). The driver grades responses against known answers.

Phase B (assisted): the bandit routes between gpt-4o-mini and gpt-4o; the
driver posts graded feedback per request while a real gpt-4o-mini judge scores
a ~15% sample in the background (first reward wins — exercising the atomic
claim under genuine latency).

Budget: reads OPENAI_API_KEY from the repo .env; enforces a hard cap by
checking both proxies' cost ledgers every batch and aborting beyond
M6_BUDGET_ABORT_USD (default $4.50 of the approved $5).

Produces the real-dollar quality-per-dollar report. Exits nonzero on any
failed check.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("E2E_SCRATCH", "/tmp"))

DECISION_PORT, PROXY_PORT = 8641, 8642
ROUTE_NAME = os.environ.get("M6_ROUTE_NAME", "m6-dogfood")
CHEAP = os.environ.get("M6_CHEAP", "gpt-4o-mini")
PREMIUM = os.environ.get("M6_PREMIUM", "gpt-4o")
RATES = {  # usd per mtok in, out
    CHEAP: (float(os.environ.get("M6_CHEAP_IN", "0.15")), float(os.environ.get("M6_CHEAP_OUT", "0.60"))),
    PREMIUM: (float(os.environ.get("M6_PREMIUM_IN", "2.50")), float(os.environ.get("M6_PREMIUM_OUT", "10.00"))),
}
# per-request params; reasoning models need max_completion_tokens + minimal effort
BODY_PARAMS = json.loads(os.environ.get("M6_BODY_PARAMS", '{"max_tokens": 30}'))
JUDGE_PARAMS = json.loads(os.environ.get("M6_JUDGE_PARAMS", "null"))
BASELINE_ROUNDS = 100
ASSISTED_ROUNDS = 300
WINDOW = 150
ABORT_USD = float(os.environ.get("M6_BUDGET_ABORT_USD", "4.50"))
JUDGE_SAMPLE = 1.0  # judge everything; it only lands where feedback leaves room
FEEDBACK_SKIP_EVERY = 7  # leave every 7th request to the judge
USD_PER_QUALITY_POINT = float(os.environ.get("M6_LAMBDA", "0.001"))


def canonical_model(versioned: str) -> str:
    """OpenAI echoes versioned ids (gpt-4o-2024-...); map back to our arm names."""
    for name in sorted(RATES, key=len, reverse=True):
        if versioned.startswith(name):
            return name
    return versioned

KEYS = {
    "M6_DECISION_API_KEY": secrets.token_urlsafe(24),
    "M6_PROXY_CLIENT_KEY": secrets.token_urlsafe(24),
    "M6_PROXY_ADMIN_KEY": secrets.token_urlsafe(24),
}

results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results.append(ok)


def load_dotenv() -> dict:
    env = {}
    for line in (REPO / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    if "OPENAI_API_KEY" not in env:
        print("FAIL: OPENAI_API_KEY missing from .env")
        sys.exit(1)
    return env


# ---- verifiable task set ----------------------------------------------------
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDS = ["telemetry", "propensity", "bandit", "gradient", "posterior", "reward",
         "counterfactual", "conversion", "latency", "inference"]


def make_task(rng, i: int) -> tuple[str, str]:
    """Return (prompt, expected_answer) — every task exactly gradable."""
    kind = i % 5
    if kind == 0:
        a, b, c = int(rng.integers(12, 99)), int(rng.integers(3, 19)), int(rng.integers(10, 99))
        return f"Compute ({a} * {b}) + {c}. Reply with only the number.", str(a * b + c)
    if kind == 1:
        w = WORDS[int(rng.integers(len(WORDS)))]
        letter = rng.choice([ch for ch in set(w) if w.count(ch) >= 1])
        return (
            f"How many times does the letter '{letter}' appear in the word '{w}'? "
            "Reply with only the number.",
            str(w.count(letter)),
        )
    if kind == 2:
        n = int(rng.integers(1000, 9999))
        d = int(rng.integers(1, 28))
        amt = f"{int(rng.integers(20, 900))}.{int(rng.integers(10, 99))}"
        return (
            f"From this line: 'Invoice INV-{n} due 2026-08-{d:02d} totaling ${amt}' — "
            "reply with only the invoice ID.",
            f"INV-{n}",
        )
    if kind == 3:
        w = WORDS[int(rng.integers(len(WORDS)))]
        return f"Reply with only the word '{w}' spelled backwards.", w[::-1]
    start = int(rng.integers(7))
    delta = int(rng.integers(3, 23))
    return (
        f"If today is {DAYS[start]}, what day of the week is it {delta} days from now? "
        "Reply with only the day name.",
        DAYS[(start + delta) % 7],
    )


def grade(response_text: str, expected: str) -> float:
    got = re.sub(r"[\s.'\"`]+$", "", response_text.strip().strip("'\"`")).strip().lower()
    return 1.0 if got == expected.lower() else 0.0


# ---- infra ------------------------------------------------------------------
class Server:
    def __init__(self, name: str, args: list[str], env: dict) -> None:
        self.name = name
        self.log = open(SCRATCH / f"m6-{name}.log", "a")
        self.proc = subprocess.Popen(
            args, cwd=REPO, env={**os.environ, **env},
            stdout=self.log, stderr=subprocess.STDOUT,
        )

    def wait_healthy(self, url: str, attempts: int = 120) -> None:
        for _ in range(attempts):
            try:
                if httpx.get(url, timeout=1.0).status_code == 200:
                    return
            except httpx.TransportError:
                time.sleep(0.25)
        raise RuntimeError(f"{self.name} never became healthy at {url}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def proxy_config(mode: str, default: str, db_name: str, with_judge: bool) -> dict:
    cfg = {
        "mode": mode,
        "default_model": default,
        "models": [
            {"name": CHEAP, "base_url": "https://api.openai.com/v1",
             "api_key_env": "OPENAI_API_KEY",
             "input_usd_per_mtok": RATES[CHEAP][0], "output_usd_per_mtok": RATES[CHEAP][1]},
            {"name": PREMIUM, "base_url": "https://api.openai.com/v1",
             "api_key_env": "OPENAI_API_KEY",
             "input_usd_per_mtok": RATES[PREMIUM][0], "output_usd_per_mtok": RATES[PREMIUM][1]},
        ],
        "decision_api": {
            "base_url": f"http://127.0.0.1:{DECISION_PORT}",
            "api_key_env": "M6_DECISION_API_KEY",
            "route_name": ROUTE_NAME,
            "seed": 7,
            "timeout_s": 1.0,
        },
        "auth": {
            "client_api_key_env": "M6_PROXY_CLIENT_KEY",
            "admin_api_key_env": "M6_PROXY_ADMIN_KEY",
        },
        "reward": {"usd_per_quality_point": USD_PER_QUALITY_POINT},
        "database_url": f"sqlite:///{SCRATCH}/{db_name}",
    }
    if with_judge:
        cfg["judge"] = {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "model": CHEAP,
            "sample_rate": JUDGE_SAMPLE,
            "params": JUDGE_PARAMS,
        }
    return cfg


def main() -> None:
    import numpy as np

    dotenv = load_dotenv()
    for f in ("m6-decision.db", "m6-proxy-a.db", "m6-proxy-b.db"):
        (SCRATCH / f).unlink(missing_ok=True)
    server_env = {**KEYS, **dotenv}

    cfg_a = SCRATCH / "m6-config-a.json"
    cfg_a.write_text(json.dumps(proxy_config("shadow", PREMIUM, "m6-proxy-a.db", with_judge=False)))
    cfg_b = SCRATCH / "m6-config-b.json"
    cfg_b.write_text(json.dumps(proxy_config("assisted", PREMIUM, "m6-proxy-b.db", with_judge=True)))

    decision = Server(
        "decision",
        ["uv", "run", "uvicorn", "decision_api.main:app", "--port", str(DECISION_PORT), "--log-level", "warning"],
        env={**server_env, "DECISION_API_DB": f"sqlite:///{SCRATCH}/m6-decision.db",
             "DECISION_API_KEY": KEYS["M6_DECISION_API_KEY"]},
    )
    proxy = None
    rng = np.random.default_rng(20260709)
    spent_a = 0.0

    def admin_summary(client: httpx.Client) -> dict:
        return client.get(
            "/admin/summary",
            headers={"authorization": f"Bearer {KEYS['M6_PROXY_ADMIN_KEY']}"},
        ).json()

    def budget_guard(client: httpx.Client, judge_calls: int) -> float:
        judge_est = judge_calls * (180 * RATES[CHEAP][0] + 12 * RATES[CHEAP][1]) / 1e6
        total = spent_a + admin_summary(client)["total_cost_usd"] + judge_est
        if total > ABORT_USD:
            print(f"\nBUDGET ABORT: ${total:.4f} > ${ABORT_USD}")
            sys.exit(1)
        return total

    try:
        decision.wait_healthy(f"http://127.0.0.1:{DECISION_PORT}/healthz")

        # ---------------- Phase A: always-premium baseline ----------------
        print(f"PHASE A — baseline: shadow mode, 100% {PREMIUM}, {BASELINE_ROUNDS} real requests")
        proxy = Server("proxy-a",
                       ["uv", "run", "uvicorn", "llm_proxy.main:app", "--port", str(PROXY_PORT), "--log-level", "warning"],
                       env={**server_env, "LLM_PROXY_CONFIG": str(cfg_a)})
        proxy.wait_healthy(f"http://127.0.0.1:{PROXY_PORT}/readyz")
        client = httpx.Client(
            base_url=f"http://127.0.0.1:{PROXY_PORT}", timeout=60.0,
            headers={"authorization": f"Bearer {KEYS['M6_PROXY_CLIENT_KEY']}"},
        )

        grades_a: list[float] = []
        t0 = time.time()
        for i in range(BASELINE_ROUNDS):
            prompt, expected = make_task(rng, i)
            r = client.post("/v1/chat/completions", json={
                "model": "auto", **BODY_PARAMS,
                "messages": [{"role": "user", "content": prompt}],
            })
            if r.status_code != 200:
                check("baseline request failed", False, f"round {i}: {r.status_code} {r.text[:120]}")
                sys.exit(1)
            grades_a.append(grade(r.json()["choices"][0]["message"]["content"] or "", expected))
            if (i + 1) % 25 == 0:
                total = budget_guard(client, 0)
                print(f"  {i + 1}/{BASELINE_ROUNDS}  accuracy so far {sum(grades_a) / len(grades_a):.2f}  spent ${total:.4f}")
        summary_a = admin_summary(client)
        spent_a = summary_a["total_cost_usd"]
        base_quality = sum(grades_a) / len(grades_a)
        base_cpr = spent_a / summary_a["n_requests"]
        print(f"  baseline done in {time.time() - t0:.0f}s: accuracy {base_quality:.3f}, "
              f"${base_cpr:.6f}/req, total ${spent_a:.4f}")
        check("baseline served 100% premium",
              summary_a["served"] == {PREMIUM: BASELINE_ROUNDS}, f"served = {summary_a['served']}")
        proxy.stop()

        # ---------------- Phase B: assisted with real feedback ----------------
        print(f"\nPHASE B — assisted: bandit routes {CHEAP} vs {PREMIUM}, "
              f"{ASSISTED_ROUNDS} real requests + feedback (judge on {JUDGE_SAMPLE:.0%})")
        proxy = Server("proxy-b",
                       ["uv", "run", "uvicorn", "llm_proxy.main:app", "--port", str(PROXY_PORT), "--log-level", "warning"],
                       env={**server_env, "LLM_PROXY_CONFIG": str(cfg_b)})
        proxy.wait_healthy(f"http://127.0.0.1:{PROXY_PORT}/readyz")

        rounds: list[dict] = []
        judge_409 = 0
        t0 = time.time()
        for i in range(ASSISTED_ROUNDS):
            prompt, expected = make_task(rng, 1000 + i)
            r = client.post("/v1/chat/completions", json={
                "model": "auto", **BODY_PARAMS,
                "messages": [{"role": "user", "content": prompt}],
            })
            if r.status_code != 200:
                check("assisted request failed", False, f"round {i}: {r.status_code} {r.text[:120]}")
                sys.exit(1)
            q = grade(r.json()["choices"][0]["message"]["content"] or "", expected)
            rid = r.headers["x-proxy-request-id"]
            left_for_judge = i % FEEDBACK_SKIP_EVERY == 3
            if not left_for_judge:
                fb = client.post("/feedback", json={"request_id": rid, "quality": q})
                if fb.status_code == 409:
                    judge_409 += 1  # the real judge won the race — the claim held
                elif fb.status_code != 200:
                    check("feedback failed", False, f"round {i}: {fb.status_code} {fb.text[:120]}")
                    sys.exit(1)
            rounds.append(
                {"rid": rid, "model": canonical_model(r.json()["model"]), "grade": q}
            )
            if (i + 1) % 25 == 0:
                total = budget_guard(client, int((i + 1) * JUDGE_SAMPLE) + 1)
                mix = {m: sum(1 for x in rounds[-25:] if x["model"] == m) for m in RATES}
                print(f"  {i + 1}/{ASSISTED_ROUNDS}  last-25 mix {mix}  spent ${total:.4f}")

        # judge scoring runs in background tasks against a real API — drain it
        expected_judged = ASSISTED_ROUNDS // FEEDBACK_SKIP_EVERY
        admin_h = {"authorization": f"Bearer {KEYS['M6_PROXY_ADMIN_KEY']}"}
        prev = -1
        for _ in range(60):
            entries = client.get("/admin/requests", params={"limit": 1000}, headers=admin_h).json()
            n_judged = sum(1 for e in entries if e.get("quality_source") == "judge")
            if n_judged == prev and n_judged >= min(expected_judged, 10):
                break
            prev = n_judged
            time.sleep(1.0)

        # streaming smoke against the real provider
        s_ok = 0
        for _ in range(3):
            with client.stream("POST", "/v1/chat/completions", json={
                "model": "auto", "stream": True, **BODY_PARAMS,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            }) as resp:
                raw = b"".join(resp.iter_bytes())
            s_ok += resp.status_code == 200 and raw.decode().endswith("data: [DONE]\n\n")
        check("real-provider streaming passes through", s_ok == 3, f"{s_ok}/3 streams ok")

        elapsed = time.time() - t0
        print(f"  assisted done in {elapsed:.0f}s")
        summary_b = admin_summary(client)
        entries = client.get("/admin/requests", params={"limit": 1000},
                             headers={"authorization": f"Bearer {KEYS['M6_PROXY_ADMIN_KEY']}"}).json()
        by_rid = {e["request_id"]: e for e in entries}

        window = rounds[-WINDOW:]
        w_quality = sum(x["grade"] for x in window) / len(window)
        w_cost = sum(by_rid[x["rid"]]["cost_usd"] for x in window) / len(window)
        mix_all = {m: sum(1 for x in rounds if x["model"] == m) for m in RATES}
        mix_w = {m: sum(1 for x in window if x["model"] == m) for m in RATES}

        # judge validation: judge score vs our exact grade on the same request
        judged = [e for e in entries if e.get("quality_source") == "judge"]
        agree = dev = None
        graded_judged = [
            (e["quality"], next(x["grade"] for x in rounds if x["rid"] == e["request_id"]))
            for e in judged if any(x["rid"] == e["request_id"] for x in rounds)
        ]
        if graded_judged:
            agree = sum(1 for jq, g in graded_judged if (jq >= 0.5) == (g >= 0.5)) / len(graded_judged)
            dev = sum(abs(jq - g) for jq, g in graded_judged) / len(graded_judged)

        dclient = httpx.Client(base_url=f"http://127.0.0.1:{DECISION_PORT}", timeout=5.0,
                               headers={"authorization": f"Bearer {KEYS['M6_DECISION_API_KEY']}"})
        route_id = dclient.get(f"/routes/by-name/{ROUTE_NAME}").json()["id"]
        state = dclient.get(f"/routes/{route_id}/state").json()

        judge_est = len(judged) * (180 * RATES[CHEAP][0] + 12 * RATES[CHEAP][1]) / 1e6
        grand_total = spent_a + summary_b["total_cost_usd"] + judge_est

        print("\nverification:")
        check("every assisted request logged with cost",
              summary_b["n_requests"] == ASSISTED_ROUNDS + 3
              and all(by_rid[x["rid"]]["cost_usd"] is not None for x in rounds),
              f"n_requests = {summary_b['n_requests']}")
        check("ledgers agree (proxy rewards == decision rewards)",
              state["n_rewards"] == summary_b["rewards_posted"],
              f"decision = {state['n_rewards']}, proxy = {summary_b['rewards_posted']}")
        check("single-reward invariant held under real judge latency",
              summary_b["rewards_posted"] <= summary_b["n_requests"],
              f"{summary_b['rewards_posted']} rewards for {summary_b['n_requests']} requests "
              f"(feedback + judge, {judge_409} judge-won races, zero double-posts)")
        check("real judge scored the feedback-free slice", len(judged) >= 20,
              f"{len(judged)} judged (expected ~{ASSISTED_ROUNDS // FEEDBACK_SKIP_EVERY} + streams; "
              f"{judge_409} judge-won claim races on feedbacked traffic)")
        top_arm, top_n = max(mix_w.items(), key=lambda kv: kv[1])
        check("bandit concentrated on one arm", top_n / WINDOW >= 0.55,
              f"{top_arm} = {top_n}/{WINDOW} of the last {WINDOW}")
        check(f"hard cap respected (${ABORT_USD} abort line, $5 approved)",
              grand_total < ABORT_USD, f"grand total ${grand_total:.4f}")

        qpd_base = base_quality / base_cpr
        qpd_w = w_quality / w_cost
        print(f"\n{'=' * 66}\nM6 DOGFOOD — QUALITY PER DOLLAR ON REAL OPENAI TRAFFIC\n{'=' * 66}")
        print(f"{'strategy':30s} {'accuracy':>8s} {'$/request':>11s} {'$/1M req':>9s} {'quality/$':>10s}")
        print(f"{'always-' + PREMIUM + ' (baseline)':30s} {base_quality:8.3f} {base_cpr:11.6f} "
              f"{base_cpr * 1e6:9.0f} {qpd_base:10.0f}")
        print(f"{'bandit-routed (last %d)' % WINDOW:30s} {w_quality:8.3f} {w_cost:11.6f} "
              f"{w_cost * 1e6:9.0f} {qpd_w:10.0f}")
        print(f"\nmix all {ASSISTED_ROUNDS} rounds: {mix_all}   last {WINDOW}: {mix_w}")
        for m in RATES:
            served = [x["grade"] for x in rounds if x["model"] == m]
            if served:
                print(f"  {m}: {len(served)} pulls, measured accuracy {sum(served) / len(served):.3f}")
        print(f"reward tradeoff: 1 accuracy point valued at ${USD_PER_QUALITY_POINT}/request "
              f"(cost penalties: {CHEAP} {RATES[CHEAP][0] * 80 / 1e6 / USD_PER_QUALITY_POINT:.4f}, "
              f"{PREMIUM} ~{base_cpr / USD_PER_QUALITY_POINT:.3f} quality points)")
        print(f"cost reduction vs baseline: {(1 - w_cost / base_cpr) * 100:.0f}%   "
              f"accuracy delta: {w_quality - base_quality:+.3f}   quality-per-dollar: x{qpd_w / qpd_base:.1f}")
        if agree is not None:
            print(f"judge vs ground truth on {len(graded_judged)} sampled requests: "
                  f"{agree:.0%} agreement, mean |deviation| {dev:.2f}")
        print(f"\nREAL SPEND: phase A ${spent_a:.4f} + phase B ${summary_b['total_cost_usd']:.4f} "
              f"+ judge ~${judge_est:.4f} = ${grand_total:.4f} of $5.00 approved")
    finally:
        for s in (proxy, decision):
            if s is not None:
                s.stop()

    if all(results):
        print(f"\nM6 DOGFOOD PASSED ({len(results)} checks)")
    else:
        print(f"\nM6 DOGFOOD FAILED ({sum(results)}/{len(results)} checks passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
