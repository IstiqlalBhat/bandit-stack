"""M4 live end-to-end: the quality-per-dollar report.

Three model arms answer arithmetic tasks with different hidden accuracies and
very different prices:

  arm          accuracy   $/req (12 in + 5 out tok)   reward = q − cost/$0.0005
  mock-mini    0.60       $0.0000048                  ≈ 0.590
  mock-sonnet  0.92       $0.000111                   ≈ 0.698   <- optimal
  mock-opus    0.95       $0.000185                   ≈ 0.580

Phase 1 (baseline): shadow mode, always-serve mock-opus — the "just use the
premium model" strategy. Driver measures its true quality and cost.

Phase 2 (assisted): the bandit routes; after each response the driver checks
the answer itself and posts explicit feedback; the proxy composes quality
with cost and posts one reward per request to the decision engine.

Verified end to end: convergence to the quality-per-dollar-optimal arm,
reward bookkeeping consistency across both services, composite math, and the
final report: comparable quality at a fraction of the cost.
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

DECISION_PORT, UPSTREAM_PORT, PROXY_PORT = 8441, 9441, 8442
ROUTE_NAME = "m4-router"
U = 0.0005  # usd per quality point
BASELINE_ROUNDS = 150
ASSISTED_ROUNDS = 700
WINDOW = 200

ANSWER_RE = re.compile(r"The answer is (-?\d+)\.")
INTERNAL_KEY = secrets.token_urlsafe(32)
CLIENT_KEY = secrets.token_urlsafe(32)
ADMIN_KEY = secrets.token_urlsafe(32)

MODELS = [
    {"name": "mock-mini", "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
     "api_key_env": None, "input_usd_per_mtok": 0.15, "output_usd_per_mtok": 0.60},
    {"name": "mock-sonnet", "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
     "api_key_env": None, "input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0},
    {"name": "mock-opus", "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
     "api_key_env": None, "input_usd_per_mtok": 5.0, "output_usd_per_mtok": 25.0},
]


class Server:
    def __init__(self, name: str, args: list[str], env: dict | None = None) -> None:
        self.name = name
        self.log = open(SCRATCH / f"m4-{name}.log", "w")
        self.proc = subprocess.Popen(
            args, cwd=REPO, env={**os.environ, **(env or {})},
            stdout=self.log, stderr=subprocess.STDOUT,
        )

    def wait_healthy(self, url: str, attempts: int = 60) -> None:
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
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results.append(ok)


def proxy_config(mode: str, default: str, db_name: str) -> dict:
    return {
        "mode": mode,
        "default_model": default,
        "models": MODELS,
        "decision_api": {
            "base_url": f"http://127.0.0.1:{DECISION_PORT}",
            "api_key_env": "M4_DECISION_API_KEY",
            "route_name": ROUTE_NAME,
        },
        "database_url": f"sqlite:///{SCRATCH}/{db_name}",
        "auth": {
            "client_api_key_env": "M4_PROXY_CLIENT_KEY",
            "admin_api_key_env": "M4_PROXY_ADMIN_KEY",
        },
        "rate_limit": {"requests_per_minute": 100000, "burst": 1000},
        "reward": {"usd_per_quality_point": U},
    }


def start_proxy(mode: str, default: str, db_name: str, config_name: str) -> Server:
    path = SCRATCH / config_name
    path.write_text(json.dumps(proxy_config(mode, default, db_name)))
    server = Server(
        f"proxy-{mode}",
        ["uv", "run", "uvicorn", "llm_proxy.main:app", "--port", str(PROXY_PORT), "--log-level", "warning"],
        env={
            "LLM_PROXY_CONFIG": str(path),
            "M4_DECISION_API_KEY": INTERNAL_KEY,
            "M4_PROXY_CLIENT_KEY": CLIENT_KEY,
            "M4_PROXY_ADMIN_KEY": ADMIN_KEY,
        },
    )
    server.wait_healthy(f"http://127.0.0.1:{PROXY_PORT}/healthz")
    server.wait_healthy(f"http://127.0.0.1:{PROXY_PORT}/readyz")
    return server


def run_task_round(client: httpx.Client, i: int, give_feedback: bool) -> dict:
    a, b = 13 + (i * 7) % 80, 4 + (i * 11) % 60
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": f"What is {a}+{b}?"}]},
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    match = ANSWER_RE.search(content)
    quality = 1.0 if (match and int(match.group(1)) == a + b) else 0.0
    out = {
        "request_id": resp.headers["x-proxy-request-id"],
        "model": resp.json()["model"],
        "quality": quality,
        "reward_posted": None,
    }
    if give_feedback:
        fb = client.post("/feedback", json={"request_id": out["request_id"], "quality": quality})
        fb.raise_for_status()
        out["reward_posted"] = fb.json()["reward_posted"]
    return out


def main() -> None:
    for f in ("m4-decision.db", "m4-proxy-a.db", "m4-proxy-b.db"):
        (SCRATCH / f).unlink(missing_ok=True)

    decision = Server(
        "decision",
        ["uv", "run", "uvicorn", "decision_api.main:app", "--port", str(DECISION_PORT), "--log-level", "warning"],
        env={
            "DECISION_API_DB": f"sqlite:///{SCRATCH}/m4-decision.db",
            "DECISION_API_KEY": INTERNAL_KEY,
        },
    )
    upstream = Server(
        "upstream",
        ["uv", "run", "uvicorn", "mock_upstream:app", "--app-dir", "apps/reports",
         "--port", str(UPSTREAM_PORT), "--log-level", "warning"],
        env={"MOCK_SEED": "4242"},
    )
    proxy = None
    try:
        decision.wait_healthy(f"http://127.0.0.1:{DECISION_PORT}/healthz")
        upstream.wait_healthy(f"http://127.0.0.1:{UPSTREAM_PORT}/healthz")

        # ---- Phase 1: baseline = always premium (shadow serves default) ----
        print(f"PHASE 1 — baseline: always mock-opus, {BASELINE_ROUNDS} task requests")
        proxy = start_proxy("shadow", "mock-opus", "m4-proxy-a.db", "m4-config-a.json")
        client = httpx.Client(
            base_url=f"http://127.0.0.1:{PROXY_PORT}",
            timeout=15.0,
            headers={"authorization": f"Bearer {CLIENT_KEY}"},
        )
        base_rounds = [run_task_round(client, i, give_feedback=False) for i in range(BASELINE_ROUNDS)]
        base_summary = client.get(
            "/admin/summary",
            headers={"authorization": f"Bearer {ADMIN_KEY}"},
        ).json()
        base_quality = sum(r["quality"] for r in base_rounds) / len(base_rounds)
        base_cpr = base_summary["total_cost_usd"] / base_summary["n_requests"]
        print(f"  quality {base_quality:.3f}, cost/request ${base_cpr:.6f}")
        check("baseline served only the premium model",
              base_summary["served"] == {"mock-opus": BASELINE_ROUNDS},
              f"served = {base_summary['served']}")
        proxy.stop()

        # ---- Phase 2: assisted = bandit routes, feedback closes the loop ----
        print(f"\nPHASE 2 — assisted: bandit routes, {ASSISTED_ROUNDS} rounds with feedback")
        proxy = start_proxy("assisted", "mock-opus", "m4-proxy-b.db", "m4-config-b.json")
        client = httpx.Client(
            base_url=f"http://127.0.0.1:{PROXY_PORT}",
            timeout=15.0,
            headers={"authorization": f"Bearer {CLIENT_KEY}"},
        )
        t0 = time.time()
        rounds = [run_task_round(client, i, give_feedback=True) for i in range(ASSISTED_ROUNDS)]
        elapsed = time.time() - t0
        print(f"  {ASSISTED_ROUNDS} decide->serve->score->feedback loops in {elapsed:.1f}s")

        admin_headers = {"authorization": f"Bearer {ADMIN_KEY}"}
        summary = client.get("/admin/summary", headers=admin_headers).json()
        logs = client.get(
            "/admin/requests", params={"limit": 1000}, headers=admin_headers
        ).json()
        logs_by_rid = {e["request_id"]: e for e in logs}
        window = rounds[-WINDOW:]
        window_share = {}
        for r in window:
            window_share[r["model"]] = window_share.get(r["model"], 0) + 1
        window_quality = sum(r["quality"] for r in window) / len(window)
        window_cost = sum(logs_by_rid[r["request_id"]]["cost_usd"] for r in window) / len(window)

        print(f"\n  served mix over all {ASSISTED_ROUNDS} rounds: "
              f"{ {m: sum(1 for r in rounds if r['model'] == m) for m in window_share} }")
        print(f"  last {WINDOW}: mix {window_share}, quality {window_quality:.3f}, cost/request ${window_cost:.6f}")

        print("\nverification:")
        sonnet_share = window_share.get("mock-sonnet", 0) / WINDOW
        check("bandit converged on quality-per-dollar optimum (mock-sonnet)",
              sonnet_share > 0.6, f"sonnet share of last {WINDOW} = {sonnet_share:.2f} (need > 0.60)")

        posted = sum(1 for r in rounds if r["reward_posted"])
        check("feedback posted a reward for (nearly) every round",
              posted >= ASSISTED_ROUNDS - 5, f"{posted}/{ASSISTED_ROUNDS} rewards posted")
        check("proxy ledger agrees", summary["rewards_posted"] == posted,
              f"proxy rewards_posted = {summary['rewards_posted']}")

        dclient = httpx.Client(
            base_url=f"http://127.0.0.1:{DECISION_PORT}",
            timeout=5.0,
            headers={"authorization": f"Bearer {INTERNAL_KEY}"},
        )
        route_id = dclient.get(f"/routes/by-name/{ROUTE_NAME}").json()["id"]
        state = dclient.get(f"/routes/{route_id}/state").json()
        check("decision engine received exactly those rewards",
              state["n_rewards"] == posted, f"decision-api n_rewards = {state['n_rewards']}")
        check("decision engine logged every decision (both phases)",
              state["n_decisions"] == BASELINE_ROUNDS + ASSISTED_ROUNDS,
              f"n_decisions = {state['n_decisions']}")

        sample = next(e for e in logs if e["reward_posted"])
        expected_reward = min(1.0, max(0.0, sample["quality"] - sample["cost_usd"] / U))
        check("composite reward math verified on a logged entry",
              abs(sample["reward_value"] - expected_reward) < 1e-12,
              f"reward {sample['reward_value']:.6f} == clamp(q {sample['quality']} - cost/U {sample['cost_usd'] / U:.4f})")

        check("quality held within 0.08 of always-premium",
              window_quality >= base_quality - 0.08,
              f"{window_quality:.3f} vs baseline {base_quality:.3f}")
        check("cost per request cut by ≥ 30%",
              window_cost <= 0.70 * base_cpr,
              f"${window_cost:.6f} vs baseline ${base_cpr:.6f} ({(1 - window_cost / base_cpr) * 100:.0f}% saved)")

        qpd_base = base_quality / base_cpr
        qpd_assisted = window_quality / window_cost
        print(f"\n{'=' * 62}\nQUALITY-PER-DOLLAR REPORT (mock task traffic)\n{'=' * 62}")
        print(f"{'strategy':28s} {'quality':>8s} {'$/request':>11s} {'$/1M req':>9s} {'quality/$':>10s}")
        print(f"{'always-premium (baseline)':28s} {base_quality:8.3f} {base_cpr:11.6f} {base_cpr * 1e6:9.0f} {qpd_base:10.0f}")
        print(f"{'bandit-routed (last %d)' % WINDOW:28s} {window_quality:8.3f} {window_cost:11.6f} {window_cost * 1e6:9.0f} {qpd_assisted:10.0f}")
        print(f"\ncost reduction: {(1 - window_cost / base_cpr) * 100:.0f}%   "
              f"quality delta: {window_quality - base_quality:+.3f}   "
              f"quality-per-dollar: x{qpd_assisted / qpd_base:.1f}")
    finally:
        for s in (proxy, upstream, decision):
            if s is not None:
                s.stop()

    if all(results):
        print(f"\nM4 E2E VERIFICATION PASSED ({len(results)} checks)")
    else:
        print(f"\nM4 E2E VERIFICATION FAILED ({sum(results)}/{len(results)} checks passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
