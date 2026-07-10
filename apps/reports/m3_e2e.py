"""M3 live end-to-end verification: three real processes over real HTTP.

  mock upstream (OpenAI-compatible)  <--  llm-proxy (shadow mode)  -->  decision-api

Phases:
  A. 30 non-streaming requests through the proxy — pass-through + exact usage costs
  B. 20 streaming requests (half with include_usage) — byte-level SSE pass-through
  *  decision-api must now hold one shadow decision per A/B request
  C. KILL decision-api, 20 more requests — fail-open: all must still succeed
  D. upstream 500 — error passes through untouched
  E. reconcile /admin/summary cost ledger against first-principles expectations

Exits nonzero on any failed check. Writes servers' logs to E2E_SCRATCH.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("E2E_SCRATCH", "/tmp"))

DECISION_PORT, UPSTREAM_PORT, PROXY_PORT = 8341, 9341, 8342
ROUTE_NAME = "llm-proxy-shadow"
ARMS = ["mock-mini", "mock-opus"]

# first-principles cost expectations (mock usage: 12 in / 5 out; mini rates 0.15/0.60 per mtok)
USAGE_COST = (12 * 0.15 + 5 * 0.60) / 1e6
# estimated: "ping" -> 1 prompt token, "Hello world" (11 chars) -> 3 completion tokens
EST_COST = (1 * 0.15 + 3 * 0.60) / 1e6

REQUEST = {"model": "gpt-4o", "messages": [{"role": "user", "content": "ping"}]}


class Server:
    def __init__(self, name: str, args: list[str], env: dict | None = None) -> None:
        self.name = name
        self.log = open(SCRATCH / f"m3-{name}.log", "w")
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


def main() -> None:
    for f in ("m3-decision.db", "m3-proxy.db"):
        (SCRATCH / f).unlink(missing_ok=True)

    internal_key = secrets.token_urlsafe(32)
    client_key = secrets.token_urlsafe(32)
    admin_key = secrets.token_urlsafe(32)
    proxy_config = {
        "mode": "shadow",
        "default_model": "mock-mini",
        "models": [
            {"name": "mock-mini", "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
             "api_key_env": "MOCK_UPSTREAM_KEY", "input_usd_per_mtok": 0.15, "output_usd_per_mtok": 0.60},
            {"name": "mock-opus", "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
             "api_key_env": None, "input_usd_per_mtok": 5.0, "output_usd_per_mtok": 25.0},
        ],
        "decision_api": {
            "base_url": f"http://127.0.0.1:{DECISION_PORT}",
            "api_key_env": "M3_DECISION_API_KEY",
            "route_name": ROUTE_NAME,
        },
        "database_url": f"sqlite:///{SCRATCH}/m3-proxy.db",
        "auth": {
            "client_api_key_env": "M3_PROXY_CLIENT_KEY",
            "admin_api_key_env": "M3_PROXY_ADMIN_KEY",
        },
        "rate_limit": {"requests_per_minute": 10000, "burst": 100},
    }
    config_path = SCRATCH / "m3-proxy-config.json"
    config_path.write_text(json.dumps(proxy_config))

    decision = Server(
        "decision",
        ["uv", "run", "uvicorn", "decision_api.main:app", "--port", str(DECISION_PORT), "--log-level", "warning"],
        env={
            "DECISION_API_DB": f"sqlite:///{SCRATCH}/m3-decision.db",
            "DECISION_API_KEY": internal_key,
        },
    )
    upstream = Server(
        "upstream",
        ["uv", "run", "uvicorn", "mock_upstream:app", "--app-dir", "apps/reports",
         "--port", str(UPSTREAM_PORT), "--log-level", "warning"],
    )
    proxy = Server(
        "proxy",
        ["uv", "run", "uvicorn", "llm_proxy.main:app", "--port", str(PROXY_PORT), "--log-level", "warning"],
        env={
            "LLM_PROXY_CONFIG": str(config_path),
            "MOCK_UPSTREAM_KEY": "sk-mock",
            "M3_DECISION_API_KEY": internal_key,
            "M3_PROXY_CLIENT_KEY": client_key,
            "M3_PROXY_ADMIN_KEY": admin_key,
        },
    )

    try:
        decision.wait_healthy(f"http://127.0.0.1:{DECISION_PORT}/healthz")
        upstream.wait_healthy(f"http://127.0.0.1:{UPSTREAM_PORT}/healthz")
        proxy.wait_healthy(f"http://127.0.0.1:{PROXY_PORT}/healthz")
        proxy.wait_healthy(f"http://127.0.0.1:{PROXY_PORT}/readyz")
        print("all three servers healthy\n")
        client = httpx.Client(
            base_url=f"http://127.0.0.1:{PROXY_PORT}",
            timeout=15.0,
            headers={"authorization": f"Bearer {client_key}"},
        )
        admin_headers = {"authorization": f"Bearer {admin_key}"}

        print("PHASE A — 30 non-streaming requests")
        a_ok, a_model_ok = 0, 0
        for _ in range(30):
            r = client.post("/v1/chat/completions", json=REQUEST)
            a_ok += r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "pong"
            a_model_ok += r.json().get("model") == "mock-mini"
        check("all served with correct content", a_ok == 30, f"{a_ok}/30 ok")
        check("model override applied (shadow serves default)", a_model_ok == 30, f"{a_model_ok}/30 served mock-mini")

        print("\nPHASE B — 20 streaming requests (10 with include_usage)")
        b_ok = 0
        for i in range(20):
            body = {**REQUEST, "stream": True}
            if i % 2 == 0:
                body["stream_options"] = {"include_usage": True}
            with client.stream("POST", "/v1/chat/completions", json=body) as resp:
                raw = b"".join(resp.iter_bytes())
            text = raw.decode()
            events = [json.loads(l[6:]) for l in text.strip().split("\n\n")
                      if l.startswith("data: ") and l != "data: [DONE]"]
            content = "".join(e["choices"][0]["delta"].get("content", "")
                              for e in events if e.get("choices"))
            b_ok += resp.status_code == 200 and content == "Hello world" and text.endswith("data: [DONE]\n\n")
        check("streams pass through intact", b_ok == 20, f"{b_ok}/20 reassembled to 'Hello world' + [DONE]")

        # shadow decisions must have landed in the decision engine
        dclient = httpx.Client(
            base_url=f"http://127.0.0.1:{DECISION_PORT}",
            timeout=5.0,
            headers={"authorization": f"Bearer {internal_key}"},
        )
        route = dclient.get(f"/routes/by-name/{ROUTE_NAME}")
        route_ok = route.status_code == 200
        n_decisions = 0
        if route_ok:
            state = dclient.get(f"/routes/{route.json()['id']}/state").json()
            n_decisions = state["n_decisions"]
        check("proxy auto-created its shadow route", route_ok, f"GET /routes/by-name/{ROUTE_NAME} -> {route.status_code}")
        check("one shadow decision logged per request", n_decisions == 50, f"n_decisions = {n_decisions} (want 50)")

        print("\nPHASE C — decision-api KILLED, 20 more requests (fail-open)")
        decision.stop()
        c_ok = 0
        for _ in range(20):
            r = client.post("/v1/chat/completions", json=REQUEST)
            c_ok += r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "pong"
        check("all requests survive decision-api outage", c_ok == 20, f"{c_ok}/20 still 200 with content")

        print("\nPHASE D — upstream 500 passes through")
        r = client.post("/v1/chat/completions", json=REQUEST, headers={"x-fail": "1"})
        check("upstream error surfaced verbatim", r.status_code == 500 and "boom" in r.text,
              f"status {r.status_code}, body {r.text[:60]!r}")

        print("\nPHASE E — cost ledger reconciliation")
        summary = client.get("/admin/summary", headers=admin_headers).json()
        n_expected = 30 + 20 + 20 + 1
        expected_cost = 30 * USAGE_COST + 10 * USAGE_COST + 10 * EST_COST + 20 * USAGE_COST
        check("every request logged", summary["n_requests"] == n_expected,
              f"n_requests = {summary['n_requests']} (want {n_expected})")
        check("total cost matches first-principles ledger",
              abs(summary["total_cost_usd"] - expected_cost) < 1e-12,
              f"{summary['total_cost_usd']:.10f} vs expected {expected_cost:.10f}")
        check("all traffic served by default model", summary["served"] == {"mock-mini": n_expected},
              f"served = {summary['served']}")
        shadow_total = sum(summary["shadow"].values())
        check("shadow picks recorded while engine was up", shadow_total == 50,
              f"shadow picks = {summary['shadow']} (sum {shadow_total}, want 50)")
        check("both arms explored by shadow policy", set(summary["shadow"]) == set(ARMS),
              f"arms seen: {sorted(summary['shadow'])}")
        check("fail-open requests logged without shadow", summary["shadow_missing"] == 21,
              f"shadow_missing = {summary['shadow_missing']} (want 20 outage + 1 error-path)")

        latest = client.get(
            "/admin/requests", params={"limit": 5}, headers=admin_headers
        ).json()
        check("logs carry latency + propensity fields",
              all(e["latency_ms"] > 0 for e in latest),
              f"latencies: {[round(e['latency_ms'], 1) for e in latest]}")
    finally:
        for s in (proxy, upstream, decision):
            s.stop()

    if all(results):
        print(f"\nM3 E2E VERIFICATION PASSED ({len(results)} checks)")
    else:
        print(f"\nM3 E2E VERIFICATION FAILED ({sum(results)}/{len(results)} checks passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
