"""M5 live acceptance: the offline pilot, hardened — run against real
processes, real HTTP, and a real (disposable, Docker-composed) PostgreSQL.

Per the M5 spec, verifies: PostgreSQL persistence across a decision-api
restart, operator onboarding (created / reused / incompatible / down),
authenticated non-streaming and streaming proxying, client-vs-admin role
separation, rate limiting, mocked-judge reward attribution, decision outage
fail-open, bounded retention, and the single-reward invariant.

Requires Docker. Exits nonzero on any failed check.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("E2E_SCRATCH", "/tmp"))

DECISION_PORT, UPSTREAM_PORT, PROXY_PORT, LIMITED_PORT = 8541, 9541, 8542, 8543
PG_PORT = int(os.environ.get("M5_PG_PORT", "55432"))
PG_URL = f"postgresql+psycopg://rl:rl@127.0.0.1:{PG_PORT}/decisions"
ROUTE_NAME = "m5-router"

KEYS = {
    "M5_DECISION_API_KEY": secrets.token_urlsafe(24),
    "M5_PROXY_CLIENT_KEY": secrets.token_urlsafe(24),
    "M5_PROXY_ADMIN_KEY": secrets.token_urlsafe(24),
}

TASK_REQUEST = {"model": "auto", "messages": [{"role": "user", "content": "What is 21+21?"}]}

results: list[bool] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    results.append(ok)


class Server:
    def __init__(self, name: str, args: list[str], env: dict | None = None) -> None:
        self.name = name
        self.args = args
        self.env = env or {}
        self.log = open(SCRATCH / f"m5-{name}.log", "a")
        self.proc = subprocess.Popen(
            args, cwd=REPO, env={**os.environ, **KEYS, **self.env},
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


def compose(*args: str) -> None:
    subprocess.run(
        ["docker", "compose", "-f", "compose.postgres.yml", "-p", "m5pg", *args],
        cwd=REPO, check=True, capture_output=True,
        env={**os.environ, "M5_PG_PORT": str(PG_PORT)},
    )


def proxy_config(*, arms_reversed: bool = False, decision_port: int = DECISION_PORT,
                 db_name: str = "m5-proxy.db", burst: int = 1000) -> dict:
    models = [
        {"name": "mock-mini", "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
         "api_key_env": None, "input_usd_per_mtok": 0.15, "output_usd_per_mtok": 0.60},
        {"name": "mock-opus", "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
         "api_key_env": None, "input_usd_per_mtok": 5.0, "output_usd_per_mtok": 25.0},
    ]
    if arms_reversed:
        models.reverse()
    return {
        "mode": "assisted",
        "default_model": "mock-mini",
        "models": models,
        "decision_api": {
            "base_url": f"http://127.0.0.1:{decision_port}",
            "api_key_env": "M5_DECISION_API_KEY",
            "route_name": ROUTE_NAME,
            "seed": 7,
        },
        "auth": {
            "client_api_key_env": "M5_PROXY_CLIENT_KEY",
            "admin_api_key_env": "M5_PROXY_ADMIN_KEY",
        },
        "rate_limit": {"requests_per_minute": 60 if burst < 10 else 60000, "burst": burst},
        "retention_days": 30,
        "judge": {
            "base_url": f"http://127.0.0.1:{UPSTREAM_PORT}/v1",
            "model": "mock-judge",
            "sample_rate": 1.0,
        },
        "database_url": f"sqlite:///{SCRATCH}/{db_name}",
    }


def write_config(name: str, cfg: dict) -> Path:
    path = SCRATCH / name
    path.write_text(json.dumps(cfg))
    return path


def onboard(config_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["uv", "run", "python", "-m", "llm_proxy.onboard", str(config_path)],
        cwd=REPO, capture_output=True, text=True, env={**os.environ, **KEYS},
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def seed_ancient_request_log(db_url: str) -> None:
    """Plant a 90-day-old row so startup retention pruning has something to prune."""
    from llm_proxy.db import RequestLog, make_session_factory, utcnow

    factory = make_session_factory(db_url)
    with factory() as session:
        session.add(RequestLog(
            request_id="ancient-row",
            created_at=utcnow() - timedelta(days=90),
            served_model="mock-mini", stream=False, status_code=200, latency_ms=1.0,
        ))
        session.commit()


def start_decision() -> Server:
    server = Server(
        "decision",
        ["uv", "run", "uvicorn", "decision_api.main:app", "--port", str(DECISION_PORT), "--log-level", "warning"],
        env={"DECISION_API_DB": PG_URL, "DECISION_API_KEY": KEYS["M5_DECISION_API_KEY"]},
    )
    server.wait_healthy(f"http://127.0.0.1:{DECISION_PORT}/healthz")
    return server


def main() -> None:
    for f in ("m5-proxy.db", "m5-limited.db"):
        (SCRATCH / f).unlink(missing_ok=True)

    main_cfg = write_config("m5-config-main.json", proxy_config())
    bad_cfg = write_config("m5-config-bad.json", proxy_config(arms_reversed=True))
    down_cfg = write_config("m5-config-down.json", proxy_config(decision_port=1))
    limited_cfg = write_config(
        "m5-config-limited.json", proxy_config(db_name="m5-limited.db", burst=3)
    )

    print("starting disposable PostgreSQL via docker compose...")
    compose("up", "-d", "--wait")
    upstream = Server(
        "upstream",
        ["uv", "run", "uvicorn", "mock_upstream:app", "--app-dir", "apps/reports",
         "--port", str(UPSTREAM_PORT), "--log-level", "warning"],
        env={"MOCK_SEED": "4242"},
    )
    decision = proxy = limited = None
    try:
        decision = start_decision()
        upstream.wait_healthy(f"http://127.0.0.1:{UPSTREAM_PORT}/healthz")

        print("\nPHASE 1 — operator onboarding (CLI)")
        code, out = onboard(main_cfg)
        check("fresh route -> created, exit 0", code == 0 and "created" in out, out[:100])
        code, out = onboard(main_cfg)
        check("identical rerun -> reused, exit 0", code == 0 and "reused" in out, out[:100])
        code, out = onboard(bad_cfg)
        check("incompatible arms -> exit 2", code == 2 and "incompatible" in out.lower(), f"exit {code}: {out[:80]}")
        code, out = onboard(down_cfg)
        check("decision API down -> exit 3", code == 3 and "unreachable" in out.lower(), f"exit {code}: {out[:80]}")

        print("\nPHASE 2 — startup: readiness + retention pruning")
        seed_ancient_request_log(proxy_config()["database_url"])
        proxy = Server(
            "proxy",
            ["uv", "run", "uvicorn", "llm_proxy.main:app", "--port", str(PROXY_PORT), "--log-level", "warning"],
            env={"LLM_PROXY_CONFIG": str(main_cfg)},
        )
        limited = Server(
            "limited",
            ["uv", "run", "uvicorn", "llm_proxy.main:app", "--port", str(LIMITED_PORT), "--log-level", "warning"],
            env={"LLM_PROXY_CONFIG": str(limited_cfg)},
        )
        for port in (PROXY_PORT, LIMITED_PORT):
            Server.wait_healthy(proxy, f"http://127.0.0.1:{port}/healthz")
            Server.wait_healthy(proxy, f"http://127.0.0.1:{port}/readyz")
        check("both proxies healthy and ready (route provisioned)", True, "/healthz + /readyz 200")

        client = httpx.Client(
            base_url=f"http://127.0.0.1:{PROXY_PORT}", timeout=15.0,
            headers={"authorization": f"Bearer {KEYS['M5_PROXY_CLIENT_KEY']}"},
        )
        admin = {"authorization": f"Bearer {KEYS['M5_PROXY_ADMIN_KEY']}"}
        logs = client.get("/admin/requests", headers=admin).json()
        check("90-day-old log row pruned at startup",
              all(e["request_id"] != "ancient-row" for e in logs) and len(logs) == 0,
              f"{len(logs)} rows remain after prune")

        print("\nPHASE 3 — auth and role separation")
        bare = httpx.Client(base_url=f"http://127.0.0.1:{PROXY_PORT}", timeout=15.0)
        no_key = bare.post("/v1/chat/completions", json=TASK_REQUEST).status_code
        wrong_key = bare.post("/v1/chat/completions", json=TASK_REQUEST,
                              headers={"authorization": "Bearer nope"}).status_code
        check("proxy /v1 rejects missing and wrong keys",
              no_key == 401 and wrong_key == 401, f"missing -> {no_key}, wrong -> {wrong_key}")
        client_on_admin = client.get("/admin/summary").status_code
        admin_ok = bare.get("/admin/summary", headers=admin).status_code
        check("role separation: client key cannot read /admin",
              client_on_admin == 401 and admin_ok == 200,
              f"client-key -> {client_on_admin}, admin-key -> {admin_ok}")
        d_unauth = httpx.get(f"http://127.0.0.1:{DECISION_PORT}/routes/by-name/{ROUTE_NAME}").status_code
        d_auth = httpx.get(
            f"http://127.0.0.1:{DECISION_PORT}/routes/by-name/{ROUTE_NAME}",
            headers={"authorization": f"Bearer {KEYS['M5_DECISION_API_KEY']}"},
        )
        check("decision API requires its own bearer",
              d_unauth == 401 and d_auth.status_code == 200,
              f"unauth -> {d_unauth}, auth -> {d_auth.status_code}")

        print("\nPHASE 4 — authenticated traffic + judge reward attribution")
        n_ok = 0
        for _ in range(30):
            r = client.post("/v1/chat/completions", json=TASK_REQUEST)
            n_ok += r.status_code == 200 and "The answer is" in r.json()["choices"][0]["message"]["content"]
        check("30 non-streaming task requests served", n_ok == 30, f"{n_ok}/30 ok")
        s_ok = 0
        for _ in range(10):
            with client.stream("POST", "/v1/chat/completions",
                               json={**TASK_REQUEST, "stream": True}) as resp:
                raw = b"".join(resp.iter_bytes())
            s_ok += resp.status_code == 200 and raw.decode().endswith("data: [DONE]\n\n")
        check("10 streaming requests pass through", s_ok == 10, f"{s_ok}/10 ok")

        judged = []
        for _ in range(40):  # judge runs in background tasks; allow it to drain
            entries = client.get("/admin/requests", params={"limit": 100}, headers=admin).json()
            judged = [e for e in entries if e["quality_source"] == "judge" and e["reward_posted"]]
            if len(judged) >= 40:
                break
            time.sleep(0.25)
        check("mocked judge scored and posted a reward for every request",
              len(judged) >= 40, f"{len(judged)}/40 judged+posted (quality 0.8 mock)")

        dclient = httpx.Client(
            base_url=f"http://127.0.0.1:{DECISION_PORT}", timeout=5.0,
            headers={"authorization": f"Bearer {KEYS['M5_DECISION_API_KEY']}"},
        )
        route_id = dclient.get(f"/routes/by-name/{ROUTE_NAME}").json()["id"]
        state = dclient.get(f"/routes/{route_id}/state").json()
        summary = client.get("/admin/summary", headers=admin).json()
        check("proxy and decision ledgers agree on rewards",
              state["n_rewards"] == summary["rewards_posted"],
              f"decision n_rewards = {state['n_rewards']}, proxy rewards_posted = {summary['rewards_posted']}")

        rid = judged[0]["request_id"]
        dup = client.post("/feedback", json={"request_id": rid, "quality": 1.0})
        check("single-reward invariant: feedback after judge -> 409",
              dup.status_code == 409, f"second reward attempt -> {dup.status_code}")

        print("\nPHASE 5 — rate limiting (burst=3, 60 rpm)")
        lclient = httpx.Client(
            base_url=f"http://127.0.0.1:{LIMITED_PORT}", timeout=15.0,
            headers={"authorization": f"Bearer {KEYS['M5_PROXY_CLIENT_KEY']}"},
        )
        codes = [lclient.post("/v1/chat/completions", json=TASK_REQUEST).status_code for _ in range(4)]
        limited_hit = codes[:3] == [200, 200, 200] and codes[3] == 429
        check("burst exhaustion returns 429", limited_hit, f"codes = {codes}")
        time.sleep(1.5)  # 60 rpm -> one token/second
        recovered = lclient.post("/v1/chat/completions", json=TASK_REQUEST).status_code
        check("bucket refills and traffic recovers", recovered == 200, f"after 1.5s -> {recovered}")

        print("\nPHASE 6 — PostgreSQL persistence across decision-api restart")
        import psycopg  # workspace dependency; also proves the driver is installed

        with psycopg.connect(f"postgresql://rl:rl@127.0.0.1:{PG_PORT}/decisions") as conn:
            n_routes = conn.execute("SELECT count(*) FROM routes").fetchone()[0]
            n_decisions_pg = conn.execute("SELECT count(*) FROM decisions").fetchone()[0]
        check("decision data physically lives in PostgreSQL",
              n_routes >= 1 and n_decisions_pg >= 40,
              f"routes = {n_routes}, decisions = {n_decisions_pg}")
        state_before = dclient.get(f"/routes/{route_id}/state").json()
        decision.stop()
        decision = start_decision()
        state_after = dclient.get(f"/routes/{route_id}/state").json()
        still_serves = client.post("/v1/chat/completions", json=TASK_REQUEST).status_code == 200
        check("posterior survives restart byte-for-byte",
              state_after["state"] == state_before["state"]
              and state_after["policy_version"] == state_before["policy_version"],
              f"policy_version {state_after['policy_version']} preserved")
        check("stack serves after restart", still_serves, "post-restart request -> 200")

        print("\nPHASE 7 — decision outage fail-open")
        decision.stop()
        outage_ok = 0
        for _ in range(10):
            r = client.post("/v1/chat/completions", json=TASK_REQUEST)
            outage_ok += r.status_code == 200
        check("proxy serves through total decision outage", outage_ok == 10, f"{outage_ok}/10 still 200")
    finally:
        for s in (proxy, limited, upstream, decision):
            if s is not None:
                s.stop()
        try:
            compose("down", "-v", "--remove-orphans")
        except subprocess.CalledProcessError as exc:
            print(f"warning: compose teardown failed: {exc}")

    if all(results):
        print(f"\nM5 E2E VERIFICATION PASSED ({len(results)} checks)")
    else:
        print(f"\nM5 E2E VERIFICATION FAILED ({sum(results)}/{len(results)} checks passed)")
        sys.exit(1)


if __name__ == "__main__":
    main()
