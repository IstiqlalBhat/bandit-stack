"""End-to-end verification against a LIVE decision-api server.

Simulates the LLM-router scenario end to end, in the two phases a real
customer would run:

Phase 1 — SHADOW: a uniform-random route logs 400 decisions with exact
propensities. Off-policy evaluation must then recover every arm's true
success rate from the log alone — the counterfactual report that tells a
customer which model is best before any traffic is handed over.
(OPE needs coverage: you can't evaluate arms the log barely contains, which
is why this phase exists and why shadow mode is the product's on-ramp.)

Phase 2 — AUTOPILOT: a Thompson-sampling route runs 500 decide->reward
rounds over live HTTP and must concentrate traffic on the best model.

Exits nonzero if any check fails.

Usage: start the server, then `uv run python apps/reports/e2e_demo.py`
(BASE_URL env var overrides the default http://127.0.0.1:8123)
"""

from __future__ import annotations

import os
import sys
import time
import uuid

import httpx
import numpy as np

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8123")
DECISION_API_KEY = os.environ.get("DECISION_API_KEY")
TRUE_PROBS = {"haiku": 0.55, "sonnet": 0.70, "opus": 0.90}
ARMS = list(TRUE_PROBS)
SHADOW_ROUNDS = 400
AUTOPILOT_ROUNDS = 500


def wait_for_server(client: httpx.Client, attempts: int = 40) -> None:
    for _ in range(attempts):
        try:
            if client.get("/healthz").status_code == 200:
                return
        except httpx.TransportError:
            time.sleep(0.25)
    print(f"FAIL: server at {BASE} not reachable")
    sys.exit(1)


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def create_route(client: httpx.Client, label: str, policy: dict, seed: int) -> str:
    resp = client.post(
        "/routes",
        json={
            "name": f"{label}-{uuid.uuid4().hex[:8]}",
            "arms": ARMS,
            "policy": policy,
            "seed": seed,
        },
    )
    resp.raise_for_status()
    return resp.json()["id"]


def run_rounds(
    client: httpx.Client, route_id: str, n: int, rng: np.random.Generator
) -> list[str]:
    chosen = []
    for _ in range(n):
        d = client.post(f"/routes/{route_id}/decide", json={}).json()
        outcome = float(rng.random() < TRUE_PROBS[d["arm_name"]])
        client.post(
            "/rewards", json={"decision_id": d["decision_id"], "value": outcome}
        ).raise_for_status()
        chosen.append(d["arm_name"])
    return chosen


def snips_per_arm(client: httpx.Client, route_id: str) -> dict[str, float]:
    estimates = {}
    for i, arm in enumerate(ARMS):
        target = [0.0] * len(ARMS)
        target[i] = 1.0
        resp = client.post(f"/routes/{route_id}/ope", json={"target_probs": target})
        resp.raise_for_status()
        estimates[arm] = resp.json()["snips"]
    return estimates


def main() -> None:
    if not DECISION_API_KEY:
        print("FAIL: DECISION_API_KEY must be set")
        sys.exit(1)
    rng = np.random.default_rng(42)
    client = httpx.Client(
        base_url=BASE,
        timeout=10.0,
        headers={"authorization": f"Bearer {DECISION_API_KEY}"},
    )
    wait_for_server(client)
    results: list[bool] = []

    print(f"true success rates (hidden from all policies): {TRUE_PROBS}")

    # ---- Phase 1: shadow mode — uniform logging, counterfactual report ----
    print(f"\nPHASE 1 — shadow mode: uniform route, {SHADOW_ROUNDS} rounds")
    shadow_id = create_route(
        client,
        "shadow",
        policy={"type": "epsilon_greedy", "params": {"epsilon": 1.0}},
        seed=11,
    )
    t0 = time.time()
    shadow_chosen = run_rounds(client, shadow_id, SHADOW_ROUNDS, rng)
    print(
        f"  logged {SHADOW_ROUNDS} uniform decisions in {time.time() - t0:.1f}s "
        f"(pulls: {({a: shadow_chosen.count(a) for a in ARMS})})"
    )
    shadow_ope = snips_per_arm(client, shadow_id)
    for arm in ARMS:
        print(f"  snips(always-{arm:8s}) = {shadow_ope[arm]:.3f}  (truth {TRUE_PROBS[arm]:.2f})")

    results.append(
        check(
            "shadow OPE recovers every arm's rate",
            all(abs(shadow_ope[a] - TRUE_PROBS[a]) < 0.12 for a in ARMS),
            ", ".join(f"{a}: {shadow_ope[a]:.3f} vs {TRUE_PROBS[a]:.2f}" for a in ARMS),
        )
    )
    results.append(
        check(
            "shadow OPE ranks the arms correctly",
            shadow_ope["opus"] > shadow_ope["sonnet"] > shadow_ope["haiku"],
            f"opus {shadow_ope['opus']:.3f} > sonnet {shadow_ope['sonnet']:.3f} "
            f"> haiku {shadow_ope['haiku']:.3f}",
        )
    )

    # ---- Phase 2: autopilot — Thompson sampling converges ----
    print(f"\nPHASE 2 — autopilot: Thompson-sampling route, {AUTOPILOT_ROUNDS} rounds")
    auto_id = create_route(
        client,
        "autopilot",
        policy={"type": "beta_ts", "params": {"propensity_samples": 64}},
        seed=7,
    )
    t0 = time.time()
    chosen = run_rounds(client, auto_id, AUTOPILOT_ROUNDS, rng)
    elapsed = time.time() - t0
    print(
        f"  {AUTOPILOT_ROUNDS} decide->reward rounds over live HTTP in {elapsed:.1f}s "
        f"({2 * AUTOPILOT_ROUNDS / elapsed:.0f} req/s)"
    )

    state = client.get(f"/routes/{auto_id}/state").json()
    alpha = np.array(state["state"]["alpha"])
    beta = np.array(state["state"]["beta"])
    posterior = alpha / (alpha + beta)

    last200 = chosen[-200:]
    print(f"\n  {'arm':10s} {'pulls':>6s} {'share last 200':>15s} {'posterior':>10s} {'truth':>6s}")
    for i, arm in enumerate(ARMS):
        print(
            f"  {arm:10s} {chosen.count(arm):6d} {last200.count(arm) / len(last200):15.2f} "
            f"{posterior[i]:10.3f} {TRUE_PROBS[arm]:6.2f}"
        )

    auto_ope = snips_per_arm(client, auto_id)
    best_share = last200.count("opus") / len(last200)
    print("\nverification:")
    results.extend(
        [
            check(
                "autopilot converged on best arm",
                best_share > 0.7,
                f"opus share of last 200 rounds = {best_share:.2f} (need > 0.70)",
            ),
            check(
                "autopilot posterior ranks best arm first",
                int(np.argmax(posterior)) == ARMS.index("opus"),
                f"posterior means = {np.round(posterior, 3).tolist()}",
            ),
            check(
                "autopilot OPE recovers best arm's rate",
                abs(auto_ope["opus"] - TRUE_PROBS["opus"]) < 0.1,
                f"snips(always-opus) = {auto_ope['opus']:.3f} vs truth 0.90 "
                "(only the well-covered arm is precisely estimable from a converged log)",
            ),
            check(
                "persistence: every round logged",
                state["n_decisions"] == AUTOPILOT_ROUNDS
                and state["n_rewards"] == AUTOPILOT_ROUNDS,
                f"n_decisions={state['n_decisions']}, n_rewards={state['n_rewards']}",
            ),
        ]
    )

    if all(results):
        print("\nE2E VERIFICATION PASSED")
    else:
        print("\nE2E VERIFICATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
