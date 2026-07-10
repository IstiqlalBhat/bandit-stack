# Bandit Stack

**A contextual-bandit decision engine and an OpenAI-compatible LLM router that learns which model earns its cost — measured on your traffic, not assumed.**

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen)
![License: MIT](https://img.shields.io/badge/license-MIT-black)
![FastAPI](https://img.shields.io/badge/FastAPI-services-009688)

Premium models' share of LLM spend is exploding, and most teams pick a model once and route everything to it. Per-request model selection is the single biggest cost lever in an LLM bill — but "always use the cheap model" is just the same guess in the other direction. Bandit Stack closes the loop instead: it **routes each request, scores the outcome, and learns** which model maximizes quality-per-dollar on *your* workload.

Sometimes that saves you 40%. Sometimes it proves the premium model is worth every cent. Both runs are in this repo, with receipts.

---

## How it works

```
                        ┌─────────────────────────────────────────────┐
                        │                  llm-proxy                  │
                        │   OpenAI-compatible endpoint (drop-in URL)  │
 your app ──────────────▶  1. ask the bandit: which model?            │
                        │  2. serve it (assisted) or your default     │
                        │     (shadow) — streams pass through intact  │
                        │  3. log decision, tokens, exact cost        │
                        └───────────────┬─────────────────────────────┘
                                        │ async
                        ┌───────────────▼─────────────────────────────┐
                        │              quality signals                │
                        │   POST /feedback (your grader / users)      │
                        │   + sampled LLM-as-judge                    │
                        │   reward = quality − cost / λ               │
                        └───────────────┬─────────────────────────────┘
                                        │ one reward per request
                        ┌───────────────▼─────────────────────────────┐
                        │            decision engine                  │
                        │  Thompson sampling · propensity-logged      │
                        │  decisions · off-policy evaluation          │
                        │  (IPS / SNIPS / doubly-robust)              │
                        └─────────────────────────────────────────────┘
```

Two products, one compounding core:

1. **`decision_api`** — a general-purpose contextual-bandit service. `POST /decide` returns an arm **with its propensity**; `POST /rewards` updates a versioned, restart-safe posterior; `POST /ope` answers *"what would policy X have earned on my logged traffic?"* It optimizes LLM routing here, but it's the same engine you'd point at pricing, paywalls, or send-times.
2. **`llm_proxy`** — an OpenAI-compatible proxy (swap one base URL) that uses the engine to pick the model per request and feeds outcomes back as rewards.

### The trust ladder

- **Shadow mode** — zero risk. 100% of traffic goes to your current default; the bandit only *logs* what it would have done. Because every decision carries a propensity, off-policy evaluation can then tell you what any routing policy *would have earned* — before you hand over a single request.
- **Assisted mode** — the bandit routes within the models you configured, and every failure path falls back to your default. The proxy is built to never be the reason a request fails: engine down → plain proxying, upstream error → passed through verbatim, judge broken → missing sample, nothing more.

---

## Real results

**Simulated workload** (mock upstream, hidden success rates — `docs/m4-quality-per-dollar-report.txt`):

| strategy | quality | $/1M requests | quality per $ |
|---|---|---|---|
| always-premium | 0.960 | $185 | 5,189 |
| **bandit-routed** | 0.935 | **$114** | **8,205 (×1.6)** |

**Real OpenAI traffic** (400+ live requests, exact usage-priced ledger — `docs/m6-dogfood-report.txt`):

| model | measured accuracy | verdict at λ = $0.001/point |
|---|---|---|
| gpt-4o-mini | 0.611 | explored 54×, correctly **abandoned** |
| gpt-4o | 0.882 | converged: 144 of the last 150 requests |

Same system, opposite conclusions — because the workloads genuinely differed. That's the point: **it measures, it doesn't guess.** The λ knob (`usd_per_quality_point`) is how you tell it what a quality point is worth to your business; the bandit does the arithmetic from there.

Bonus finding from the live run: a gpt-4o-mini judge agreed with ground truth only 56% of the time (mean deviation 0.45). Cheap judges are noisy — prefer explicit feedback or verifiers where your task allows, and treat judge scores as a sample, not gospel.

---

## Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync && uv run pytest        # install + 126 tests

# every service authenticates with Bearer keys — generate throwaways:
export DECISION_API_KEY=$(openssl rand -hex 24)
export LLM_PROXY_CLIENT_KEY=$(openssl rand -hex 24)   # callers of /v1/*
export LLM_PROXY_ADMIN_KEY=$(openssl rand -hex 24)    # callers of /admin/*
export OPENAI_API_KEY=sk-...                          # or any OpenAI-compatible provider

uv run uvicorn decision_api.main:app --port 8123               # 1. decision engine
uv run python -m llm_proxy.onboard configs/llm_proxy.example.json   # 2. provision the route
LLM_PROXY_CONFIG=configs/llm_proxy.example.json \
  uv run uvicorn llm_proxy.main:app --port 8200                # 3. proxy
```

Point your OpenAI client at `http://localhost:8200/v1` with the client key, and post quality when you know it:

```sh
curl -X POST localhost:8200/v1/chat/completions \
  -H "Authorization: Bearer $LLM_PROXY_CLIENT_KEY" -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hello"}]}'
# response header x-proxy-request-id: <rid>

curl -X POST localhost:8200/feedback \
  -H "Authorization: Bearer $LLM_PROXY_CLIENT_KEY" -H "Content-Type: application/json" \
  -d '{"request_id": "<rid>", "quality": 1.0}'
```

`GET /admin/summary` shows spend, served/shadow mix, and rewards posted. SQLite by default; point `DECISION_API_DB` / `database_url` at `postgresql+psycopg://...` for Postgres (`compose.postgres.yml` runs a disposable one).

---

## What's inside

- **Policies** — Beta-Bernoulli & Gaussian Thompson sampling, LinTS (contextual), ε-greedy baseline. Pure library, zero I/O, regret-curve validated against known-optimal synthetic environments.
- **Propensity logging on every decision** — the thing that makes counterfactual evaluation possible. Monte-Carlo estimated, Laplace-smoothed so IPS weights stay finite.
- **Off-policy evaluation** — IPS, SNIPS, doubly-robust. In the live demo, OPE recovers every arm's hidden success rate from 400 uniform logged decisions within ±0.02.
- **One reward per request, causally attributed** — a reward posts only when the arm the bandit picked is the arm that was served (always in assisted, on coincidence in shadow); explicit feedback and the async judge race through an **atomic claim**, so real-world latency can't double-train the policy. This bug was caught by dogfooding against a real judge; the fix is `claim_reward_slot` and it held live: 301 rewards / 303 requests, zero double-posts.
- **Exact cost accounting** — usage-based when the provider reports tokens (`cost_source: "usage"`), chars/4 estimate otherwise (`"estimated"`), never silently guessed. The M3 verification reconciles the ledger against first-principles expectations to the last digit.
- **Production hardening** — constant-time Bearer auth with client/admin role separation, token-bucket rate limiting, bounded log retention, Postgres persistence (posterior survives restarts byte-for-byte), `/readyz`, an operator onboarding CLI with branchable exit codes, and fail-open on every internal dependency.

## Verified, not vibed

Every milestone gates on a live multi-process verification script that spawns real servers, drives real HTTP, and exits nonzero on any failed check. The artifacts are committed:

| run | what it proves | artifact |
|---|---|---|
| regret curves | TS/LinTS beat baselines, sublinear regret | `docs/plots/m1-regret-curves.png` |
| shadow→autopilot | OPE recovers hidden arm values; TS converges | `docs/m2-e2e-verification.txt` |
| proxy (14 checks) | pass-through, streaming, kill-the-engine fail-open, exact cost ledger | `docs/m3-e2e-verification.txt` |
| reward loop (9 checks) | bandit finds the quality-per-dollar optimum | `docs/m4-quality-per-dollar-report.txt` |
| hardening (20 checks) | auth, rate limits, retention, Postgres restart, onboarding | `docs/m5-e2e-verification.txt` |
| **real traffic (8 checks)** | everything above against live OpenAI, ~$0.03 | `docs/m6-dogfood-report.txt` |

```sh
uv run python apps/reports/m3_e2e.py    # spawns everything it needs
uv run python apps/reports/m4_e2e.py
uv run python apps/reports/m5_e2e.py    # needs Docker (disposable Postgres)
uv run python apps/reports/m6_dogfood.py  # needs OPENAI_API_KEY in .env; hard budget cap built in
```

## Project structure

```
packages/bandit_core/     policies · off-policy evaluation · simulation harness (pure, zero I/O)
packages/evals/           composite quality-per-dollar reward · scorers · LLM-as-judge client
services/decision_api/    the bandit engine over HTTP, propensity-logged store, snapshots, OPE
services/llm_proxy/       OpenAI-compatible router: shadow/assisted, feedback, judge, auth,
                          rate limiting, retention, onboarding CLI
apps/reports/             regret report + the live end-to-end verification suites
docs/                     architecture, specs/plans, and every verification artifact
```

## Roadmap & honest limits

Single-tenant Bearer keys (one client key, one admin key) · process-local rate limiting (single worker) · schema is create-only (no migrations tooling yet) · judge scores are noisy with small judge models · LinTS is implemented and tested but the proxy currently routes non-contextually — request-feature routing is the natural next step, the plumbing (context logging) is already in place.

## License

[MIT](LICENSE)
