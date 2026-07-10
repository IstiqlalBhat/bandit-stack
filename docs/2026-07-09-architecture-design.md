# Architecture: Bandit Decision Engine + LLM Quality-per-Dollar Optimizer

Two products, one compounding core. Project one is a general-purpose contextual-bandit decision engine. Project two points that engine at LLM traffic — an OpenAI-compatible proxy that learns which model/prompt to use per request, scored by evals, optimizing quality-per-dollar.

## System overview

```
                        ┌─────────────────────────────────────────────┐
                        │              PRODUCT 2: llm-proxy           │
                        │  OpenAI-compatible endpoint (drop-in URL)   │
 customer app ──────────▶  1. extract context features from request   │
                        │  2. ask bandit: which model/prompt?  ───────┼──┐
                        │  3. forward to provider, stream back        │  │
                        │  4. log decision + cost                     │  │
                        └───────────────┬─────────────────────────────┘  │
                                        │ async                          │
                        ┌───────────────▼─────────────────────────────┐  │
                        │            evals (reward pipeline)          │  │
                        │  sampled LLM-judge · heuristics (retries,   │  │
                        │  JSON validity, regeneration) · explicit    │  │
                        │  feedback API · task verifiers              │  │
                        │  → composite reward = quality − λ·cost      │  │
                        └───────────────┬─────────────────────────────┘  │
                                        │ reward(decision_id, value)     │
                        ┌───────────────▼─────────────────────────────┐  │
                        │       PRODUCT 1: bandit decision engine     │◀─┘
                        │  decide(context, arms) → arm + propensity   │
                        │  reward(decision_id, value)                 │
                        │  policies: TS / LinTS / ε-greedy            │
                        │  off-policy evaluation (IPS, doubly-robust) │
                        │  decision log (Postgres)                    │
                        └─────────────────────────────────────────────┘
```

## Repo layout (monorepo)

```
rl-pipeline/
├── packages/
│   ├── bandit_core/          # pure library: zero I/O, fully unit-testable
│   │   ├── policies/          # thompson.py, lin_ts.py, epsilon_greedy.py
│   │   ├── ope/               # ips.py, doubly_robust.py
│   │   └── simulation/        # synthetic envs, regret-curve harness
│   └── evals/                 # judges, heuristic scorers, reward composition
├── services/
│   ├── decision_api/          # FastAPI wrapper: /decide, /reward, /policies
│   └── llm_proxy/             # OpenAI-compatible /v1/chat/completions
├── apps/
│   └── reports/               # CLI/notebook reports first; dashboard later
├── migrations/                # Postgres schema
└── docs/
```

## Key decisions and rationale

1. **Python everywhere (FastAPI + numpy/scipy + SQLAlchemy/Postgres).** One language across bandit math, evals, and the future GRPO tier. The proxy adds ~ms-level overhead which is noise against LLM latency (seconds).
2. **bandit_core is a pure library; the decision API is a thin wrapper.** The proxy imports the core in-process for v1 (no network hop); the HTTP decision API is the standalone Product 1 surface for non-LLM customers (pricing, paywalls, send-times).
3. **Every decision logs its propensity score.** This is non-negotiable — it's what makes off-policy evaluation possible, which is what makes shadow mode credible ("here's what our policy *would have* earned on your logged traffic").
4. **Thompson sampling as the default policy family.** Beta-Bernoulli for binary rewards, Gaussian for continuous, LinTS for contextual. ε-greedy kept as a baseline for regret comparisons.
5. **Composite reward = quality − λ·cost.** Quality from: sampled LLM-as-judge (5–10% of traffic, async), cheap heuristics on 100% (retry/regeneration detection, JSON/schema validity, latency), explicit feedback endpoint, and task verifiers where the customer has them. λ is a per-route customer knob. Framing is quality-per-dollar, not cost-cutting (the research showed pure cost-cutting durability is contested).
6. **Trust ladder: shadow → assisted → autopilot.** Shadow: 100% traffic to customer's default model, counterfactual policies scored via OPE + small sampled exploration. Assisted: policy routes within a customer-approved model allowlist with a quality floor. Autopilot: full routing. Nobody hands over routing on day one; shadow mode is the sales demo.
7. **Fail open, always.** Any proxy/bandit/provider error falls back to the customer's default model and passes the request through. The proxy must never be the reason a customer request fails. Streaming is pass-through.
8. **Guardrails are constraints, not rewards.** Per-route: allowed-model list, max cost per request, min quality threshold, daily budget cap. Enforced before the bandit chooses, so exploration can never violate them.
9. **Simulation before traffic.** The simulation harness (synthetic environments with known optimal arms) validates regret curves for every policy before anything touches real requests, and doubles as the regression suite.

## Data model (Postgres)

- `routes` — customer route config: candidate arms (models/prompts), λ, guardrails, mode (shadow/assisted/autopilot)
- `decisions` — id, route_id, context features (JSONB), candidate arms, chosen arm, propensity, policy version, latency, cost, ts
- `rewards` — decision_id, component (judge/heuristic/explicit/verifier), value, ts (composite computed at read/training time so λ can be retuned retroactively)
- `policy_snapshots` — serialized posterior state per route, versioned (enables rollback and audit)

## Milestones

- **M1 — bandit_core + simulation harness.** Policies implemented; regret curves beat ε-greedy baseline on synthetic envs; property-based tests on posterior updates. *Exit: plots + passing suite.*
- **M2 — decision API + logging + OPE.** /decide and /reward live against Postgres; IPS/DR reports over logged data. *Exit: replay evaluation on a synthetic log reproduces known policy values.*
- **M3 — llm_proxy in shadow mode.** OpenAI-compatible endpoint, provider adapters (Anthropic, OpenAI, Google, open models via a gateway), streaming, cost tracking, fail-open. *Exit: own traffic routed through it for a week, zero broken requests.*
- **M4 — evals + assisted routing.** Judge sampling + heuristics feeding composite rewards; assisted mode on own traffic; measured quality-per-dollar delta vs always-premium baseline. *Exit: a real savings/quality report on dogfooded traffic — this is the pilot-customer pitch artifact.*
- **M5 — pilot.** Reports app, onboarding flow (route config + API key swap), first external shadow-mode pilot with a VC-backed startup. Price >$250/mo.

## Deferred (deliberately)

Dashboard UI (CLI/notebook reports first) · GRPO post-training tier (premium, after routing proves value) · Redis hot-path cache (Postgres is fine at pilot scale) · multi-tenant auth beyond API keys · prompt-variant arms (model arms first; prompts are arm type #2).
