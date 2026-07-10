# M5 Offline Pilot Hardening Design

## Goal

Turn the verified M1–M4 prototype into a safe, reproducible single-tenant
pilot rehearsal without calling a real LLM provider. Actual M5 remains open
until real dogfood and an external customer are completed with user approval.

## Chosen scope

The offline rehearsal will exercise both service databases on a disposable
local PostgreSQL 17 instance, provision the proxy's decision route before
traffic, secure all non-health surfaces, bound request history, and provide a
five-minute mock-only quickstart. M3 and M4 remain compatible and continue to
run against SQLite so both supported database paths stay covered.

The deployment boundary is deliberately one decision worker and one proxy
worker. Policy state and the rate limiter are process-local; horizontal
replication is out of scope and will be documented as unsafe.

## Configuration and route identity

One proxy config represents one immutable customer route. Its identity is the
ordered model-arm list, policy type and parameters, seed, and reward exchange
rate `usd_per_quality_point` (`U`). The implemented reward remains
`clamp(quality - cost / U, 0, 1)`; equivalently, the multiplier on cost is
`lambda = 1 / U`.

The decision API stores the optional reward configuration inside the existing
`policy_config` JSON document, avoiding a schema change while making the
tradeoff queryable. Provisioning creates only after a lookup returns `404`.
An existing route is reused only on an exact ordered match. A mismatch never
mutates a learned posterior: readiness fails with guidance to choose a new,
versioned route name, while customer chat still fails open to the configured
default model.

All configuration models reject unknown keys. Secret values are never allowed
in config files; only environment-variable names may appear.

## Security and rate limiting

Three independent Bearer credentials are supplied through environment
variables:

- a client credential for chat and feedback;
- an admin credential for request logs and summaries;
- an internal credential for every decision API endpoint except health.

Missing configured credentials fail service startup. Comparisons use constant
time. The proxy never forwards the client credential upstream; it continues to
replace Authorization with the selected model's server-side key.

Authenticated chat calls pass through a process-local token bucket configured
by requests per minute and burst size. Rejection is an intentional `429` with
an OpenAI-shaped error and `Retry-After`, not a failure of the fail-open
dependency rule. Health and readiness are not rate limited.

## Retention

Startup sweeps remove proxy request logs older than the configured number of
days. Decision sweeps delete old rewards before their decisions, then remove
old policy snapshots while preserving the newest snapshot for every route.
Routes and the newest posterior always remain. Retention is configurable and
bounds the history available to OPE; it does not alter current policy state.

## PostgreSQL and operations

Both service packages declare psycopg 3 and normalize legacy/bare PostgreSQL
SQLAlchemy URLs to the explicit `postgresql+psycopg://` dialect. Portable JSON
columns use PostgreSQL JSONB and timestamps are timezone-aware on PostgreSQL.
SQLite remains the zero-setup default.

A Compose file supplies local PostgreSQL. The M5 verification script owns a
unique disposable Compose project, supervises the real decision API, mock
upstream, and proxy processes, and removes its volume in `finally`. It uses
only loopback URLs and mock keys and emits a secret-free artifact under
`docs/`.

## Public flows

`python -m llm_proxy.onboard --config <path>` validates configuration,
provisions or verifies the immutable route, and prints safe next steps without
reading or displaying provider-key values. `/healthz` remains liveness;
`/readyz` reports whether the proxy has a compatible decision route.

The live M5 acceptance checks PostgreSQL persistence/restart, onboarding,
authenticated non-streaming and streaming proxying, role separation, rate
limiting, mocked judge reward attribution, decision outage fail-open, and the
single-reward invariant.

## Invariants

- Every decision retains a strictly positive logged propensity.
- A proxy reward is attempted only when `decision_id` exists and
  `shadow_model == served_model`; the first explicit-feedback/judge reward wins.
- Decision and judge failures never break or delay upstream customer traffic;
  upstream errors pass through verbatim; client Authorization never reaches an
  upstream.
- Every composite reward is clamped to `[0, 1]` before reaching `beta_ts`.
- No real provider endpoint is called by tests, quickstarts, or M5 verification.

## Deferred

Real provider/judge dogfood, an external customer, distributed rate limiting,
multi-replica policy coordination, multi-tenant authorization, migrations for
pre-existing production schemas, and native Anthropic/Google adapters remain
outside this offline scope.
