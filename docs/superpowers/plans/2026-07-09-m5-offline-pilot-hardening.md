# M5 Offline Pilot Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. The user's no-commit instruction overrides commit steps.

**Goal:** Deliver a mock-only, PostgreSQL-backed, secured and reproducible
single-tenant pilot rehearsal while keeping M3/M4 behavior intact.

**Architecture:** Keep one proxy config per immutable decision route. Add
small persistence, provisioning, security, rate-limit, and retention modules;
exercise their public behavior through FastAPI and real-process tests.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, psycopg 3, Pydantic 2,
httpx, pytest, Docker Compose, PostgreSQL 17.

## Global Constraints

- Never call a real LLM provider or spend money; every upstream and judge is
  local or `httpx.MockTransport`.
- Never print secret values. Config files contain environment-variable names
  only.
- Never commit or push.
- Every decision carries a propensity.
- Reward attribution requires a decision ID and identical picked/served arm;
  at most one reward per request.
- Dependency failures are fail-open; upstream failures pass through; client
  Authorization is not forwarded.
- Rewards sent to `beta_ts` stay in `[0, 1]`.
- Each production behavior is preceded by a focused failing test and followed
  by focused and full-suite green runs.
- Final verification runs `uv run pytest`, M3, M4, and the new M5 live script.

---

### Task 1: PostgreSQL persistence path

**Files:**
- Modify: `services/decision_api/pyproject.toml`
- Modify: `services/llm_proxy/pyproject.toml`
- Modify: `services/decision_api/src/decision_api/db.py`
- Modify: `services/llm_proxy/src/llm_proxy/db.py`
- Create: `services/decision_api/tests/test_db.py`
- Create: `services/decision_api/tests/test_postgres.py`
- Modify: `uv.lock`

**Interfaces:**
- Produce `normalize_database_url(url: str) -> str` in both persistence modules.
- Preserve `make_session_factory(database_url: str) -> sessionmaker`.

- [ ] Add URL normalization tests for `postgres://`, bare `postgresql://`,
  explicit dialect URLs, credentials, ports and query strings; run them and
  confirm missing behavior is red.
- [ ] Implement only the normalization and PostgreSQL type variants; rerun the
  focused tests green.
- [ ] Add a public API round-trip test gated by `TEST_POSTGRES_URL`; run it
  against a disposable PostgreSQL container and confirm red before psycopg is
  declared.
- [ ] Add `psycopg[binary]>=3.2` to both service packages, regenerate the lock,
  then run the PostgreSQL round trip and full suite green.

### Task 2: Immutable per-route provisioning and readiness

**Files:**
- Modify: `services/decision_api/src/decision_api/schemas.py`
- Modify: `services/decision_api/src/decision_api/app.py`
- Modify: `services/llm_proxy/src/llm_proxy/config.py`
- Create: `services/llm_proxy/src/llm_proxy/provisioning.py`
- Modify: `services/llm_proxy/src/llm_proxy/shadow.py`
- Modify: `services/llm_proxy/src/llm_proxy/app.py`
- Create: `services/llm_proxy/tests/test_provisioning.py`
- Modify: `services/decision_api/tests/test_api.py`
- Modify: `services/llm_proxy/tests/test_proxy.py`

**Interfaces:**
- Decision `RouteCreate`/`RouteOut` gain optional `reward` with positive
  `usd_per_quality_point`.
- Proxy `DecisionAPIConfig` gains typed `policy`, `seed`, and internal-key env.
- `provision_route(client, config, ordered_arms, reward)` returns the compatible
  route and whether it was created, otherwise raises a safe provisioning error.

- [ ] Add one failing decision-route validation/reward round-trip test; add the
  minimal schema/storage behavior; rerun green.
- [ ] Add one failing provisioning behavior at a time: exact reuse, create only
  on 404, ordered-arm mismatch, policy/seed/reward mismatch, 409 race re-fetch,
  and non-404 lookup failure; implement minimally after each red.
- [ ] Add failing readiness and outage tests; prepare best-effort during proxy
  lifespan, expose `/readyz`, and retain chat fail-open; rerun green.

### Task 3: Authentication and rate limiting

**Files:**
- Create: `services/decision_api/src/decision_api/security.py`
- Modify: `services/decision_api/src/decision_api/app.py`
- Modify: `services/decision_api/src/decision_api/main.py`
- Create: `services/llm_proxy/src/llm_proxy/security.py`
- Create: `services/llm_proxy/src/llm_proxy/rate_limit.py`
- Modify: `services/llm_proxy/src/llm_proxy/config.py`
- Modify: `services/llm_proxy/src/llm_proxy/app.py`
- Create: `services/decision_api/tests/test_security.py`
- Create: `services/llm_proxy/tests/test_security.py`
- Modify: M3/M4 scripts and existing test helpers to send mock credentials.

**Interfaces:**
- `create_app(..., api_key: str | None = None, ...)` protects decision routes
  when configured; `/healthz` stays public.
- Proxy config has separate client/admin key env names and decision internal-key
  env name.
- Chat rate limiter returns `429`, an OpenAI error body, and `Retry-After`.

- [ ] Add failing decision health/auth tests, then constant-time Bearer auth and
  a health endpoint; rerun green.
- [ ] Add failing proxy role-separation and no-upstream-on-rejection tests, then
  client/admin dependencies; rerun green.
- [ ] Add one failing burst/refill public endpoint test, implement a
  process-local token bucket, then rerun focused and full tests green.
- [ ] Update M3/M4 drivers only after tests define the new contract.

### Task 4: Bounded retention and transaction consistency

**Files:**
- Modify: `services/decision_api/src/decision_api/db.py`
- Modify: `services/decision_api/src/decision_api/app.py`
- Modify: `services/decision_api/src/decision_api/main.py`
- Modify: `services/llm_proxy/src/llm_proxy/db.py`
- Modify: `services/llm_proxy/src/llm_proxy/config.py`
- Modify: `services/llm_proxy/src/llm_proxy/app.py`
- Create: `services/decision_api/tests/test_retention.py`
- Create: `services/llm_proxy/tests/test_retention.py`

**Interfaces:**
- `prune_history(session_factory, cutoff)` removes expired history and returns
  per-table counts while preserving each route's newest snapshot.
- `retention_days` is positive and configurable; decision main reads
  `DECISION_API_RETENTION_DAYS`.

- [ ] Add a failing proxy retention test, implement startup pruning, rerun green.
- [ ] Add a failing decision retention/restart test, implement ordered deletes
  plus newest-snapshot preservation, rerun green.
- [ ] Add a failing reward-commit rollback regression test, then keep runtime
  state/version synchronized with the transaction; rerun green.

### Task 5: Onboarding, operations, and live acceptance

**Files:**
- Create: `services/llm_proxy/src/llm_proxy/onboard.py`
- Create: `services/llm_proxy/tests/test_onboard.py`
- Create: `compose.yaml`
- Create: `.env.example`
- Create: `configs/llm_proxy.offline.json`
- Modify: `configs/llm_proxy.example.json`
- Create: `apps/reports/m5_e2e.py`
- Create/update: `docs/m5-offline-verification.txt`
- Create: `docs/m5-pilot-operations.md`
- Modify: `.gitignore`
- Modify: `README.md`

**Interfaces:**
- `uv run python -m llm_proxy.onboard --config <path>` provisions/verifies a
  route and never prints credential values.
- `uv run python apps/reports/m5_e2e.py` owns all child processes and its
  disposable PostgreSQL Compose project, exits nonzero on any failed check,
  and writes a sanitized verification artifact.

- [ ] Add failing onboarding tests for created, reused, incompatible and down
  routes plus output-secret scanning; implement CLI and rerun green.
- [ ] Write M5 live checks first and run them red against the missing operations
  setup.
- [ ] Add Compose/offline config/process supervision until every live check is
  green; capture the artifact without keys.
- [ ] Update the external quickstart, status and limitations. Keep overall M5
  unchecked because real traffic/customer work requires the user.

### Task 6: Final verification and review

- [ ] Run `uv run pytest` fresh and read the complete result.
- [ ] Run `uv run python apps/reports/m3_e2e.py` and require all 14 checks.
- [ ] Run `uv run python apps/reports/m4_e2e.py` and require all 9 checks.
- [ ] Run `uv run python apps/reports/m5_e2e.py` and verify the generated
  artifact contains no key values or external provider calls.
- [ ] Re-read every global invariant against the diff.
- [ ] Dispatch an independent whole-change review; fix every Critical/Important
  finding through another red-green cycle and rerun all affected verification.
