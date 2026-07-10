"""FastAPI decision service.

Policies live in memory per route (guarded by a lock); every decision and
reward is persisted, and every reward update writes a versioned policy
snapshot — so a process restart resumes from the last posterior, and the
decision log (with propensities) feeds off-policy evaluation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from bandit_core import BanditPolicy
from bandit_core.ope import LoggedData, doubly_robust, ips, snips
from decision_api import schemas
from decision_api.db import (
    DecisionRecord,
    PolicySnapshot,
    RewardRecord,
    Route,
    make_session_factory,
)
from decision_api.policies import apply_state, build_policy
from decision_api.security import bearer_auth


@dataclass
class RouteRuntime:
    policy: BanditPolicy
    version: int
    lock: threading.Lock = field(default_factory=threading.Lock)


def create_app(database_url: str, api_key: str | None = None) -> FastAPI:
    session_factory = make_session_factory(database_url)
    runtimes: dict[str, RouteRuntime] = {}
    registry_lock = threading.Lock()

    app = FastAPI(
        title="decision-api",
        version="0.1.0",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )
    require_api_key = bearer_auth(api_key)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    def get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    def load_route(session: Session, route_id: str) -> Route:
        route = session.get(Route, route_id)
        if route is None:
            raise HTTPException(status_code=404, detail=f"route {route_id} not found")
        return route

    def get_runtime(session: Session, route: Route) -> RouteRuntime:
        with registry_lock:
            runtime = runtimes.get(route.id)
            if runtime is None:
                policy = build_policy(route.policy_config, len(route.arms), route.seed)
                snapshot = session.execute(
                    select(PolicySnapshot)
                    .where(PolicySnapshot.route_id == route.id)
                    .order_by(PolicySnapshot.version.desc())
                    .limit(1)
                ).scalar_one_or_none()
                version = 0
                if snapshot is not None:
                    apply_state(policy, snapshot.state)
                    version = snapshot.version
                runtime = RouteRuntime(policy=policy, version=version)
                runtimes[route.id] = runtime
            return runtime

    @app.post(
        "/routes",
        response_model=schemas.RouteOut,
        status_code=201,
        dependencies=[Depends(require_api_key)],
    )
    def create_route(
        body: schemas.RouteCreate, session: Session = Depends(get_session)
    ) -> schemas.RouteOut:
        try:
            build_policy(body.policy.model_dump(), len(body.arms), body.seed)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        policy_config = body.policy.model_dump()
        if body.reward is not None:
            policy_config["reward"] = body.reward.model_dump()
        route = Route(
            name=body.name,
            arms=body.arms,
            policy_config=policy_config,
            seed=body.seed,
        )
        session.add(route)
        try:
            session.commit()
        except IntegrityError:
            raise HTTPException(status_code=409, detail=f"route name {body.name!r} exists")
        return schemas.RouteOut(
            id=route.id,
            name=route.name,
            arms=route.arms,
            policy=body.policy,
            reward=body.reward,
            seed=route.seed,
        )

    @app.get(
        "/routes/by-name/{name}",
        response_model=schemas.RouteOut,
        dependencies=[Depends(require_api_key)],
    )
    def route_by_name(name: str, session: Session = Depends(get_session)) -> schemas.RouteOut:
        route = session.execute(
            select(Route).where(Route.name == name)
        ).scalar_one_or_none()
        if route is None:
            raise HTTPException(status_code=404, detail=f"route named {name!r} not found")
        return schemas.RouteOut(
            id=route.id,
            name=route.name,
            arms=route.arms,
            policy=schemas.PolicySpec(
                type=route.policy_config["type"],
                params=route.policy_config.get("params"),
            ),
            reward=(
                schemas.RewardSpec(**route.policy_config["reward"])
                if route.policy_config.get("reward") is not None
                else None
            ),
            seed=route.seed,
        )

    @app.post(
        "/routes/{route_id}/decide",
        response_model=schemas.DecisionOut,
        dependencies=[Depends(require_api_key)],
    )
    def decide(
        route_id: str,
        body: schemas.DecideRequest,
        session: Session = Depends(get_session),
    ) -> schemas.DecisionOut:
        route = load_route(session, route_id)
        runtime = get_runtime(session, route)
        with runtime.lock:
            decision = runtime.policy.choose()
            version = runtime.version
        record = DecisionRecord(
            route_id=route.id,
            context=body.context,
            arm_index=decision.arm,
            arm_name=route.arms[decision.arm],
            propensity=decision.propensity,
            policy_version=version,
        )
        session.add(record)
        session.commit()
        return schemas.DecisionOut(
            decision_id=record.id,
            arm_index=decision.arm,
            arm_name=record.arm_name,
            propensity=decision.propensity,
        )

    @app.post(
        "/rewards",
        response_model=schemas.RewardOut,
        dependencies=[Depends(require_api_key)],
    )
    def reward(
        body: schemas.RewardRequest, session: Session = Depends(get_session)
    ) -> schemas.RewardOut:
        decision = session.get(DecisionRecord, body.decision_id)
        if decision is None:
            raise HTTPException(
                status_code=404, detail=f"decision {body.decision_id} not found"
            )
        route = load_route(session, decision.route_id)
        runtime = get_runtime(session, route)
        with runtime.lock:
            try:
                runtime.policy.update(decision.arm_index, body.value)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            runtime.version += 1
            version = runtime.version
            state = runtime.policy.state_dict()
        session.add(
            RewardRecord(decision_id=decision.id, component=body.component, value=body.value)
        )
        session.add(PolicySnapshot(route_id=route.id, version=version, state=state))
        session.commit()
        return schemas.RewardOut(ok=True, policy_version=version)

    @app.get(
        "/routes/{route_id}/state",
        response_model=schemas.StateOut,
        dependencies=[Depends(require_api_key)],
    )
    def state(route_id: str, session: Session = Depends(get_session)) -> schemas.StateOut:
        route = load_route(session, route_id)
        runtime = get_runtime(session, route)
        n_decisions = session.scalar(
            select(func.count())
            .select_from(DecisionRecord)
            .where(DecisionRecord.route_id == route.id)
        )
        n_rewards = session.scalar(
            select(func.count())
            .select_from(RewardRecord)
            .join(DecisionRecord, RewardRecord.decision_id == DecisionRecord.id)
            .where(DecisionRecord.route_id == route.id)
        )
        with runtime.lock:
            return schemas.StateOut(
                route_id=route.id,
                policy_version=runtime.version,
                n_decisions=n_decisions,
                n_rewards=n_rewards,
                state=runtime.policy.state_dict(),
            )

    @app.post(
        "/routes/{route_id}/ope",
        response_model=schemas.OPEOut,
        dependencies=[Depends(require_api_key)],
    )
    def ope_report(
        route_id: str, body: schemas.OPERequest, session: Session = Depends(get_session)
    ) -> schemas.OPEOut:
        route = load_route(session, route_id)
        if len(body.target_probs) != len(route.arms):
            raise HTTPException(
                status_code=422,
                detail=f"target_probs must have {len(route.arms)} entries",
            )
        rows = session.execute(
            select(
                DecisionRecord.arm_index,
                DecisionRecord.propensity,
                func.avg(RewardRecord.value),
            )
            .join(RewardRecord, RewardRecord.decision_id == DecisionRecord.id)
            .where(DecisionRecord.route_id == route.id)
            .group_by(DecisionRecord.id, DecisionRecord.arm_index, DecisionRecord.propensity)
        ).all()
        if not rows:
            raise HTTPException(status_code=409, detail="no rewarded decisions logged yet")
        log = LoggedData(
            arms=np.array([r[0] for r in rows], dtype=int),
            propensities=np.array([r[1] for r in rows], dtype=float),
            rewards=np.array([r[2] for r in rows], dtype=float),
        )
        target = np.asarray(body.target_probs, dtype=float)
        try:
            return schemas.OPEOut(
                n=len(rows),
                ips=ips(log, target),
                snips=snips(log, target),
                doubly_robust=doubly_robust(log, target),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    return app
