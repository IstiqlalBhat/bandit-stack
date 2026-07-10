"""Shadow routing: ask the decision engine what it WOULD do, never let that
call break or delay a customer request. Every failure path returns None."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from llm_proxy.config import DecisionAPIConfig, RewardConfig
from llm_proxy.provisioning import provision_route


@dataclass(frozen=True)
class ShadowPick:
    model: str
    decision_id: str
    propensity: float


class ShadowRouter:
    def __init__(
        self,
        client: httpx.AsyncClient,
        cfg: DecisionAPIConfig,
        arm_names: list[str],
        reward: RewardConfig,
    ) -> None:
        self._client = client
        self._cfg = cfg
        self._arms = arm_names
        self._reward = reward
        self._route_id: str | None = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def prepare(self) -> None:
        try:
            route = await provision_route(
                self._client, self._cfg, self._arms, self._reward
            )
        except Exception:
            self._route_id = None
            self._ready = False
            return
        self._route_id = route.route_id
        self._ready = True

    async def decide(self, context: dict) -> ShadowPick | None:
        try:
            for attempt in range(2):
                if self._route_id is None:
                    await self.prepare()
                if self._route_id is None:
                    return None
                resp = await self._client.post(
                    f"/routes/{self._route_id}/decide", json={"context": context}
                )
                if resp.status_code == 404 and attempt == 0:
                    self._route_id = None
                    self._ready = False
                    continue
                resp.raise_for_status()
                d = resp.json()
                return ShadowPick(
                    model=d["arm_name"],
                    decision_id=d["decision_id"],
                    propensity=d["propensity"],
                )
        except Exception:
            return None  # fail open: shadow is best-effort by design
        return None
