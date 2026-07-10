from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from decision_api.policies import POLICY_TYPES


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicySpec(StrictSchema):
    type: Literal[POLICY_TYPES]  # type: ignore[valid-type]
    params: dict | None = None


class RewardSpec(StrictSchema):
    usd_per_quality_point: float = Field(gt=0)


class RouteCreate(StrictSchema):
    name: str = Field(
        min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    arms: list[str] = Field(min_length=2)
    policy: PolicySpec
    reward: RewardSpec | None = None
    seed: int | None = None

    @field_validator("arms")
    @classmethod
    def unique_arms(cls, arms: list[str]) -> list[str]:
        if any(not arm.strip() for arm in arms):
            raise ValueError("arm names must not be blank")
        if len(set(arms)) != len(arms):
            raise ValueError("arms must be unique")
        return arms


class RouteOut(StrictSchema):
    id: str
    name: str
    arms: list[str]
    policy: PolicySpec
    reward: RewardSpec | None = None
    seed: int | None


class DecideRequest(StrictSchema):
    context: dict | None = None


class DecisionOut(StrictSchema):
    decision_id: str
    arm_index: int
    arm_name: str
    propensity: float


class RewardRequest(StrictSchema):
    decision_id: str
    value: float
    component: str = "explicit"


class RewardOut(StrictSchema):
    ok: bool
    policy_version: int


class StateOut(StrictSchema):
    route_id: str
    policy_version: int
    n_decisions: int
    n_rewards: int
    state: dict


class OPERequest(StrictSchema):
    target_probs: list[float]


class OPEOut(StrictSchema):
    n: int
    ips: float
    snips: float
    doubly_robust: float
