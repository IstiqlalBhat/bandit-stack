from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelSpec(StrictModel):
    name: str
    base_url: str  # OpenAI-compatible root, e.g. "https://api.openai.com/v1"
    api_key_env: str | None = None  # env var holding the key; never the key itself
    input_usd_per_mtok: float = Field(ge=0)
    output_usd_per_mtok: float = Field(ge=0)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class RoutePolicyConfig(StrictModel):
    type: Literal["beta_ts", "gaussian_ts", "epsilon_greedy"] = "beta_ts"
    params: dict | None = Field(
        default_factory=lambda: {"propensity_samples": 64}
    )


class DecisionAPIConfig(StrictModel):
    base_url: str
    api_key_env: str
    route_name: str = Field(
        min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    policy: RoutePolicyConfig = Field(default_factory=RoutePolicyConfig)
    seed: int | None = None
    # tight budget: the shadow call must never meaningfully delay a request
    timeout_s: float = 0.25

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env)


class RewardConfig(StrictModel):
    # dollars of request cost that offset one full quality point
    usd_per_quality_point: float = Field(default=0.01, gt=0)


class AuthConfig(StrictModel):
    client_api_key_env: str
    admin_api_key_env: str

    @property
    def client_api_key(self) -> str | None:
        return os.environ.get(self.client_api_key_env)

    @property
    def admin_api_key(self) -> str | None:
        return os.environ.get(self.admin_api_key_env)


class RateLimitConfig(StrictModel):
    requests_per_minute: int = Field(default=600, gt=0)
    burst: int = Field(default=100, gt=0)


class JudgeConfig(StrictModel):
    base_url: str  # OpenAI-compatible root for the judge model
    model: str
    api_key_env: str | None = None
    sample_rate: float = Field(default=0.1, ge=0, le=1)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class ProxyConfig(StrictModel):
    mode: Literal["shadow", "assisted"] = "shadow"
    default_model: str
    models: list[ModelSpec] = Field(min_length=2)
    decision_api: DecisionAPIConfig
    database_url: str
    auth: AuthConfig
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    retention_days: int = Field(default=30, gt=0)
    reward: RewardConfig = RewardConfig()
    judge: JudgeConfig | None = None

    @model_validator(mode="after")
    def _check_models(self) -> "ProxyConfig":
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError("model names must be unique")
        if self.default_model not in names:
            raise ValueError(
                f"default_model {self.default_model!r} is not one of {names}"
            )
        return self

    def model_by_name(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)


def load_config(path: str | Path) -> ProxyConfig:
    return ProxyConfig.model_validate(json.loads(Path(path).read_text()))
