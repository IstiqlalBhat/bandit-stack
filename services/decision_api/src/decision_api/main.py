"""Server entrypoint: uv run uvicorn decision_api.main:app --port 8123"""

import os
from pathlib import Path

from decision_api.app import create_app
from decision_api.security import required_env_key


def _database_url() -> str:
    url = os.environ.get("DECISION_API_DB")
    if url:
        return url
    Path("data").mkdir(exist_ok=True)
    return "sqlite:///data/decision_api.db"


app = create_app(_database_url(), api_key=required_env_key("DECISION_API_KEY"))
