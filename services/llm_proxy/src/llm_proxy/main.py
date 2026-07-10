"""Server entrypoint:
LLM_PROXY_CONFIG=configs/llm_proxy.json uv run uvicorn llm_proxy.main:app --port 8200
"""

import os

from llm_proxy.app import create_app
from llm_proxy.config import load_config

app = create_app(load_config(os.environ["LLM_PROXY_CONFIG"]))
