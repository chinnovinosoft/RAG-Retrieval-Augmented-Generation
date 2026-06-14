from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────
    openai_api_key: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"

    # ── MCP connection ───────────────────────────────────────────
    mcp_server_url: str = "http://localhost:8000/mcp"
    mcp_timeout: int = 30

    # ── Agent behaviour ──────────────────────────────────────────
    max_iterations: int = 10
    max_parallel_searches: int = 5
    verbose: bool = True

    # ── Observability ────────────────────────────────────────────
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache(maxsize=1)
def get_settings() -> AgentSettings:
    return AgentSettings()


settings = get_settings()
