from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    truefoundry_api_key: str
    gateway_base_url: str
    generation_model: str
    embedding_model: str
    data_dir: Path
    index_path: Path
    embeddings_path: Path
    top_k: int
    min_similarity: float
    generation_max_tokens: int = 1200
    max_context_chars_per_source: int = 1800

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("ASKTF_DATA_DIR", "data"))
        return cls(
            truefoundry_api_key=_env("TRUEFOUNDRY_API_KEY"),
            gateway_base_url=os.getenv("TFY_GATEWAY_BASE_URL", "https://gateway.truefoundry.ai"),
            generation_model=os.getenv(
                "TFY_GENERATION_MODEL",
                os.getenv("MODEL", "openai/gpt-5.4-nano"),
            ),
            embedding_model=os.getenv("TFY_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            data_dir=data_dir,
            index_path=Path(os.getenv("ASKTF_INDEX_PATH", str(data_dir / "index.jsonl"))),
            embeddings_path=Path(
                os.getenv("ASKTF_EMBEDDINGS_PATH", str(data_dir / "embeddings.npy"))
            ),
            top_k=_env_int("ASKTF_TOP_K", 4),
            min_similarity=_env_float("ASKTF_MIN_SIMILARITY", 0.15),
            generation_max_tokens=_env_int("TFY_GENERATION_MAX_TOKENS", 1200),
            max_context_chars_per_source=_env_int("ASKTF_MAX_CONTEXT_CHARS_PER_SOURCE", 1800),
        )
