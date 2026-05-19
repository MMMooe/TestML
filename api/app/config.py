from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_mode: Literal["production-cuda"] = Field(default="production-cuda", validation_alias="APP_MODE")
    storage_dir: Path = Field(
        default=Path(__file__).resolve().parents[2] / "storage",
        validation_alias="APP_STORAGE_DIR",
    )
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        validation_alias="APP_CORS_ORIGINS",
    )
    max_upload_mb: int = Field(default=4096, validation_alias="APP_MAX_UPLOAD_MB")
    require_tensorrt: bool = Field(default=True, validation_alias="APP_REQUIRE_TENSORRT")

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def is_production_cuda(self) -> bool:
        return self.app_mode == "production-cuda"

    @property
    def parsed_cors_origins(self) -> list[str]:
        origins: list[str] = []
        for raw_origin in self.cors_origins.split(","):
            origin = raw_origin.strip()
            if not origin:
                continue
            if origin != "*":
                origin = origin.rstrip("/")
            if origin not in origins:
                origins.append(origin)
        return origins


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings