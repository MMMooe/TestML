from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_mode: str = Field(default="dev-mock", validation_alias="APP_MODE")
    storage_dir: Path = Field(
        default=Path(__file__).resolve().parents[2] / "storage",
        validation_alias="APP_STORAGE_DIR",
    )
    cors_origins: str = Field(default="http://localhost:3000", validation_alias="APP_CORS_ORIGINS")
    max_upload_mb: int = Field(default=4096, validation_alias="APP_MAX_UPLOAD_MB")

    model_config = SettingsConfigDict(extra="ignore")

    @property
    def is_production_cuda(self) -> bool:
        return self.app_mode == "production-cuda"

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings