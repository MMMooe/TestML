from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from app.config import Settings, get_settings


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Storage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.storage_dir.resolve()
        self.models_dir = self.root / "uploads" / "models"
        self.datasets_dir = self.root / "uploads" / "datasets"
        self.jobs_dir = self.root / "jobs"
        self.results_dir = self.root / "results"
        for directory in [self.models_dir, self.datasets_dir, self.jobs_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def model_dir(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.datasets_dir / dataset_id

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def asset_path(self, absolute_path: Path) -> str:
        relative = absolute_path.resolve().relative_to(self.root)
        return relative.as_posix()

    def resolve_asset(self, asset_path: str) -> Path:
        candidate = (self.root / asset_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("asset path escapes storage root")
        return candidate


def write_json(path: Path, payload: BaseModel | dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        data = payload.model_dump(mode="json")
    else:
        data = payload
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


async def save_upload(upload, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            output.write(chunk)
    return size