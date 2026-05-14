from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.evaluator.dataset_loader import create_dataset_manifest
from app.schemas import DatasetUploadResponse, ModelRecord, ModelUploadResponse
from app.storage import Storage, new_id, save_upload, write_json

router = APIRouter(tags=["uploads"])


@router.post("/models", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)) -> ModelUploadResponse:
    if not file.filename or Path(file.filename).suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail="Upload a .pt model file")

    storage = Storage()
    model_id = new_id("model")
    model_dir = storage.model_dir(model_id)
    target = model_dir / Path(file.filename).name
    size = await save_upload(file, target)
    record = ModelRecord(
        id=model_id,
        filename=Path(file.filename).name,
        path=str(target),
        size_bytes=size,
        created_at=datetime.now(timezone.utc),
    )
    write_json(model_dir / "metadata.json", record)
    return ModelUploadResponse(model=record)


@router.post("/datasets", response_model=DatasetUploadResponse)
async def upload_dataset(
    images: list[UploadFile] = File(default=[]),
    annotation: UploadFile | None = File(default=None),
    archive: UploadFile | None = File(default=None),
) -> DatasetUploadResponse:
    storage = Storage()
    manifest = await create_dataset_manifest(images, annotation, archive, storage)
    return DatasetUploadResponse(dataset=manifest)