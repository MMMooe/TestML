from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from fastapi import HTTPException, UploadFile

from app.evaluator.annotations import build_annotation_index, find_annotation_for_filename
from app.schemas import DatasetManifest, ImageRecord
from app.storage import Storage, new_id, save_upload, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


async def create_dataset_manifest(
    images: list[UploadFile] | None,
    annotation: UploadFile | None,
    archive: UploadFile | None,
    storage: Storage,
) -> DatasetManifest:
    dataset_id = new_id("dataset")
    dataset_dir = storage.dataset_dir(dataset_id)
    raw_dir = dataset_dir / "raw"
    image_dir = dataset_dir / "images"
    annotation_dir = dataset_dir / "annotations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    candidate_images: list[Path] = []
    annotation_payload: Any | None = None
    annotation_filename: str | None = None

    if archive is not None and archive.filename:
        if not archive.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Archive upload must be a .zip file")
        archive_path = raw_dir / "upload.zip"
        await save_upload(archive, archive_path)
        extracted_dir = raw_dir / "extracted"
        safe_extract_zip(archive_path, extracted_dir)
        candidate_images.extend(find_images(extracted_dir))
        json_files = sorted(path for path in extracted_dir.rglob("*") if path.suffix.lower() == ".json")
        if json_files:
            annotation_path = json_files[0]
            annotation_filename = annotation_path.name
            annotation_payload = load_json(annotation_path)
            if len(json_files) > 1:
                warnings.append(f"Multiple JSON files found in archive; using {annotation_filename}")

    for upload in images or []:
        if not upload.filename:
            continue
        if Path(upload.filename).suffix.lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {upload.filename}")
        safe_name = safe_filename(upload.filename)
        target = raw_dir / "multipart" / safe_name
        await save_upload(upload, target)
        candidate_images.append(target)

    if annotation is not None and annotation.filename:
        if Path(annotation.filename).suffix.lower() != ".json":
            raise HTTPException(status_code=400, detail="Annotation upload must be a JSON file")
        annotation_filename = safe_filename(annotation.filename)
        annotation_path = annotation_dir / annotation_filename
        await save_upload(annotation, annotation_path)
        annotation_payload = load_json(annotation_path)

    if not candidate_images:
        raise HTTPException(status_code=400, detail="Upload at least one image file or a zip containing images")

    annotation_index = build_annotation_index(annotation_payload) if annotation_payload is not None else {}
    normalized_images: list[ImageRecord] = []
    for image_path in sorted(candidate_images, key=lambda path: path.name):
        image_id = new_id("image")
        filename = safe_filename(image_path.name)
        target = image_dir / f"{image_id}_{filename}"
        shutil.copyfile(image_path, target)
        linked_annotation = find_annotation_for_filename(annotation_index, image_path.name) if annotation_index else None
        normalized_images.append(
            ImageRecord(
                id=image_id,
                filename=image_path.name,
                path=str(target),
                asset_path=storage.asset_path(target),
                annotation=linked_annotation,
            )
        )

    has_annotations = annotation_payload is not None
    if has_annotations and annotation_index and all(image.annotation is None for image in normalized_images):
        warnings.append("JSON was provided, but no annotations matched uploaded image filenames")

    manifest = DatasetManifest(
        id=dataset_id,
        created_at=datetime.now(timezone.utc),
        image_count=len(normalized_images),
        has_annotations=has_annotations,
        job_kind="evaluation" if has_annotations else "inference",
        annotation_filename=annotation_filename,
        images=normalized_images,
        warnings=warnings,
    )
    write_json(dataset_dir / "manifest.json", manifest)
    return manifest


def load_manifest(dataset_id: str, storage: Storage) -> DatasetManifest:
    path = storage.dataset_dir(dataset_id) / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def find_images(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination_root):
                raise HTTPException(status_code=400, detail="Zip archive contains an unsafe path")
        archive.extractall(destination)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail=f"Invalid JSON annotation file: {error.msg}") from error


def safe_filename(filename: str) -> str:
    name = Path(filename).name.replace(" ", "_")
    return "".join(character for character in name if character.isalnum() or character in {".", "_", "-"}) or "file"