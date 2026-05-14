from __future__ import annotations

from pathlib import Path
from typing import Any


FILENAME_KEYS = ("filename", "file_name", "image", "image_path", "path", "name")
LABEL_KEYS = ("label", "class", "category", "category_name", "target", "class_name")
LABEL_ID_KEYS = ("label_id", "class_id", "category_id", "target_id")


def build_annotation_index(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        coco_index = _build_coco_index(payload)
        if coco_index:
            return coco_index

        for key in ("items", "records", "annotations", "images", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                indexed = _index_records(value)
                if indexed:
                    return indexed

        filename_like_keys = [key for key in payload.keys() if Path(str(key)).suffix]
        if filename_like_keys:
            return {Path(str(key)).name: value for key, value in payload.items()}

    if isinstance(payload, list):
        return _index_records(payload)

    return {}


def find_annotation_for_filename(index: dict[str, Any], filename: str) -> Any | None:
    basename = Path(filename).name
    candidates = [filename, basename]
    for candidate in candidates:
        if candidate in index:
            return index[candidate]
    for key, value in index.items():
        if Path(str(key)).name == basename:
            return value
    return None


def extract_label(annotation: Any | None) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, str | int | float):
        return str(annotation)
    if isinstance(annotation, dict):
        for key in LABEL_KEYS:
            value = annotation.get(key)
            if value is not None:
                return str(value)
        for key in LABEL_ID_KEYS:
            value = annotation.get(key)
            if value is not None:
                return f"class_{value}"
        nested = annotation.get("annotation") or annotation.get("ground_truth")
        if nested is not None:
            return extract_label(nested)
    return None


def is_prediction_correct(predicted_label: str | None, annotation: Any | None) -> bool | None:
    target = extract_label(annotation)
    if target is None or predicted_label is None:
        return None
    return str(predicted_label) == str(target)


def _index_records(records: list[Any]) -> dict[str, Any]:
    indexed: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        filename = _record_filename(record)
        if filename:
            indexed[Path(filename).name] = record
    return indexed


def _record_filename(record: dict[str, Any]) -> str | None:
    for key in FILENAME_KEYS:
        value = record.get(key)
        if value:
            return str(value)
    return None


def _build_coco_index(payload: dict[str, Any]) -> dict[str, Any]:
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        return {}

    image_by_id: dict[Any, dict[str, Any]] = {}
    for image in images:
        if isinstance(image, dict) and "id" in image:
            image_by_id[image["id"]] = image

    annotations_by_image: dict[Any, list[Any]] = {}
    for annotation in annotations:
        if isinstance(annotation, dict) and "image_id" in annotation:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    indexed: dict[str, Any] = {}
    for image_id, image in image_by_id.items():
        filename = _record_filename(image)
        if filename:
            indexed[Path(filename).name] = {
                "image": image,
                "annotations": annotations_by_image.get(image_id, []),
            }
    return indexed