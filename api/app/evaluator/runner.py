from __future__ import annotations

from datetime import datetime, timezone

from app.evaluator.adapters.classification import ClassificationAdapter
from app.evaluator.annotations import is_prediction_correct
from app.evaluator.dataset_loader import load_manifest
from app.schemas import JobCreateRequest, JobStatus, MetricsResponse, ResultGalleryItem
from app.storage import Storage, read_json, write_json


def run_job(job_id: str, request: JobCreateRequest) -> None:
    storage = Storage()
    status_path = storage.job_dir(job_id) / "status.json"
    status = JobStatus.model_validate(read_json(status_path))

    try:
        manifest = load_manifest(request.dataset_id, storage)
        status.state = "running"
        status.started_at = datetime.now(timezone.utc)
        status.message = "Running CUDA inference"
        status.total = manifest.image_count
        write_json(status_path, status)

        model_metadata = read_json(storage.model_dir(request.model_id) / "metadata.json")
        adapter = _select_adapter(request.adapter, model_metadata["path"])

        results: list[ResultGalleryItem] = []
        correct = 0
        comparable = 0
        failed = 0

        for index, image in enumerate(manifest.images, start=1):
            error = None
            prediction = None
            try:
                prediction = adapter.predict(image)
            except Exception as exc:  # keep the gallery useful even when one image fails
                failed += 1
                error = str(exc)

            if prediction is None:
                from app.schemas import PredictionSummary

                prediction = PredictionSummary(raw={})

            is_correct = is_prediction_correct(prediction.label, image.annotation) if manifest.has_annotations else None
            if is_correct is not None:
                comparable += 1
                if is_correct:
                    correct += 1

            results.append(
                ResultGalleryItem(
                    id=image.id,
                    filename=image.filename,
                    image_url=f"/assets/{image.asset_path}",
                    job_kind=manifest.job_kind,
                    prediction=prediction,
                    ground_truth=image.annotation if manifest.has_annotations else None,
                    is_correct=is_correct,
                    error=error,
                )
            )

            status.processed = index
            status.progress = index / max(manifest.image_count, 1)
            status.message = f"Processed {index} of {manifest.image_count} images"
            write_json(status_path, status)

        write_json(storage.results_dir / job_id / "results.json", [result.model_dump(mode="json") for result in results])

        if manifest.has_annotations:
            metrics = {
                "image_count": manifest.image_count,
                "comparable_count": comparable,
                "correct_count": correct,
                "error_count": failed,
                "accuracy": correct / comparable if comparable else None,
            }
            write_json(storage.results_dir / job_id / "metrics.json", metrics)

        status.state = "completed"
        status.progress = 1.0
        status.completed_at = datetime.now(timezone.utc)
        status.message = "Job completed"
        write_json(status_path, status)
    except Exception as exc:
        status.state = "failed"
        status.error = str(exc)
        status.completed_at = datetime.now(timezone.utc)
        status.message = "Job failed"
        write_json(status_path, status)


def _select_adapter(adapter_name: str, model_path: str):
    if adapter_name in {"classification", "custom/no-metrics", "detection"}:
        return ClassificationAdapter(model_path)
    raise ValueError(f"Unsupported adapter: {adapter_name}")


def load_results(job_id: str, storage: Storage) -> list[ResultGalleryItem]:
    path = storage.results_dir / job_id / "results.json"
    if not path.exists():
        return []
    return [ResultGalleryItem.model_validate(item) for item in read_json(path)]


def load_metrics(job_id: str, status: JobStatus, storage: Storage) -> MetricsResponse:
    path = storage.results_dir / job_id / "metrics.json"
    if not path.exists():
        return MetricsResponse(
            job_id=job_id,
            available=False,
            job_kind=status.job_kind,
            metrics={},
            message="No annotations provided; metrics are not available for inference-only jobs"
            if status.job_kind == "inference"
            else "Metrics are not available yet",
        )
    return MetricsResponse(job_id=job_id, available=True, job_kind=status.job_kind, metrics=read_json(path))