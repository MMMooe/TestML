from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

from app.evaluator.adapters.classification import ClassificationAdapter
from app.evaluator.annotations import is_prediction_correct
from app.evaluator.dataset_loader import load_manifest
from app.schemas import JobCreateRequest, JobStatus, MetricsResponse, ResultGalleryItem
from app.storage import Storage, read_json, write_json


def run_job(job_id: str, request: JobCreateRequest) -> None:
    storage = Storage()
    status_path = storage.job_dir(job_id) / "status.json"
    status = JobStatus.model_validate(read_json(status_path))
    started = perf_counter()

    try:
        manifest = load_manifest(request.dataset_id, storage)
        status.state = "running"
        status.started_at = datetime.now(timezone.utc)
        status.message = "Preparing inference backend"
        status.total = manifest.image_count
        write_json(status_path, status)

        model_metadata = read_json(storage.model_dir(request.model_id) / "metadata.json")
        adapter = _select_adapter(
            request.adapter,
            model_metadata["path"],
            model_metadata.get("plan_path"),
            use_tensorrt=request.use_tensorrt,
            allow_tensorrt_fallback=request.allow_tensorrt_fallback,
        )
        status.inference_backend = getattr(adapter, "backend", None)
        effective_batch_size = request.batch_size if getattr(adapter, "supports_batch", False) else 1
        batch_detail = f"batch size {effective_batch_size}" if effective_batch_size > 1 else "single-image batches"
        status.message = f"Running {status.inference_backend or 'CUDA'} inference with {batch_detail}"
        write_json(status_path, status)

        results: list[ResultGalleryItem] = []
        correct = 0
        comparable = 0
        failed = 0

        for batch_start, batch_images in _batched(manifest.images, effective_batch_size):
            batch_predictions = _predict_batch(adapter, batch_images)

            for offset, image in enumerate(batch_images):
                prediction, error = batch_predictions[offset]
                if error is not None:
                    failed += 1

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

            status.processed = min(batch_start + len(batch_images), manifest.image_count)
            status.progress = status.processed / max(manifest.image_count, 1)
            status.message = f"Processed {status.processed} of {manifest.image_count} images with {batch_detail}"
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
        elapsed = max(perf_counter() - started, 0.001)
        status.message = f"Job completed in {elapsed:.1f}s ({manifest.image_count / elapsed:.2f} images/s)"
        write_json(status_path, status)
    except Exception as exc:
        status.state = "failed"
        status.error = str(exc)
        status.completed_at = datetime.now(timezone.utc)
        status.message = "Job failed"
        write_json(status_path, status)


def _select_adapter(
    adapter_name: str,
    model_path: str,
    plan_path: str | None = None,
    *,
    use_tensorrt: bool = True,
    allow_tensorrt_fallback: bool = False,
):
    if adapter_name in {"classification", "custom/no-metrics", "detection"}:
        return ClassificationAdapter(
            model_path,
            plan_path=plan_path,
            use_tensorrt=use_tensorrt,
            allow_tensorrt_fallback=allow_tensorrt_fallback,
        )
    raise ValueError(f"Unsupported adapter: {adapter_name}")


def _batched(items: list, batch_size: int):
    size = max(batch_size, 1)
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def _predict_batch(adapter, images):
    try:
        predictions = adapter.predict_many(images)
        if len(predictions) != len(images):
            raise RuntimeError(f"Adapter returned {len(predictions)} predictions for {len(images)} images")
        return [(prediction, None) for prediction in predictions]
    except Exception as batch_error:
        if len(images) == 1:
            return [(None, str(batch_error))]

    predictions = []
    for image in images:
        try:
            predictions.append((adapter.predict(image), None))
        except Exception as error:  # keep one failed image from failing the whole batch
            predictions.append((None, str(error)))
    return predictions


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