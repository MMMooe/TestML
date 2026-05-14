from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.evaluator.dataset_loader import load_manifest
from app.evaluator.runner import load_metrics, load_results, run_job
from app.schemas import JobCreateRequest, JobStatus, MetricsResponse, PaginatedResults
from app.storage import Storage, new_id, read_json, write_json

router = APIRouter(tags=["jobs"])


@router.post("/jobs", response_model=JobStatus)
async def create_job(request: JobCreateRequest, background_tasks: BackgroundTasks) -> JobStatus:
    storage = Storage()
    if not (storage.model_dir(request.model_id) / "metadata.json").exists():
        raise HTTPException(status_code=404, detail="Model not found")
    manifest = load_manifest(request.dataset_id, storage)

    job_id = new_id("job")
    status = JobStatus(
        id=job_id,
        model_id=request.model_id,
        dataset_id=request.dataset_id,
        adapter=request.adapter,
        job_kind=manifest.job_kind,
        state="queued",
        total=manifest.image_count,
        message="Queued",
        created_at=datetime.now(timezone.utc),
    )
    write_json(storage.job_dir(job_id) / "status.json", status)
    background_tasks.add_task(run_job, job_id, request)
    return status


@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    return _load_status(job_id)


@router.get("/jobs/{job_id}/metrics", response_model=MetricsResponse)
async def get_metrics(job_id: str) -> MetricsResponse:
    storage = Storage()
    status = _load_status(job_id)
    return load_metrics(job_id, status, storage)


@router.get("/jobs/{job_id}/results", response_model=PaginatedResults)
async def get_results(
    job_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    q: str | None = None,
    class_label: str | None = None,
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
    correctness: str | None = Query(default=None, pattern="^(correct|incorrect|error)$"),
) -> PaginatedResults:
    storage = Storage()
    status = _load_status(job_id)
    results = load_results(job_id, storage)
    filtered = []
    for item in results:
        if q and q.lower() not in item.filename.lower():
            continue
        if class_label and item.prediction.label != class_label:
            continue
        if min_confidence is not None and (item.prediction.confidence is None or item.prediction.confidence < min_confidence):
            continue
        if correctness == "correct" and item.is_correct is not True:
            continue
        if correctness == "incorrect" and item.is_correct is not False:
            continue
        if correctness == "error" and item.error is None:
            continue
        filtered.append(item)

    start = (page - 1) * page_size
    end = start + page_size
    return PaginatedResults(job_id=status.id, page=page, page_size=page_size, total=len(filtered), items=filtered[start:end])


def _load_status(job_id: str) -> JobStatus:
    storage = Storage()
    path = storage.job_dir(job_id) / "status.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus.model_validate(read_json(path))