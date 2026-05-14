from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JobKind = Literal["inference", "evaluation"]
JobState = Literal["queued", "running", "completed", "failed"]


class RuntimeInfo(BaseModel):
    app_mode: str
    cuda_available: bool
    selected_device: str
    torch_version: str | None = None
    cuda_version: str | None = None
    device_name: str | None = None
    tensorrt_required: bool = True
    tensorrt_available: bool = False
    tensorrt_version: str | None = None
    tensorrt_message: str = "TensorRT runtime was not checked"
    message: str


class ModelRecord(BaseModel):
    id: str
    filename: str
    path: str
    plan_filename: str | None = None
    plan_path: str | None = None
    size_bytes: int
    created_at: datetime


class ModelUploadResponse(BaseModel):
    model: ModelRecord


class ImageRecord(BaseModel):
    id: str
    filename: str
    path: str
    asset_path: str
    annotation: Any | None = None


class DatasetManifest(BaseModel):
    id: str
    created_at: datetime
    image_count: int
    has_annotations: bool
    job_kind: JobKind
    annotation_filename: str | None = None
    images: list[ImageRecord]
    warnings: list[str] = Field(default_factory=list)


class DatasetUploadResponse(BaseModel):
    dataset: DatasetManifest


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    dataset_id: str
    adapter: str = "classification"
    batch_size: int = Field(default=1, ge=1, le=128)
    confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    use_tensorrt: bool = True
    allow_tensorrt_fallback: bool = False


class JobStatus(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    model_id: str
    dataset_id: str
    adapter: str
    job_kind: JobKind
    state: JobState
    inference_backend: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    processed: int = 0
    total: int = 0
    message: str = ""
    error: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class PredictionSummary(BaseModel):
    label: str | None = None
    confidence: float | None = None
    top_k: list[dict[str, Any]] = Field(default_factory=list)
    raw: Any | None = None


class ResultGalleryItem(BaseModel):
    id: str
    filename: str
    image_url: str
    job_kind: JobKind
    prediction: PredictionSummary
    ground_truth: Any | None = None
    is_correct: bool | None = None
    error: str | None = None


class MetricsResponse(BaseModel):
    job_id: str
    available: bool
    job_kind: JobKind
    metrics: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class PaginatedResults(BaseModel):
    job_id: str
    page: int
    page_size: int
    total: int
    items: list[ResultGalleryItem]