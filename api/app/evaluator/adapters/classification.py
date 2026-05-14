from __future__ import annotations

from app.evaluator.adapters.base import InferenceAdapter
from app.evaluator.model_loader import create_model_runner
from app.schemas import ImageRecord, PredictionSummary


class ClassificationAdapter(InferenceAdapter):
    name = "classification"

    def __init__(
        self,
        model_path: str,
        plan_path: str | None = None,
        *,
        use_tensorrt: bool = True,
        allow_tensorrt_fallback: bool = False,
    ) -> None:
        self.runner = create_model_runner(
            model_path,
            plan_path,
            use_tensorrt=use_tensorrt,
            allow_tensorrt_fallback=allow_tensorrt_fallback,
        )
        self.backend = self.runner.backend
        self.supports_batch = hasattr(self.runner, "predict_batch")

    def predict(self, image: ImageRecord) -> PredictionSummary:
        return self.runner.predict(image.path)

    def predict_many(self, images: list[ImageRecord]) -> list[PredictionSummary]:
        if self.supports_batch:
            return self.runner.predict_batch([image.path for image in images])
        return super().predict_many(images)