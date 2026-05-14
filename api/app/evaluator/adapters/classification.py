from __future__ import annotations

from app.evaluator.adapters.base import InferenceAdapter
from app.evaluator.model_loader import CudaModelRunner
from app.schemas import ImageRecord, PredictionSummary


class ClassificationAdapter(InferenceAdapter):
    name = "classification"

    def __init__(self, model_path: str) -> None:
        self.runner = CudaModelRunner(model_path)

    def predict(self, image: ImageRecord) -> PredictionSummary:
        return self.runner.predict(image.path)