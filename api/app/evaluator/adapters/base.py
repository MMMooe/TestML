from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.schemas import ImageRecord, PredictionSummary


class InferenceAdapter(ABC):
    name = "base"
    supports_batch = False

    @abstractmethod
    def predict(self, image: ImageRecord) -> PredictionSummary:
        raise NotImplementedError

    def predict_many(self, images: list[ImageRecord]) -> list[PredictionSummary]:
        return [self.predict(image) for image in images]

    def raw_prediction(self, prediction: PredictionSummary) -> Any:
        return prediction.model_dump(mode="json")