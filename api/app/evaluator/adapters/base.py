from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.schemas import ImageRecord, PredictionSummary


class InferenceAdapter(ABC):
    name = "base"

    @abstractmethod
    def predict(self, image: ImageRecord) -> PredictionSummary:
        raise NotImplementedError

    def raw_prediction(self, prediction: PredictionSummary) -> Any:
        return prediction.model_dump(mode="json")