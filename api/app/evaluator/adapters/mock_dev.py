from __future__ import annotations

import hashlib

from app.evaluator.adapters.base import InferenceAdapter
from app.schemas import ImageRecord, PredictionSummary


class MockDevAdapter(InferenceAdapter):
    name = "mock-dev"

    def predict(self, image: ImageRecord) -> PredictionSummary:
        digest = hashlib.sha256(image.filename.encode("utf-8")).digest()
        label_index = digest[0] % 5
        confidence = 0.55 + (digest[1] % 40) / 100
        label = f"class_{label_index}"
        top_k = [
            {"label": label, "score": round(min(confidence, 0.99), 4), "index": label_index},
            {"label": f"class_{(label_index + 1) % 5}", "score": 0.25, "index": (label_index + 1) % 5},
        ]
        return PredictionSummary(label=label, confidence=top_k[0]["score"], top_k=top_k, raw={"mode": "dev-mock"})