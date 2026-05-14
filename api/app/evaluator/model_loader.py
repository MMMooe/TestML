from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.schemas import PredictionSummary


class CudaModelRunner:
    def __init__(self, model_path: str) -> None:
        try:
            import torch
        except Exception as error:
            raise RuntimeError("PyTorch is required for production-cuda inference") from error

        if not torch.cuda.is_available():
            raise RuntimeError("production-cuda mode requires CUDA, but CUDA is not available")

        self.torch = torch
        self.device = torch.device("cuda")
        self.model = self._load_model(Path(model_path))
        if hasattr(self.model, "eval"):
            self.model.eval()

    def predict(self, image_path: str) -> PredictionSummary:
        tensor = self._preprocess(image_path)
        with self.torch.no_grad():
            output = self.model(tensor)
        return self._to_prediction(output)

    def _load_model(self, model_path: Path) -> Any:
        try:
            model = self.torch.jit.load(str(model_path), map_location=self.device)
        except Exception:
            model = self.torch.load(str(model_path), map_location=self.device)
            if isinstance(model, dict):
                for key in ("model", "module", "net"):
                    candidate = model.get(key)
                    if candidate is not None:
                        model = candidate
                        break
        if hasattr(model, "to"):
            model = model.to(self.device)
        if not callable(model):
            raise RuntimeError("Loaded .pt file is not callable. Prefer a TorchScript model or saved nn.Module.")
        return model

    def _preprocess(self, image_path: str):
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        tensor = self.torch.from_numpy(array).unsqueeze(0).to(self.device)
        return tensor

    def _to_prediction(self, output: Any) -> PredictionSummary:
        tensor = self._extract_tensor(output)
        if tensor is None:
            return PredictionSummary(raw=self._jsonable(output))

        tensor = tensor.detach().float().squeeze()
        if tensor.ndim == 0:
            value = float(tensor.item())
            return PredictionSummary(label=str(round(value, 4)), confidence=None, raw=value)
        if tensor.ndim > 1:
            tensor = tensor.reshape(-1)

        scores = self.torch.softmax(tensor, dim=0)
        top_count = min(5, scores.numel())
        values, indices = self.torch.topk(scores, top_count)
        top_k = [
            {"label": f"class_{int(index.item())}", "score": float(value.item()), "index": int(index.item())}
            for value, index in zip(values, indices, strict=False)
        ]
        best = top_k[0] if top_k else {}
        return PredictionSummary(
            label=best.get("label"),
            confidence=best.get("score"),
            top_k=top_k,
            raw={"shape": list(tensor.shape)},
        )

    def _extract_tensor(self, output: Any):
        if self.torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            for key in ("logits", "scores", "output", "pred"):
                value = output.get(key)
                if self.torch.is_tensor(value):
                    return value
        if isinstance(output, (list, tuple)) and output:
            return self._extract_tensor(output[0])
        return None

    def _jsonable(self, value: Any) -> Any:
        if self.torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)