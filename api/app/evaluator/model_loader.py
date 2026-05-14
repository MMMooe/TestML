from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.schemas import PredictionSummary


def create_model_runner(
    model_path: str,
    plan_path: str | None = None,
    *,
    use_tensorrt: bool = True,
    allow_tensorrt_fallback: bool = False,
) -> Any:
    if plan_path and use_tensorrt:
        candidate = Path(plan_path)
        if candidate.exists():
            try:
                return TensorRTPlanRunner(str(candidate))
            except Exception as error:
                if not allow_tensorrt_fallback:
                    raise RuntimeError(
                        "TensorRT engine was uploaded but cannot be used. Rebuild the .plan for this image's "
                        f"TensorRT/CUDA/GPU runtime or enable explicit .pt fallback. Details: {error}"
                    ) from error
                print(f"[model_loader] TensorRT plan disabled: {error}. Falling back to .pt CUDA model.")
        else:
            if not allow_tensorrt_fallback:
                raise RuntimeError(f"TensorRT plan file not found: {candidate}")
            print(f"[model_loader] TensorRT plan not found at {candidate}. Falling back to .pt CUDA model.")
    return CudaModelRunner(model_path)


class CudaModelRunner:
    backend = "torch-cuda"

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
        return self.predict_batch([image_path])[0]

    def predict_batch(self, image_paths: list[str]) -> list[PredictionSummary]:
        if not image_paths:
            return []
        tensor = self._preprocess_batch(image_paths)
        with self.torch.inference_mode():
            output = self.model(tensor)
        return self._to_predictions(output, len(image_paths))

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

    def _preprocess_batch(self, image_paths: list[str]):
        arrays = [self._preprocess_array(image_path) for image_path in image_paths]
        batch = np.stack(arrays, axis=0)
        return self.torch.from_numpy(batch).to(self.device, non_blocking=True)

    def _preprocess_array(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))

    def _to_prediction(self, output: Any) -> PredictionSummary:
        return self._to_predictions(output, 1)[0]

    def _to_predictions(self, output: Any, batch_size: int) -> list[PredictionSummary]:
        tensor = self._extract_tensor(output)
        if tensor is None:
            raw = self._jsonable(output)
            return [PredictionSummary(raw=raw) for _ in range(batch_size)]

        tensor = tensor.detach().float()
        if batch_size == 1:
            return [self._tensor_to_prediction(tensor.squeeze())]

        if tensor.ndim == 0:
            raise RuntimeError("Model returned a scalar for a batched input")
        if tensor.shape[0] != batch_size:
            raise RuntimeError(
                f"Model output batch dimension {tensor.shape[0]} does not match input batch size {batch_size}"
            )

        rows = tensor.reshape(batch_size, -1)
        return [self._tensor_to_prediction(row) for row in rows]

    def _tensor_to_prediction(self, tensor: Any) -> PredictionSummary:
        tensor = tensor.reshape(-1)
        if tensor.numel() == 0:
            return PredictionSummary(raw={"shape": list(tensor.shape), "backend": "torch-cuda"})
        if tensor.numel() == 1:
            value = float(tensor.item())
            return PredictionSummary(label=str(round(value, 4)), confidence=None, raw={"value": value, "backend": "torch-cuda"})

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
            raw={"shape": list(tensor.shape), "backend": "torch-cuda"},
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


class TensorRTPlanRunner:
    backend = "tensorrt"

    def __init__(self, plan_path: str) -> None:
        plan = Path(plan_path)
        if not plan.exists():
            raise RuntimeError(f"TensorRT plan file not found: {plan}")

        try:
            import tensorrt as trt
            from cuda import cudart
        except Exception as error:
            raise RuntimeError("TensorRT and CUDA Python bindings are unavailable") from error

        self.trt = trt
        self.cudart = cudart
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.uses_tensor_api = hasattr(self.engine, "num_io_tensors")
        if self.uses_tensor_api:
            self._init_tensor_api_layout()
        else:
            self._init_binding_api_layout()

        self.input_height, self.input_width = self._resolve_input_size()

    def predict(self, image_path: str) -> PredictionSummary:
        input_array = self._preprocess(image_path)
        if self.uses_tensor_api:
            output = self._infer_tensor_api(input_array)
        else:
            output = self._infer_binding_api(input_array)
        return _prediction_from_scores(output, backend="tensorrt")

    def _init_tensor_api_layout(self) -> None:
        input_names: list[str] = []
        output_names: list[str] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)
        if len(input_names) != 1:
            raise RuntimeError("TensorRT plan must have exactly one input tensor")
        if not output_names:
            raise RuntimeError("TensorRT plan has no output tensors")
        self.input_tensor_name = input_names[0]
        self.output_tensor_names = output_names

    def _init_binding_api_layout(self) -> None:
        self.binding_count = self.engine.num_bindings
        self.input_binding_indices = [index for index in range(self.binding_count) if self.engine.binding_is_input(index)]
        self.output_binding_indices = [index for index in range(self.binding_count) if not self.engine.binding_is_input(index)]
        if len(self.input_binding_indices) != 1:
            raise RuntimeError("TensorRT plan must have exactly one input binding")
        if not self.output_binding_indices:
            raise RuntimeError("TensorRT plan has no output bindings")

    def _resolve_input_size(self) -> tuple[int, int]:
        if self.uses_tensor_api:
            shape = tuple(self.engine.get_tensor_shape(self.input_tensor_name))
        else:
            shape = tuple(self.engine.get_binding_shape(self.input_binding_indices[0]))

        if len(shape) >= 4 and shape[-2] > 0 and shape[-1] > 0:
            return int(shape[-2]), int(shape[-1])
        return 224, 224

    def _preprocess(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB").resize((self.input_width, self.input_height))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, axis=0)

    def _infer_tensor_api(self, input_array: np.ndarray) -> np.ndarray:
        allocations = []
        stream = self._cuda_call(self.cudart.cudaStreamCreate())
        output_buffers: dict[str, tuple[np.ndarray, Any]] = {}
        try:
            input_dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(self.input_tensor_name)))
            input_host = np.ascontiguousarray(input_array.astype(input_dtype, copy=False))
            if any(dimension < 0 for dimension in self.context.get_tensor_shape(self.input_tensor_name)):
                self.context.set_input_shape(self.input_tensor_name, tuple(input_host.shape))

            input_device = self._cuda_call(self.cudart.cudaMalloc(input_host.nbytes))
            allocations.append(input_device)
            self._cuda_call(
                self.cudart.cudaMemcpyAsync(
                    int(input_device),
                    input_host.ctypes.data,
                    input_host.nbytes,
                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )
            self.context.set_tensor_address(self.input_tensor_name, int(input_device))

            for name in self.output_tensor_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"TensorRT output shape unresolved for tensor: {name}")
                dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
                host = np.empty(shape, dtype=dtype)
                device = self._cuda_call(self.cudart.cudaMalloc(host.nbytes))
                allocations.append(device)
                self.context.set_tensor_address(name, int(device))
                output_buffers[name] = (host, device)

            if not self.context.execute_async_v3(stream):
                raise RuntimeError("TensorRT execution failed")

            for name in self.output_tensor_names:
                host, device = output_buffers[name]
                self._cuda_call(
                    self.cudart.cudaMemcpyAsync(
                        host.ctypes.data,
                        int(device),
                        host.nbytes,
                        self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    )
                )

            self._cuda_call(self.cudart.cudaStreamSynchronize(stream))
            return output_buffers[self.output_tensor_names[0]][0]
        finally:
            for allocation in allocations:
                try:
                    self._cuda_call(self.cudart.cudaFree(allocation))
                except Exception:
                    pass
            try:
                self._cuda_call(self.cudart.cudaStreamDestroy(stream))
            except Exception:
                pass

    def _infer_binding_api(self, input_array: np.ndarray) -> np.ndarray:
        allocations = []
        stream = self._cuda_call(self.cudart.cudaStreamCreate())
        bindings = [0] * self.binding_count
        outputs: dict[int, tuple[np.ndarray, Any]] = {}
        try:
            input_index = self.input_binding_indices[0]
            input_shape = tuple(input_array.shape)
            if any(dimension < 0 for dimension in self.engine.get_binding_shape(input_index)):
                self.context.set_binding_shape(input_index, input_shape)

            input_dtype = np.dtype(self.trt.nptype(self.engine.get_binding_dtype(input_index)))
            input_host = np.ascontiguousarray(input_array.astype(input_dtype, copy=False))
            input_device = self._cuda_call(self.cudart.cudaMalloc(input_host.nbytes))
            allocations.append(input_device)
            self._cuda_call(
                self.cudart.cudaMemcpyAsync(
                    int(input_device),
                    input_host.ctypes.data,
                    input_host.nbytes,
                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    stream,
                )
            )
            bindings[input_index] = int(input_device)

            for index in self.output_binding_indices:
                shape = tuple(self.context.get_binding_shape(index))
                if any(dimension < 0 for dimension in shape):
                    raise RuntimeError(f"TensorRT output shape unresolved for binding {index}")
                dtype = np.dtype(self.trt.nptype(self.engine.get_binding_dtype(index)))
                host = np.empty(shape, dtype=dtype)
                device = self._cuda_call(self.cudart.cudaMalloc(host.nbytes))
                allocations.append(device)
                bindings[index] = int(device)
                outputs[index] = (host, device)

            if not self.context.execute_async_v2(bindings=bindings, stream_handle=stream):
                raise RuntimeError("TensorRT execution failed")

            for index, output in outputs.items():
                host, device = output
                self._cuda_call(
                    self.cudart.cudaMemcpyAsync(
                        host.ctypes.data,
                        int(device),
                        host.nbytes,
                        self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    )
                )

            self._cuda_call(self.cudart.cudaStreamSynchronize(stream))
            return outputs[self.output_binding_indices[0]][0]
        finally:
            for allocation in allocations:
                try:
                    self._cuda_call(self.cudart.cudaFree(allocation))
                except Exception:
                    pass
            try:
                self._cuda_call(self.cudart.cudaStreamDestroy(stream))
            except Exception:
                pass

    def _cuda_call(self, result: Any) -> Any:
        if isinstance(result, tuple):
            error = result[0]
            values = result[1:]
        else:
            error = result
            values = ()

        if error != self.cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA runtime call failed: {self._cuda_error(error)}")
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return values

    def _cuda_error(self, error: Any) -> str:
        try:
            _, name = self.cudart.cudaGetErrorName(error)
            _, message = self.cudart.cudaGetErrorString(error)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="replace")
            return f"{name}: {message}"
        except Exception:
            return str(error)


def _prediction_from_scores(output: Any, backend: str) -> PredictionSummary:
    array = np.asarray(output)
    if array.size == 0:
        return PredictionSummary(raw={"shape": list(array.shape), "backend": backend})

    vector = array.astype(np.float32).reshape(-1)
    if vector.size == 1:
        value = float(vector[0])
        return PredictionSummary(label=str(round(value, 4)), confidence=None, raw={"value": value, "backend": backend})

    shifted = vector - float(np.max(vector))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    if denom <= 0:
        return PredictionSummary(raw={"shape": list(array.shape), "backend": backend})

    scores = exp / denom
    top_indices = np.argsort(scores)[::-1][: min(5, scores.size)]
    top_k = [
        {"label": f"class_{int(index)}", "score": float(scores[index]), "index": int(index)}
        for index in top_indices
    ]
    best = top_k[0] if top_k else {}
    return PredictionSummary(
        label=best.get("label"),
        confidence=best.get("score"),
        top_k=top_k,
        raw={"shape": list(array.shape), "backend": backend},
    )