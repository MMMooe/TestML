from __future__ import annotations

from app.config import get_settings
from app.schemas import RuntimeInfo


def get_runtime_info() -> RuntimeInfo:
    settings = get_settings()
    torch_version = None
    cuda_version = None
    device_name = None
    cuda_available = False
    tensorrt_version = None
    tensorrt_available = False
    tensorrt_message = "TensorRT Python bindings are unavailable"

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    try:
        import tensorrt as trt
        from cuda import cudart

        if not hasattr(cudart, "cudaMalloc"):
            raise RuntimeError("CUDA Python runtime bindings are incomplete")

        tensorrt_version = trt.__version__
        tensorrt_available = True
        tensorrt_message = "TensorRT runtime is ready"
    except Exception as error:
        tensorrt_message = f"TensorRT runtime is unavailable: {error}"

    selected_device = "cuda" if cuda_available else "unavailable"
    if not cuda_available:
        selected_device = "unavailable"
        message = "production-cuda mode requires CUDA, but CUDA is not available"
    else:
        message = "CUDA runtime is ready"

    return RuntimeInfo(
        app_mode=settings.app_mode,
        cuda_available=cuda_available,
        selected_device=selected_device,
        torch_version=torch_version,
        cuda_version=cuda_version,
        device_name=device_name,
        tensorrt_required=settings.require_tensorrt,
        tensorrt_available=tensorrt_available,
        tensorrt_version=tensorrt_version,
        tensorrt_message=tensorrt_message,
        message=message,
    )


def assert_runtime_ready() -> None:
    settings = get_settings()
    runtime = get_runtime_info()
    if not runtime.cuda_available:
        raise RuntimeError(runtime.message)
    if settings.require_tensorrt and not runtime.tensorrt_available:
        raise RuntimeError(runtime.tensorrt_message)