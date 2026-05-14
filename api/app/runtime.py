from __future__ import annotations

from app.config import get_settings
from app.schemas import RuntimeInfo


def get_runtime_info() -> RuntimeInfo:
    settings = get_settings()
    torch_version = None
    cuda_version = None
    device_name = None
    cuda_available = False

    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

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
        message=message,
    )


def assert_runtime_ready() -> None:
    settings = get_settings()
    runtime = get_runtime_info()
    if not runtime.cuda_available:
        raise RuntimeError(runtime.message)