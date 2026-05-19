# Model Evaluation Gallery

Conda-first local app for running PyTorch `.pt` inference and optional evaluation on an Ubuntu PC with an NVIDIA GPU.

Chinese Ubuntu setup guide: [README.zh-CN.md](README.zh-CN.md)

Users upload a model, test images, and optionally a JSON annotation file. Images-only runs inference. Images plus JSON runs inference and evaluation. Results are displayed in a gallery with per-image details and exports.

TensorRT `.plan` engines are supported as a first-class production path. When a `.plan` file is uploaded with a `.pt` model, the job uses TensorRT by default and fails clearly if the engine cannot run on the current TensorRT/CUDA/GPU runtime. The UI has an explicit fallback checkbox if you want to allow the `.pt` CUDA model to run instead.

## Quick Start With Conda

Run these commands from the repository root on the Ubuntu NVIDIA machine:

```bash
chmod +x scripts/*.sh
./scripts/setup_conda.sh
./scripts/start_conda.sh
```

Open:

- UI: http://localhost:3000
- API docs: http://localhost:8000/docs
- Runtime status: http://localhost:8000/runtime

Stop the app with `Ctrl-C` in the terminal running `scripts/start_conda.sh`.

## Local Runtime Requirements

The production runtime is intentionally strict. The backend starts in `production-cuda` mode and fails fast unless CUDA is available. TensorRT is required by default so `.plan` engines are validated against the same local runtime that will execute jobs.

Prerequisites:

```bash
nvidia-smi
conda --version
```

Recommended Ubuntu host packages:

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libgl1
```

The Conda environment includes Python 3.10, Node.js 20, FastAPI, Next.js dependencies, PyTorch CUDA 12.1, and CUDA Python. TensorRT packaging varies by Ubuntu/NVIDIA setup, so verify it explicitly inside the Conda environment:

```bash
conda activate testlm
python - <<'PY'
import torch
import tensorrt as trt
from cuda import cudart

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(trt.__version__, hasattr(cudart, "cudaMalloc"))
PY
```

If TensorRT is not available, install the TensorRT runtime and Python bindings that match the CUDA runtime, GPU, and `.plan` files you intend to use. One pip-based option to try is:

```bash
INSTALL_TENSORRT=1 ./scripts/setup_conda.sh
```

If that does not match your host, install TensorRT through NVIDIA's Ubuntu packages or another verified local method, then rerun:

```bash
./scripts/check_conda_runtime.sh
```

## Setup Details

The default environment name is `testlm`. To use another name:

```bash
ENV_NAME=my-testlm ./scripts/setup_conda.sh
ENV_NAME=my-testlm ./scripts/start_conda.sh
```

The setup script performs these steps:

- Creates or updates the Conda environment from `environment.yml`.
- Installs API GPU dependencies from `api/requirements-gpu.txt`.
- Installs frontend dependencies in `web/` using `npm ci` when `package-lock.json` exists.
- Creates the runtime storage directories under `storage/`.
- Runs a CUDA and TensorRT preflight check.

To skip only the preflight while preparing a machine that does not yet have TensorRT installed:

```bash
SKIP_RUNTIME_CHECK=1 ./scripts/setup_conda.sh
```

## Starting Modes

Development mode starts the Next.js dev server:

```bash
./scripts/start_conda.sh
```

Production-like local mode builds the frontend first and then starts `next start`:

```bash
WEB_MODE=production ./scripts/start_conda.sh
```

Important environment variables:

- `APP_MODE=production-cuda`
- `APP_REQUIRE_TENSORRT=true`
- `APP_STORAGE_DIR=/absolute/path/to/storage`
- `APP_CORS_ORIGINS=http://localhost:3000`
- `NEXT_PUBLIC_API_URL=http://localhost:8000`

Copy `.env.example` to `.env` if you need to override backend defaults. Copy `web/.env.local.example` to `web/.env.local` if you need to override the frontend API URL outside the launcher scripts.

## Runtime Check

Before startup, check the local Conda runtime:

```bash
./scripts/check_conda_runtime.sh
```

After startup, also check the running API:

```bash
CHECK_SERVER=1 ./scripts/check_conda_runtime.sh
```

The `/runtime` response should report `cuda_available: true`, `selected_device: cuda`, `tensorrt_required: true`, and `tensorrt_available: true`.

## Dataset Modes

- Model file: `.pt`, required.
- TensorRT engine: `.plan`, optional but production-ready.
- Images: required, either as individual image files or a zip archive.
- JSON annotations: optional.

Run behavior:

- Images only: inference-only job, predictions gallery, no metrics.
- Images plus JSON: evaluation-and-inference job, predictions gallery, metrics, and ground-truth details.

## Batch Size And Throughput

The run setup batch size is used for PyTorch CUDA inference. Larger batches reduce per-image GPU launch overhead and reduce status-file writes because the backend updates job progress once per batch. Increase batch size until GPU memory becomes the limiting factor.

TensorRT engines may be fixed to the batch shape used when the `.plan` was built. If a TensorRT engine is uploaded, build it with the batch/input profile you plan to use in production.

## TensorRT Engines

TensorRT engines are not portable model files. Build each `.plan` for the same TensorRT major/minor version, CUDA runtime family, GPU architecture, input shape, and precision profile used by the local Conda runtime.

Expected behavior:

- No `.plan`: the app runs the uploaded `.pt` model with PyTorch CUDA.
- `.plan` uploaded: the app runs TensorRT and records `tensorrt` as the job backend.
- `.plan` uploaded and incompatible: the job fails before processing images unless explicit `.pt` fallback is enabled in the UI.

Check runtime status at:

```bash
curl http://localhost:8000/runtime
```

## Storage

Runtime data is stored under `storage/` by default:

- `storage/uploads/models/`
- `storage/uploads/datasets/`
- `storage/jobs/`
- `storage/results/`

These folders are ignored by git except for `.gitkeep` placeholders. Uploaded models, datasets, jobs, and generated results should stay out of version control.

## Notes On `.pt` Files

Plain PyTorch `.pt` files can use pickle under the hood. Treat uploaded models as trusted local engineering artifacts. TorchScript `.pt` files are preferred for predictable deployment.

## Legacy Docker Files

Docker files remain in the repository only as a transition reference for the previous packaging scheme. The supported local workflow is the Conda path above.
