# Model Evaluation Gallery

Dockerized app for running PyTorch `.pt` inference and optional evaluation on an Ubuntu PC with an NVIDIA GPU.

Users upload a model, test images, and optionally a JSON annotation file. Images-only runs inference. Images plus JSON runs inference and evaluation. Results are displayed in a gallery with per-image details and exports.

TensorRT `.plan` engines are supported as a first-class production path. When a `.plan` file is uploaded with a `.pt` model, the job uses TensorRT by default and fails clearly if the engine cannot run on the current TensorRT/CUDA/GPU runtime. The UI has an explicit fallback checkbox if you want to allow the `.pt` CUDA model to run instead.

## Quick Start After Clone (Ubuntu Production)

From the repository root:

```bash
chmod +x scripts/*.sh
./scripts/setup_and_start_ubuntu.sh
```

What this does:

- Installs web dependencies in `web/`.
- Creates `api/.venv` and installs API dependencies.
- Pre-pulls CUDA and Node base images with retry.
- Builds Docker images.
- Verifies GPU availability and starts the production stack.

Open:

- UI: http://localhost:3000
- API docs: http://localhost:8000/docs

To stop:

```bash
docker compose down
```

If the machine uses the legacy Compose binary, use `docker-compose down`.

## Install and Start Separately (Ubuntu)

```bash
chmod +x scripts/*.sh
./scripts/install_ubuntu.sh
./scripts/start_ubuntu_production.sh
```

The start script builds images first, then starts containers with `up --no-build`. This avoids a second build-time registry metadata lookup during startup.

## Slow Docker Hub / Timeout Recovery

If your Ubuntu server has slow or unstable access to Docker Hub, use longer retries:

```bash
PULL_RETRY_COUNT=20 PULL_RETRY_DELAY=30 ./scripts/start_ubuntu_production.sh
```

The scripts default to `DOCKER_BUILDKIT=0` and `COMPOSE_DOCKER_CLI_BUILD=0` so Docker can use already-pulled local base images instead of repeatedly resolving remote image metadata during build.

If the images were already built successfully and you only need to start containers:

```bash
SKIP_BUILD=1 ./scripts/start_ubuntu_production.sh
```

If your registry/network requires alternate image names, override the base images:

```bash
CUDA_BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04 NODE_BASE_IMAGE=node:20-alpine ./scripts/start_ubuntu_production.sh
```

The API image installs TensorRT runtime packages from the NVIDIA APT repository. If package names differ on your selected mirror, override them:

```bash
TENSORRT_APT_PACKAGES="python3-libnvinfer libnvinfer8 libnvinfer-plugin8 libnvinfer-bin" ./scripts/start_ubuntu_production.sh
```

If it still fails, verify host DNS/network:

```bash
getent hosts registry-1.docker.io
curl -I https://registry-1.docker.io/v2/
docker pull node:20-alpine
docker pull nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

## Slow Ubuntu / NVIDIA APT Recovery

After the Docker base images are already pulled, the API image still needs Ubuntu packages during `apt-get update`. The scripts default to official Ubuntu and NVIDIA repositories, which is usually best when the Ubuntu PC uses a VPN exit node outside mainland China:

```bash
UBUNTU_APT_MIRROR=http://archive.ubuntu.com/ubuntu
UBUNTU_SECURITY_APT_MIRROR=http://security.ubuntu.com/ubuntu
NVIDIA_APT_MIRROR=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
```

Docker builds use the host network by default so `apt-get` follows the Ubuntu host's VPN and DNS route more closely:

```bash
DOCKER_BUILD_NETWORK=host ./scripts/setup_and_start_ubuntu.sh
```

If the VPN is off and the machine needs mainland mirrors, opt in explicitly:

```bash
UBUNTU_APT_MIRROR=https://mirrors.aliyun.com/ubuntu UBUNTU_SECURITY_APT_MIRROR=https://mirrors.aliyun.com/ubuntu NVIDIA_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/nvidia-cuda/ubuntu2204/x86_64 ./scripts/setup_and_start_ubuntu.sh
```

If `apt-get update` is still stuck, test the selected repositories from the Ubuntu PC:

```bash
curl -I http://archive.ubuntu.com/ubuntu
curl -I http://security.ubuntu.com/ubuntu
curl -I https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/
```

## Production: Ubuntu NVIDIA GPU

Prerequisites on the Ubuntu PC:

```bash
nvidia-smi
docker --version
docker compose version
```

Install NVIDIA Container Toolkit if Docker cannot see the GPU. A useful Docker GPU smoke test is:

```bash
docker run --rm --gpus all --entrypoint /bin/sh nvidia/cuda:12.1.1-base-ubuntu22.04 -c 'test -e /dev/nvidiactl || test -e /dev/nvidia0'
```

If the host `nvidia-smi` works but containers print `WARNING: The NVIDIA Driver was not detected`, reconfigure Docker's NVIDIA runtime:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info | grep -i nvidia
```

Then start the app:

```bash
./scripts/start_ubuntu_production.sh
```

The backend runs in `production-cuda` mode and fails fast if CUDA is unavailable.

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

TensorRT engines are not portable model files. Build each `.plan` for the same TensorRT major/minor version, CUDA runtime family, GPU architecture, input shape, and precision profile used by the production API image.

Expected behavior:

- No `.plan`: the app runs the uploaded `.pt` model with PyTorch CUDA.
- `.plan` uploaded: the app runs TensorRT and records `tensorrt` as the job backend.
- `.plan` uploaded and incompatible: the job fails before processing images unless explicit `.pt` fallback is enabled in the UI.

Check runtime status at:

```bash
curl http://localhost:8000/runtime
```

## Storage

Runtime data is mounted under `storage/`:

- `storage/uploads/models/`
- `storage/uploads/datasets/`
- `storage/jobs/`
- `storage/results/`

These folders are ignored by git except for `.gitkeep` placeholders.

## Notes On `.pt` Files

Plain PyTorch `.pt` files can use pickle under the hood. Treat uploaded models as trusted local engineering artifacts. TorchScript `.pt` files are preferred for predictable deployment.



docker compose logs --tail=200 api
api-1  | 
api-1  | ==========
api-1  | == CUDA ==
api-1  | ==========
api-1  | 
api-1  | CUDA Version 12.1.1
api-1  | 
api-1  | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
api-1  | 
api-1  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
api-1  | By pulling and using the container, you accept the terms and conditions of this license:
api-1  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
api-1  | 
api-1  | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
api-1  | 
api-1  | INFO:     Started server process [1]
api-1  | INFO:     Waiting for application startup.
api-1  | /usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:118: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
api-1  |   return torch._C._cuda_getDeviceCount() > 0
api-1  | ERROR:    Traceback (most recent call last):
api-1  |   File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 732, in lifespan
api-1  |     async with self.lifespan_context(app) as maybe_state:
api-1  |   File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 608, in __aenter__
api-1  |     await self._router.startup()
api-1  |   File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 711, in startup
api-1  |     handler()
api-1  |   File "/app/app/main.py", line 29, in startup
api-1  |     assert_runtime_ready()
api-1  |   File "/app/app/runtime.py", line 67, in assert_runtime_ready
api-1  |     raise RuntimeError(runtime.message)
api-1  | RuntimeError: production-cuda mode requires CUDA, but CUDA is not available
api-1  | 
api-1  | ERROR:    Application startup failed. Exiting.
api-1  | 
api-1  | ==========
api-1  | == CUDA ==
api-1  | ==========
api-1  | 
api-1  | CUDA Version 12.1.1
api-1  | 
api-1  | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
api-1  | 
api-1  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
api-1  | By pulling and using the container, you accept the terms and conditions of this license:
api-1  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
api-1  | 
api-1  | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
api-1  | 
api-1  | INFO:     Started server process [1]
api-1  | INFO:     Waiting for application startup.
api-1  | /usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:118: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
api-1  |   return torch._C._cuda_getDeviceCount() > 0
api-1  | ERROR:    Traceback (most recent call last):
api-1  |   File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 732, in lifespan
api-1  |     async with self.lifespan_context(app) as maybe_state:
api-1  |   File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 608, in __aenter__
api-1  |     await self._router.startup()
api-1  |   File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 711, in startup
api-1  |     handler()
api-1  |   File "/app/app/main.py", line 29, in startup
api-1  |     assert_runtime_ready()
api-1  |   File "/app/app/runtime.py", line 67, in assert_runtime_ready
api-1  |     raise RuntimeError(runtime.message)
api-1  | RuntimeError: production-cuda mode requires CUDA, but CUDA is not available
api-1  | 
api-1  | ERROR:    Application startup failed. Exiting.