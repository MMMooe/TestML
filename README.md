# Model Evaluation Gallery

Dockerized app for running PyTorch `.pt` inference and optional evaluation on an Ubuntu PC with an NVIDIA GPU.

Users upload a model, test images, and optionally a JSON annotation file. Images-only runs inference. Images plus JSON runs inference and evaluation. Results are displayed in a gallery with per-image details and exports.

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

If it still fails, verify host DNS/network:

```bash
getent hosts registry-1.docker.io
curl -I https://registry-1.docker.io/v2/
docker pull node:20-alpine
docker pull nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

## Slow Ubuntu / NVIDIA APT Recovery

After the Docker base images are already pulled, the API image still needs Ubuntu packages during `apt-get update`. For mainland China networks, the CUDA Dockerfile defaults to China-accessible APT mirrors:

```bash
UBUNTU_APT_MIRROR=https://mirrors.aliyun.com/ubuntu
NVIDIA_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/nvidia-cuda/ubuntu2204/x86_64
```

You can override them when starting the app:

```bash
UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu NVIDIA_APT_MIRROR=https://mirrors.ustc.edu.cn/nvidia-cuda/ubuntu2204/x86_64 ./scripts/setup_and_start_ubuntu.sh
```

If `apt-get update` is still stuck, test the mirrors from the Ubuntu PC:

```bash
curl -I https://mirrors.aliyun.com/ubuntu
curl -I https://mirrors.tuna.tsinghua.edu.cn/nvidia-cuda/ubuntu2204/x86_64/
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

Then start the app:

```bash
./scripts/start_ubuntu_production.sh
```

The backend runs in `production-cuda` mode and fails fast if CUDA is unavailable.

## Dataset Modes

- Model file: `.pt`, required.
- Images: required, either as individual image files or a zip archive.
- JSON annotations: optional.

Run behavior:

- Images only: inference-only job, predictions gallery, no metrics.
- Images plus JSON: evaluation-and-inference job, predictions gallery, metrics, and ground-truth details.

## Storage

Runtime data is mounted under `storage/`:

- `storage/uploads/models/`
- `storage/uploads/datasets/`
- `storage/jobs/`
- `storage/results/`

These folders are ignored by git except for `.gitkeep` placeholders.

## Notes On `.pt` Files

Plain PyTorch `.pt` files can use pickle under the hood. Treat uploaded models as trusted local engineering artifacts. TorchScript `.pt` files are preferred for predictable deployment.


chmod +x scripts/*.sh
PULL_RETRY_COUNT=20 PULL_RETRY_DELAY=30 ./scripts/setup_and_start_ubuntu.sh


UBUNTU_APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/ubuntu \
NVIDIA_APT_MIRROR=https://mirrors.ustc.edu.cn/nvidia-cuda/ubuntu2204/x86_64 \
PULL_RETRY_COUNT=20 \
PULL_RETRY_DELAY=30 \
./scripts/setup_and_start_ubuntu.sh