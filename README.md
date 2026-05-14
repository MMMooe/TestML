# Model Evaluation Gallery

Dockerized app for running PyTorch `.pt` inference and optional evaluation on an Ubuntu PC with an NVIDIA GPU.

Users upload a model, test images, and optionally a JSON annotation file. Images-only runs inference. Images plus JSON runs inference and evaluation. Results are displayed in a gallery with per-image details and exports.

## Quick Start After Clone (Ubuntu Production)

From the repository root:

```bash
chmod +x scripts/*.sh
./scripts/setup_and_start_ubuntu.sh
```

If your server has slow or unstable Docker Hub connectivity, you can increase pull retries:

```bash
PULL_RETRY_COUNT=8 PULL_RETRY_DELAY=20 ./scripts/setup_and_start_ubuntu.sh
```

What this does:

- Installs web dependencies in `web/`.
- Creates `api/.venv` and installs API dependencies.
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

You can override the CUDA base image if your registry/network prefers a different tag:

```bash
CUDA_BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04 ./scripts/start_ubuntu_production.sh
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
docker compose up --build
```

If the machine has the legacy Compose binary instead of the Docker Compose v2 plugin, use `docker-compose up --build`.

Open:

- UI: http://localhost:3000
- API docs: http://localhost:8000/docs

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






docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
[install] Building Docker images
[+] Building 30.3s (5/5) FINISHED                                                            
 => [internal] load local bake definitions                                              0.0s
 => => reading from stdin 1.14kB                                                        0.0s
 => [web internal] load build definition from Dockerfile                                0.0s
 => => transferring dockerfile: 612B                                                    0.0s
 => [api internal] load build definition from Dockerfile.gpu                            0.0s
 => => transferring dockerfile: 655B                                                    0.0s
 => ERROR [web internal] load metadata for docker.io/library/node:20-alpine            30.0s
 => ERROR [api internal] load metadata for docker.io/nvidia/cuda:12.1.1-cudnn8-runtim  30.0s
------
 > [web internal] load metadata for docker.io/library/node:20-alpine:
------
------
 > [api internal] load metadata for docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04:
------
[+] build 0/2
 ⠙ Image testml-api Building                                                           30.3s 
 ⠙ Image testml-web Building                                                           30.3s 
Dockerfile:14

--------------------

  12 |     RUN npm run build

  13 |     

  14 | >>> FROM node:20-alpine AS runner

  15 |     WORKDIR /app

  16 |     ENV NODE_ENV=production

--------------------

target web: failed to solve: DeadlineExceeded: node:20-alpine: failed to resolve source metadata for docker.io/library/node:20-alpine: failed to do request: Head "https://registry-1.docker.io/v2/library/node/manifests/20-alpine": dial tcp: lookup registry-1.docker.io: i/o timeout