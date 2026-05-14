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






ubuntu22@ubuntu22-O-E-M:~/Documents/TestML/TestML$ /home/ubuntu22/Documents/TestML/TestML/scripts/start_ubuntu_production.sh
[start] Checking NVIDIA GPU visibility
[start] Running Docker GPU smoke test
Unable to find image 'nvidia/cuda:12.1.1-base-ubuntu22.04' locally
12.1.1-base-ubuntu22.04: Pulling from nvidia/cuda
aece8493d397: Pull complete 
dd4939a04761: Pull complete 
b0d7cc89b769: Pull complete 
1532d9024b9c: Pull complete 
04fc8a31fa53: Pull complete 
Digest: sha256:457a4076c56025f51217bff647ca631c7880ad3dbf546b03728ba98297ebbc22
Status: Downloaded newer image for nvidia/cuda:12.1.1-base-ubuntu22.04
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: exec: "nvidia-smi": executable file not found in $PATH: unknown.