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

Step 1/17 : ARG CUDA_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
Step 2/17 : FROM ${CUDA_BASE_IMAGE}
 ---> 02f0c5f1a54b
Step 3/17 : ARG UBUNTU_APT_MIRROR=http://archive.ubuntu.com/ubuntu
 ---> Running in 50c0c946c3cf
 ---> Removed intermediate container 50c0c946c3cf
 ---> f814acd0727d
Step 4/17 : ARG UBUNTU_SECURITY_APT_MIRROR=http://security.ubuntu.com/ubuntu
 ---> Running in 16b91837de7a
 ---> Removed intermediate container 16b91837de7a
 ---> 56cc04252335
Step 5/17 : ARG NVIDIA_APT_MIRROR=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
 ---> Running in b434d8e82757
 ---> Removed intermediate container b434d8e82757
 ---> f7c429283da3
Step 6/17 : ARG APT_RETRIES=3
 ---> Running in 17b581190b9e
 ---> Removed intermediate container 17b581190b9e
 ---> bf710de2b4bf
Step 7/17 : ARG APT_TIMEOUT=20
 ---> Running in f8af4b299b9d
 ---> Removed intermediate container f8af4b299b9d
 ---> f342c973107d
Step 8/17 : ARG APT_FORCE_IPV4=true
 ---> Running in cd16c1058fcc
 ---> Removed intermediate container cd16c1058fcc
 ---> 18143e1cec4d
Step 9/17 : ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1
 ---> Running in 524427ac0349
 ---> Removed intermediate container 524427ac0349
 ---> eedef93c7e7c
Step 10/17 : WORKDIR /app
 ---> Running in e9ce855f6f5c
 ---> Removed intermediate container e9ce855f6f5c
 ---> ceea76e2ff2d
Step 11/17 : RUN set -eux;         apt_ipv4_option="";         if [ "$APT_FORCE_IPV4" = "true" ] || [ "$APT_FORCE_IPV4" = "1" ]; then             apt_ipv4_option="-o Acquire::ForceIPv4=true";         fi;         for apt_source in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do             if [ -f "$apt_source" ]; then                 sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g; s|https://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g; s|http://security.ubuntu.com/ubuntu|${UBUNTU_SECURITY_APT_MIRROR}|g; s|https://security.ubuntu.com/ubuntu|${UBUNTU_SECURITY_APT_MIRROR}|g; s|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|${NVIDIA_APT_MIRROR}|g; s|http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|${NVIDIA_APT_MIRROR}|g" "$apt_source";             fi;         done;         apt-get update -o Acquire::Retries="${APT_RETRIES}" -o Acquire::http::Timeout="${APT_TIMEOUT}" -o Acquire::https::Timeout="${APT_TIMEOUT}" ${apt_ipv4_option}     && apt-get install -y --no-install-recommends python3 python3-pip python3-venv libglib2.0-0 libgl1 curl     && rm -rf /var/lib/apt/lists/*
 ---> Running in 9ec6258abacb
+ apt_ipv4_option=
+ [ true = true ]
+ apt_ipv4_option=-o Acquire::ForceIPv4=true
+ [ -f /etc/apt/sources.list ]
+ sed -i s|http://archive.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g; s|https://archive.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://security.ubuntu.com/ubuntu|g; s|https://security.ubuntu.com/ubuntu|http://security.ubuntu.com/ubuntu|g; s|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|g; s|http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|g /etc/apt/sources.list
+ [ -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]
+ sed -i s|http://archive.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g; s|https://archive.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://security.ubuntu.com/ubuntu|g; s|https://security.ubuntu.com/ubuntu|http://security.ubuntu.com/ubuntu|g; s|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|g; s|http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64|g /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
+ [ -f /etc/apt/sources.list.d/*.sources ]
+ apt-get update -o Acquire::Retries=3 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 -o Acquire::ForceIPv4=true
Ign:1 http://archive.ubuntu.com/ubuntu jammy InRelease
Ign:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
Ign:3 http://security.ubuntu.com/ubuntu jammy-security InRelease
Ign:4 http://archive.ubuntu.com/ubuntu jammy-updates InRelease
Ign:3 http://security.ubuntu.com/ubuntu jammy-security InRelease
Ign:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64  InRelease
