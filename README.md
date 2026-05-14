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








The new error is precise: Docker is running the test container, but no NVIDIA device nodes are being injected into it. I’ll verify the current Compose and script settings so the next advice matches the repo state.

Read [](file:///Users/miles.liu/Documents/projects/TestLM/docker-compose.yml)

Read [](file:///Users/miles.liu/Documents/projects/TestLM/scripts/start_ubuntu_production.sh)

Read [](file:///Users/miles.liu/Documents/projects/TestLM/README.md#110-110), lines 110 to 150

The problem is now very narrow:

**Host NVIDIA works, but Docker GPU injection does not.**

This command failing with:

```text
cannot access '/dev/nvidia*': No such file or directory
```

means `docker run --gpus all ...` starts the container, but NVIDIA device files are not mounted into it. So the app cannot work yet, because the backend requires CUDA.

Even if these were already run:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Docker is still not actually applying the NVIDIA runtime hook.

Check these next.

**1. Do not skip the smoke test**

Run:

```bash
SKIP_BUILD=1 SKIP_CUDA_PREPULL=1 SKIP_NODE_PREPULL=1 ./scripts/start_ubuntu_production.sh
```

Do not set `SKIP_GPU_SMOKE_TEST=1`. The smoke test is correctly catching the real blocker.

**2. Check host device files**

On Ubuntu host:

```bash
ls -l /dev/nvidia*
```

Expected: files like:

```text
/dev/nvidia0
/dev/nvidiactl
/dev/nvidia-uvm
```

If they are missing, try:

```bash
sudo modprobe nvidia
sudo modprobe nvidia_uvm
ls -l /dev/nvidia*
```

**3. Check whether Docker is Snap-installed**

This is a common cause. `nvidia-ctk` may configure `/etc/docker/daemon.json`, but Snap Docker may ignore it.

Run:

```bash
which docker
snap list docker 2>/dev/null || true
systemctl status docker --no-pager
docker info | grep -Ei 'Runtimes|Default Runtime|Rootless|Cgroup'
cat /etc/docker/daemon.json
```

If `which docker` shows something under `/snap/bin/docker` or `snap list docker` shows Docker, I would remove Snap Docker and install official Docker Engine from apt. Snap Docker often causes NVIDIA runtime pain.

**4. Test NVIDIA runtime directly**

Run:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --entrypoint /bin/sh nvidia/cuda:12.1.1-base-ubuntu22.04 -c \
'echo "devices:"; ls -l /dev/nvidia*; echo "env:"; env | grep NVIDIA'
```

Also:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

The second command should show GPU info inside the container. Until it does, the app cannot start in `production-cuda`.

**5. Check NVIDIA toolkit itself**

```bash
nvidia-container-cli -k -d /dev/tty info
```

If this fails, the toolkit install/config is broken even though Docker lists an NVIDIA runtime.

Short answer: all app images are built, but production startup is blocked by Docker/NVIDIA runtime configuration. The app is behaving correctly by refusing to run without CUDA. The next thing to fix is making this command work:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```