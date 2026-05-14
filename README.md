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


 docker run --rm --gpus all --entrypoint /bin/sh nvidia/cuda:12.1.1-base-ubuntu22.04 -c '
echo devices;idia*;
ls -l /dev/nvidia*;
echo libs;p | grep -E "libcuda|libnvidia-ml" || true
ldconfig -p | grep -E "libcuda|libnvidia-ml" || true
'
devices
crw-rw-rw- 1 root root 235,   0 May 14 01:34 /dev/nvidia-uvm
crw-rw-rw- 1 root root 235,   1 May 14 01:34 /dev/nvidia-uvm-tools
crw-rw-rw- 1 root root 195,   0 May 14 01:34 /dev/nvidia0
crw-rw-rw- 1 root root 195, 255 May 14 01:34 /dev/nvidiactl
libs
        libnvidia-ml.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
        libcudart.so.12 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
        libcudadebugger.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1
        libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1
ubuntu22@ubuntu22-O-E-M:~/Documents/TestML/TestML$ 

sudo apt-get install -y nvidia-container-toolkitvidia-ctk runtime configure --runtime=docker --set-as-default
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart dockermes|Default Runtime"
docker info | grep -Ei "Runtimes|Default Runtime"
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
nvidia-container-toolkit is already the newest version (1.19.0-1).
The following packages were automatically installed and are no longer required:
  javascript-common libc-ares2 libjs-highlight.js libnode72
Use 'sudo apt autoremove' to remove them.
0 upgraded, 0 newly installed, 0 to remove and 122 not upgraded.
INFO[0000] Loading config from /etc/docker/daemon.json  
INFO[0000] Wrote updated config to /etc/docker/daemon.json 
INFO[0000] It is recommended that docker daemon be restarted. 
 Runtimes: runc io.containerd.runc.v2 nvidia
 Default Runtime: nvidia

If PyTorch inside the container cannot initialize CUDA, solve it at the **Docker NVIDIA runtime layer** first. The app cannot fix this from Python because CUDA is failing before model inference.




docker compose run --rm --no-deps --entrypoint /bin/sh api -lc '
echo env;
env | grep -E "CUDA|NVIDIA";
echo devices;
ls -l /dev/nvidia* || true;
echo libs;
ldconfig -p | grep -E "libcuda|libnvidia-ml" || true
'
Container testml-api-run-832e5b9cffee Creating 
Container testml-api-run-832e5b9cffee Created 
env
NV_CUDA_COMPAT_PACKAGE=cuda-compat-12-1
CUDA_VERSION=12.1.1
NVIDIA_REQUIRE_CUDA=cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NV_CUDA_LIB_VERSION=12.1.1-1
NV_CUDA_CUDART_VERSION=12.1.105-1
NVIDIA_PRODUCT_NAME=CUDA
NVIDIA_VISIBLE_DEVICES=all
devices
crw-rw-rw- 1 root root 235,   0 May 14 01:34 /dev/nvidia-uvm
crw-rw-rw- 1 root root 235,   1 May 14 01:34 /dev/nvidia-uvm-tools
crw-rw-rw- 1 root root 195,   0 May 14 01:34 /dev/nvidia0
crw-rw-rw- 1 root root 195, 255 May 14 01:34 /dev/nvidiactl
libs
        libnvidia-ml.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
        libcudart.so.12 (libc6,x86-64) => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
        libcudadebugger.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcudadebugger.so.1
        libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1
        libcuda.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so



docker compose run --rm --no-deps --entrypoint python3 api - <<'PY'
import os
import torch

print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
PYint("cuda tensor ok:", float(x.sum()))cuda")le())
Container testml-api-run-f21d31bdf217 Creating 
Container testml-api-run-f21d31bdf217 Created 
NVIDIA_VISIBLE_DEVICES: all
CUDA_VISIBLE_DEVICES: None
torch: 2.3.1+cu121
torch cuda: 12.1
device count: 1
cuda available: False
/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:118: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
  return torch._C._cuda_getDeviceCount() > 0
Traceback (most recent call last):
  File "<stdin>", line 11, in <module>
  File "/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py", line 293, in _lazy_init
    torch._C._cuda_init()
RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero.










Do this in order on the Ubuntu PC.

**1. Verify Docker GPU injection outside the app**
Run a clean CUDA container, not your app image:

```bash
docker run --rm --gpus all --entrypoint /bin/sh nvidia/cuda:12.1.1-base-ubuntu22.04 -c '
echo devices;
ls -l /dev/nvidia*;
echo libs;
ldconfig -p | grep -E "libcuda|libnvidia-ml" || true
'
```

Expected: `/dev/nvidia0`, `/dev/nvidiactl`, and `libcuda.so` / `libnvidia-ml.so`.

If this fails, fix NVIDIA Container Toolkit.

**2. Reconfigure NVIDIA Container Toolkit**
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
docker info | grep -Ei "Runtimes|Default Runtime"
```

You want to see `nvidia` listed, ideally:

```text
Default Runtime: nvidia
```

Then recreate containers:

```bash
docker compose down --remove-orphans
docker compose up -d --force-recreate
```

**3. Check for bad CUDA env vars**
On the host:

```bash
env | grep -E "CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES"
```

If `CUDA_VISIBLE_DEVICES` is set, unset it before starting Compose:

```bash
unset CUDA_VISIBLE_DEVICES
docker compose down --remove-orphans
docker compose up -d --force-recreate
```

Your Compose already sets `NVIDIA_VISIBLE_DEVICES=all`, which is fine.

**4. If Docker GPU test works but Compose still fails**
Inspect what the app container actually receives:

```bash
docker compose run --rm --no-deps --entrypoint /bin/sh api -lc '
echo env;
env | grep -E "CUDA|NVIDIA";
echo devices;
ls -l /dev/nvidia* || true;
echo libs;
ldconfig -p | grep -E "libcuda|libnvidia-ml" || true
'
```

If this shows devices/libs correctly, then test PyTorch inside the same container:

```bash
docker compose run --rm --no-deps --entrypoint python3 api - <<'PY'
import os
import torch

print("NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
print("cuda available:", torch.cuda.is_available())

x = torch.randn(1, 3, 224, 224, device="cuda")
torch.cuda.synchronize()
print("cuda tensor ok:", float(x.sum()))
PY
```

**5. If PyTorch still says CUDA unknown error**
Try a full NVIDIA runtime config reset:

```bash
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups=false --in-place
sudo nvidia-ctk config --set nvidia-container-runtime.mode=legacy --in-place
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
docker compose down --remove-orphans
docker compose up -d --force-recreate
```

Then rerun the PyTorch test.

**6. Check rootless Docker**
If you are using rootless Docker, GPU passthrough can be more fragile. Check:

```bash
docker info | grep -i rootless
```

If it says rootless, use normal system Docker for this app.

Your host GPU and driver look fine. CUDA `13.0` on the host with CUDA `12.1` in the container is okay. The likely fix is reconfiguring NVIDIA Container Toolkit and recreating the containers so PyTorch receives a clean `/dev/nvidia*` and `libcuda` injection.