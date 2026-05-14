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

 which docker
snap list docker 2>/dev/null || true
systemctl status docker --no-pager
docker info | grep -Ei 'Runtimes|Default Runtime|Rootless|Cgroup'
cat /etc/docker/daemon.json
/usr/bin/docker
● docker.service - Docker Application Container Engine
     Loaded: loaded (/lib/systemd/system/docker.service; enabled; vendor preset: enabled)
     Active: active (running) since Thu 2026-05-14 13:19:32 CST; 6min ago
TriggeredBy: ● docker.socket
       Docs: https://docs.docker.com
   Main PID: 3355063 (dockerd)
      Tasks: 45
     Memory: 48.5M
        CPU: 1.015s
     CGroup: /system.slice/docker.service
             ├─3355063 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
             ├─3355550 /usr/bin/docker-proxy -proto tcp -host-ip 0.0.0.0 -host-port 5000 -co…
             └─3355556 /usr/bin/docker-proxy -proto tcp -host-ip :: -host-port 5000 -contain…

5月 14 13:19:31 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:19:31.995259222+08…ne."
5月 14 13:19:32 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:19:32.017717100+08….1.4
5月 14 13:19:32 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:19:32.017781287+08…ion"
5月 14 13:19:32 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:19:32.385212421+08…ock"
5月 14 13:19:32 ubuntu22-O-E-M systemd[1]: Started Docker Application Container Engine.
5月 14 13:19:45 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:19:45.391946590+08…ete"
5月 14 13:19:45 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:19:45.622376133+08…db01
5月 14 13:19:45 ubuntu22-O-E-M dockerd[3355063]: 2026/05/14 13:19:45 http: superfluous …:98)
5月 14 13:22:33 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:22:33.138524711+08…ete"
5月 14 13:25:52 ubuntu22-O-E-M dockerd[3355063]: time="2026-05-14T13:25:52.377587200+08…ete"
Hint: Some lines were ellipsized, use -l to show in full.
 Cgroup Driver: systemd
 Cgroup Version: 2
 Runtimes: io.containerd.runc.v2 nvidia runc
 Default Runtime: runc
  cgroupns
{
    "dns": [
        "8.8.8.8",
        "1.1.1.1",
        "223.5.5.5"
    ],
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }

    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --entrypoint /bin/sh nvidia/cuda:12.1.1-base-ubuntu22.04 -c \
'echo "devices:"; ls -l /dev/nvidia*; echo "env:"; env | grep NVIDIA'
devices:
ls: cannot access '/dev/nvidia*': No such file or directory
env:
NVIDIA_REQUIRE_CUDA=cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526
NVIDIA_DRIVER_CAPABILITIES=compute,utility
NVIDIA_VISIBLE_DEVICES=all


 docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: exec: "nvidia-smi": executable file not found in $PATH: unknown.


 nvidia-container-cli -k -d /dev/tty info

-- WARNING, the following logs are for debugging purposes only --

I0514 05:28:58.047819 3500294 nvc.c:396] initializing library context (version=1.19.0, build=7585946c6471402577e14474d7c56ca5be0348d7)
I0514 05:28:58.047878 3500294 nvc.c:367] using root /
I0514 05:28:58.047886 3500294 nvc.c:368] using ldcache /etc/ld.so.cache
I0514 05:28:58.047895 3500294 nvc.c:369] using unprivileged user 1000:1000
I0514 05:28:58.047923 3500294 nvc.c:413] attempting to load dxcore to see if we are running under Windows Subsystem for Linux (WSL)
I0514 05:28:58.048063 3500294 nvc.c:415] dxcore initialization failed, continuing assuming a non-WSL environment
W0514 05:28:58.049657 3500295 nvc.c:273] failed to set inheritable capabilities
W0514 05:28:58.049696 3500295 nvc.c:274] skipping kernel modules load due to failure
I0514 05:28:58.049957 3500296 rpc.c:71] starting driver rpc service
I0514 05:28:58.056572 3500298 rpc.c:71] starting nvcgo rpc service
I0514 05:28:58.057281 3500294 nvc_info.c:809] requesting driver information with ''
I0514 05:28:58.057518 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvoptix.so.580.126.09
I0514 05:28:58.057568 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-tls.so.580.126.09
I0514 05:28:58.057602 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.580.126.09
I0514 05:28:58.057634 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.126.09
I0514 05:28:58.057677 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-pkcs11.so.580.126.09
I0514 05:28:58.057703 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-pkcs11-openssl3.so.580.126.09
I0514 05:28:58.057734 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so.580.126.09
I0514 05:28:58.057778 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.580.126.09
I0514 05:28:58.057809 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.580.126.09
I0514 05:28:58.057849 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-ngx.so.580.126.09
I0514 05:28:58.057878 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.126.09
I0514 05:28:58.057920 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.580.126.09
I0514 05:28:58.057953 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv.so.580.126.09
I0514 05:28:58.057986 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-glsi.so.580.126.09
I0514 05:28:58.058018 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.580.126.09
I0514 05:28:58.058049 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-fbc.so.580.126.09
I0514 05:28:58.058088 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.580.126.09
I0514 05:28:58.058127 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.580.126.09
I0514 05:28:58.058164 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-cfg.so.580.126.09
I0514 05:28:58.058207 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvidia-allocator.so.580.126.09
I0514 05:28:58.058248 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libnvcuvid.so.580.126.09
I0514 05:28:58.058367 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libcudadebugger.so.580.126.09
I0514 05:28:58.058399 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libcuda.so.580.126.09
I0514 05:28:58.058489 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.126.09
I0514 05:28:58.058519 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libGLESv2_nvidia.so.580.126.09
I0514 05:28:58.058548 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libGLESv1_CM_nvidia.so.580.126.09
I0514 05:28:58.058576 3500294 nvc_info.c:177] selecting /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.580.126.09
I0514 05:28:58.058618 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-tls.so.580.126.09
I0514 05:28:58.058648 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-ptxjitcompiler.so.580.126.09
I0514 05:28:58.058691 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-opticalflow.so.580.126.09
I0514 05:28:58.058730 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-opencl.so.580.126.09
I0514 05:28:58.058768 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-nvvm.so.580.126.09
I0514 05:28:58.058810 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-ml.so.580.126.09
I0514 05:28:58.058850 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-gpucomp.so.580.126.09
I0514 05:28:58.058881 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-glvkspirv.so.580.126.09
I0514 05:28:58.058912 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-glsi.so.580.126.09
I0514 05:28:58.058941 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-glcore.so.580.126.09
I0514 05:28:58.058971 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-fbc.so.580.126.09
I0514 05:28:58.059010 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-encode.so.580.126.09
I0514 05:28:58.059049 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvidia-eglcore.so.580.126.09
I0514 05:28:58.059081 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libnvcuvid.so.580.126.09
I0514 05:28:58.059130 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libcuda.so.580.126.09
I0514 05:28:58.059175 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libGLX_nvidia.so.580.126.09
I0514 05:28:58.059206 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libGLESv2_nvidia.so.580.126.09
I0514 05:28:58.059235 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libGLESv1_CM_nvidia.so.580.126.09
I0514 05:28:58.059264 3500294 nvc_info.c:177] selecting /usr/lib/i386-linux-gnu/libEGL_nvidia.so.580.126.09
W0514 05:28:58.059284 3500294 nvc_info.c:413] missing library libnvidia-nscq.so
W0514 05:28:58.059289 3500294 nvc_info.c:413] missing library libnvidia-fatbinaryloader.so
W0514 05:28:58.059293 3500294 nvc_info.c:413] missing library libnvidia-compiler.so
W0514 05:28:58.059296 3500294 nvc_info.c:413] missing library libnvidia-gpucomp.so
W0514 05:28:58.059300 3500294 nvc_info.c:413] missing library libnvidia-tileiras.so
W0514 05:28:58.059303 3500294 nvc_info.c:413] missing library libvdpau_nvidia.so
W0514 05:28:58.059306 3500294 nvc_info.c:413] missing library libnvidia-ifr.so
W0514 05:28:58.059310 3500294 nvc_info.c:413] missing library libnvidia-cbl.so
W0514 05:28:58.059313 3500294 nvc_info.c:417] missing compat32 library libnvidia-cfg.so
W0514 05:28:58.059318 3500294 nvc_info.c:417] missing compat32 library libnvidia-nscq.so
W0514 05:28:58.059322 3500294 nvc_info.c:417] missing compat32 library libcudadebugger.so
W0514 05:28:58.059325 3500294 nvc_info.c:417] missing compat32 library libnvidia-fatbinaryloader.so
W0514 05:28:58.059329 3500294 nvc_info.c:417] missing compat32 library libnvidia-allocator.so
W0514 05:28:58.059334 3500294 nvc_info.c:417] missing compat32 library libnvidia-compiler.so
W0514 05:28:58.059338 3500294 nvc_info.c:417] missing compat32 library libnvidia-pkcs11.so
W0514 05:28:58.059341 3500294 nvc_info.c:417] missing compat32 library libnvidia-pkcs11-openssl3.so
W0514 05:28:58.059346 3500294 nvc_info.c:417] missing compat32 library libnvidia-gpucomp.so
W0514 05:28:58.059350 3500294 nvc_info.c:417] missing compat32 library libnvidia-tileiras.so
W0514 05:28:58.059355 3500294 nvc_info.c:417] missing compat32 library libnvidia-ngx.so
W0514 05:28:58.059358 3500294 nvc_info.c:417] missing compat32 library libvdpau_nvidia.so
W0514 05:28:58.059361 3500294 nvc_info.c:417] missing compat32 library libnvidia-ifr.so
W0514 05:28:58.059365 3500294 nvc_info.c:417] missing compat32 library libnvidia-rtcore.so
W0514 05:28:58.059369 3500294 nvc_info.c:417] missing compat32 library libnvoptix.so
W0514 05:28:58.059373 3500294 nvc_info.c:417] missing compat32 library libnvidia-cbl.so
I0514 05:28:58.059756 3500294 nvc_info.c:303] selecting /usr/bin/nvidia-smi
I0514 05:28:58.059770 3500294 nvc_info.c:303] selecting /usr/bin/nvidia-debugdump
I0514 05:28:58.059799 3500294 nvc_info.c:303] selecting /usr/bin/nvidia-persistenced
I0514 05:28:58.059820 3500294 nvc_info.c:303] selecting /usr/bin/nvidia-cuda-mps-control
I0514 05:28:58.059829 3500294 nvc_info.c:303] selecting /usr/bin/nvidia-cuda-mps-server
W0514 05:28:58.059869 3500294 nvc_info.c:439] missing binary nv-fabricmanager
I0514 05:28:58.059903 3500294 nvc_info.c:499] listing firmware path /lib/firmware/nvidia/580.126.09/gsp_ga10x.bin
I0514 05:28:58.059906 3500294 nvc_info.c:499] listing firmware path /lib/firmware/nvidia/580.126.09/gsp_tu10x.bin
I0514 05:28:58.059919 3500294 nvc_info.c:572] listing device /dev/nvidiactl
I0514 05:28:58.059923 3500294 nvc_info.c:572] listing device /dev/nvidia-uvm
I0514 05:28:58.059926 3500294 nvc_info.c:572] listing device /dev/nvidia-uvm-tools
I0514 05:28:58.059929 3500294 nvc_info.c:572] listing device /dev/nvidia-modeset
I0514 05:28:58.059942 3500294 nvc_info.c:355] listing ipc path /run/nvidia-persistenced/socket
W0514 05:28:58.059953 3500294 nvc_info.c:361] missing ipc path /var/run/nvidia-fabricmanager/socket
W0514 05:28:58.059960 3500294 nvc_info.c:361] missing ipc path /tmp/nvidia-mps
I0514 05:28:58.059963 3500294 nvc_info.c:865] requesting device information with ''
I0514 05:28:58.067629 3500294 nvc_info.c:756] listing device /dev/nvidia0 (GPU-774c6ebd-adc7-ea5d-795d-101cecb6b8f4 at 00000000:01:00.0)
NVRM version:   580.126.09
CUDA version:   13.0

Device Index:   0
Device Minor:   0
Model:          NVIDIA GeForce RTX 4080
Brand:          GeForce
GPU UUID:       GPU-774c6ebd-adc7-ea5d-795d-101cecb6b8f4
Bus Location:   00000000:01:00.0
Architecture:   8.9
I0514 05:28:58.067649 3500294 nvc.c:455] shutting down library context
I0514 05:28:58.067671 3500298 rpc.c:95] terminating nvcgo rpc service
I0514 05:28:58.067949 3500294 rpc.c:135] nvcgo rpc service terminated successfully
I0514 05:28:58.070262 3500296 rpc.c:95] terminating driver rpc service
I0514 05:28:58.070470 3500294 rpc.c:135] driver rpc service terminated successfully




docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: exec: "nvidia-smi": executable file not found in $PATH: unknown.
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