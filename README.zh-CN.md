# Model Evaluation Gallery Ubuntu 安装指南

本文说明如何在用户自己的 Ubuntu NVIDIA GPU 主机上，用 Conda 本地运行 Model Evaluation Gallery。当前推荐方式是 Conda-first，本地直接启动后端 FastAPI 和前端 Next.js，不再依赖 Docker 作为默认运行方案。

应用用于上传 PyTorch `.pt` 模型、测试图片，以及可选的 JSON 标注文件。只上传图片时执行推理；上传图片和 JSON 标注时执行推理加评估。结果会在网页画廊中展示，并提供逐图详情和导出结果。

## 1. 目标机器要求

推荐运行环境：

- Ubuntu 22.04 或兼容版本
- NVIDIA GPU
- 已安装可用的 NVIDIA 驱动
- 可执行 `nvidia-smi`
- 已安装 Conda、Miniconda、Anaconda 或 Mambaforge
- 可访问 Python 和 npm 包源

先在 Ubuntu 机器上确认 GPU 和 Conda：

```bash
nvidia-smi
conda --version
```

如果 `nvidia-smi` 不能正常显示 GPU，请先安装或修复 NVIDIA 驱动。后端会在启动时检查 CUDA；如果 CUDA 不可用，应用会直接启动失败。

建议安装这些 Ubuntu 系统库：

```bash
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libgl1
```

## 2. 获取项目代码

进入你希望放置项目的目录，然后克隆项目：

```bash
git clone https://github.com/MMMooe/TestML.git
cd TestLM
```

如果代码已经在机器上，只需要进入项目根目录：

```bash
cd TestLM
```

后续命令都默认从项目根目录执行。

## 3. 创建 Conda 环境

项目根目录提供了 `environment.yml`。默认环境名是 `testlm`。

```bash
chmod +x scripts/*.sh
./scripts/setup_conda.sh
```

这个脚本会执行以下工作：

- 创建或更新 `testlm` Conda 环境
- 安装 Python 3.10
- 安装 Node.js 20
- 安装 FastAPI 后端依赖
- 安装 PyTorch CUDA 12.1 和 CUDA Python
- 在 `web/` 中安装 Next.js 前端依赖
- 创建 `storage/` 下的运行时目录
- 检查 CUDA 和 TensorRT 是否可用

如果你想使用其他 Conda 环境名：

```bash
ENV_NAME=my-testlm ./scripts/setup_conda.sh
ENV_NAME=my-testlm ./scripts/start_conda.sh
```

## 4. TensorRT 安装与检查

当前生产运行模式默认要求 TensorRT 可用。TensorRT `.plan` 引擎不是通用文件，必须匹配当前机器的 TensorRT 版本、CUDA 版本、GPU 架构、输入形状和精度配置。

先尝试自动安装 TensorRT Python 包：

```bash
INSTALL_TENSORRT=1 ./scripts/setup_conda.sh
```

如果该方式不适合你的主机，请按照 NVIDIA 官方文档安装与本机 CUDA 和驱动匹配的 TensorRT 运行时和 Python bindings。安装完成后执行：

```bash
./scripts/check_conda_runtime.sh
```

也可以手动进入 Conda 环境检查：

```bash
conda activate testlm
python - <<'PY'
import torch
import tensorrt as trt
from cuda import cudart

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("tensorrt:", trt.__version__)
print("cudaMalloc available:", hasattr(cudart, "cudaMalloc"))
PY
```

预期结果：

- `cuda available` 为 `True`
- 能正常打印 TensorRT 版本
- `cudaMalloc available` 为 `True`

如果你只是先准备环境，TensorRT 还没装好，可以临时跳过安装时的运行时检查：

```bash
SKIP_RUNTIME_CHECK=1 ./scripts/setup_conda.sh
```

注意：默认后端仍然会在启动时要求 CUDA 和 TensorRT 可用。

## 5. 启动应用

开发模式启动：

```bash
./scripts/start_conda.sh
```

启动成功后打开：

- 前端界面：http://localhost:3000
- API 文档：http://localhost:8000/docs
- 运行时状态：http://localhost:8000/runtime

按 `Ctrl-C` 可以同时停止后端和前端。

如果要用更接近生产的前端模式，先构建再启动 Next.js：

```bash
WEB_MODE=production ./scripts/start_conda.sh
```

## 6. 环境变量配置

默认情况下，启动脚本会设置本地运行所需的环境变量。常用变量如下：

- `APP_MODE=production-cuda`
- `APP_REQUIRE_TENSORRT=true`
- `APP_STORAGE_DIR=/absolute/path/to/storage`
- `APP_CORS_ORIGINS=http://localhost:3000`
- `APP_MAX_UPLOAD_MB=4096`
- `NEXT_PUBLIC_API_URL=http://localhost:8000`

如果需要覆盖后端配置，可以复制模板：

```bash
cp .env.example .env
```

然后编辑 `.env`。

如果需要单独覆盖前端 API 地址，可以复制：

```bash
cp web/.env.local.example web/.env.local
```

注意：`NEXT_PUBLIC_API_URL` 会在前端构建时写入浏览器 bundle。如果修改了该变量，并且使用 `WEB_MODE=production`，需要重新构建前端。

## 7. 启动后的验证

应用启动后，检查后端运行时：

```bash
curl http://localhost:8000/runtime
```

关键字段应满足：

- `cuda_available` 为 `true`
- `selected_device` 为 `cuda`
- `tensorrt_required` 为 `true`
- `tensorrt_available` 为 `true`

也可以让检查脚本同时检查正在运行的 API：

```bash
CHECK_SERVER=1 ./scripts/check_conda_runtime.sh
```

## 8. 使用方式

网页中需要上传：

- 模型文件：`.pt`，必需
- TensorRT 引擎：`.plan`，可选
- 测试图片：必需，可以上传单张、多张或 zip 压缩包
- JSON 标注：可选

运行行为：

- 只上传图片：执行 inference-only job，只生成预测结果画廊
- 上传图片和 JSON 标注：执行 evaluation-and-inference job，生成预测结果、指标和 ground-truth 详情
- 上传 `.plan`：默认使用 TensorRT 后端
- `.plan` 不兼容：任务会失败，除非在 UI 中显式允许回退到 `.pt` CUDA 模型

## 9. 数据存储位置

运行时数据默认放在项目根目录的 `storage/` 下：

- `storage/uploads/models/`
- `storage/uploads/datasets/`
- `storage/jobs/`
- `storage/results/`

这些目录中的上传模型、数据集、任务状态和生成结果不应该提交到 Git。

如果要把运行时数据放到其他磁盘，设置：

```bash
APP_STORAGE_DIR=/data/testlm-storage ./scripts/start_conda.sh
```

## 10. 常见问题

### 后端启动失败，提示 CUDA 不可用

先检查：

```bash
nvidia-smi
conda activate testlm
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

如果 `torch.cuda.is_available()` 是 `False`，通常是 NVIDIA 驱动、CUDA 兼容性或 PyTorch CUDA wheel 安装问题。

### 后端启动失败，提示 TensorRT 不可用

检查：

```bash
conda activate testlm
python -c "import tensorrt as trt; from cuda import cudart; print(trt.__version__, hasattr(cudart, 'cudaMalloc'))"
```

如果无法导入 `tensorrt`，请安装匹配当前机器的 TensorRT Python bindings。`.plan` 文件需要和运行环境匹配，不能直接跨不同 TensorRT/CUDA/GPU 环境复用。

### 前端无法访问 API

确认 API 正在运行：

```bash
curl http://localhost:8000/health
```

确认前端 API 地址：

```bash
cat web/.env.local 2>/dev/null || true
```

默认应该访问 `http://localhost:8000`。如果前后端不在同一台机器，需要同时修改 `NEXT_PUBLIC_API_URL` 和后端 `APP_CORS_ORIGINS`。

### 端口被占用

默认端口：

- API：`8000`
- Web：`3000`

可以改端口启动：

```bash
API_PORT=18000 WEB_PORT=13000 ./scripts/start_conda.sh
```

对应地，前端 API 地址也要改为新的 API 端口：

```bash
API_PORT=18000 WEB_PORT=13000 NEXT_PUBLIC_API_URL=http://localhost:18000 ./scripts/start_conda.sh
```

## 11. Docker 说明

仓库中仍保留 Docker 相关文件，主要作为旧方案和迁移参考。当前推荐用户在 Ubuntu NVIDIA 主机上使用本文档中的 Conda 本地运行方式。
