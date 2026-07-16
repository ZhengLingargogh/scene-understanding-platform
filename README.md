# Scene Understanding Platform

面向视觉定位与场景理解的 Web 平台：**FastAPI** 后端 + **React (Vite)** 前端。支持场景/数据集管理、模型注册、单图推理与场景级批量推理，已接入 SALAD、NetVLAD、SuperPoint、SIFT、LightGlue、SAM 等真实模型。

## 目录

- [功能概览](#功能概览)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [模型权重与数据集](#模型权重与数据集)
- [前端路由](#前端路由)
- [项目结构](#项目结构)
- [数据集说明](#数据集说明)
- [已接入模型](#已接入模型)
- [插件化架构](#插件化架构)
- [API 概览](#api-概览)
- [实现状态](#实现状态)
- [环境排错](#环境排错)
- [仓库说明](#仓库说明)
- [更多文档](#更多文档)

## 功能概览

| 模块 | 说明 |
|------|------|
| **场景管理** | 内置 CrossLoc / UAVD4L；支持扫描目录注册自定义数据集；推理页下拉与 catalog 同步 |
| **模型管理** | 6 个内置模型，展示能力标签；支持卸载（内置模型可恢复） |
| **单图推理** | 图像检索、双图匹配可视化、局部关键点可视化、SAM 点提示交互分割 |
| **场景推理** | 批量特征提取、SAM 批量语义分割（自动掩码 + manifest） |

### 单图推理 Hub

| 功能 | 路径 | 卡片描述 |
|------|------|----------|
| 图像检索 | `/inference/retrieval` | Top-8 相似图检索 |
| 图像匹配 | `/inference/matching` | 双图匹配可视化 |
| 可视特征 | `/inference/feature-visualization` | 局部关键点可视化 |
| 交互分割 | `/inference/interactive-segmentation` | 点提示交互分割 |

### 场景推理 Hub

| 功能 | 路径 | 卡片描述 |
|------|------|----------|
| 特征提取 | `/scene-inference/feature-extraction` | 提取场景局部/全局特征 |
| 语义分割 | `/scene-inference/semantic-segmentation` | 自动掩码生成并保存掩码图 |

各推理子页面左上角 **「功能选择」** 为下拉菜单，可在同一 Hub 内切换模块，无需返回上级。

## 技术栈

| 层级 | 技术 |
|------|------|
| 后端 | Python 3.10+、FastAPI、SQLAlchemy、SQLite |
| 前端 | React 18、TypeScript、Vite |
| 推理 | PyTorch、OpenCV、Kornia |
| 第三方 | [segment-anything](third_party/segment-anything)、[LightGlue](third_party/lightglue)、SALAD、NetVLAD |

## 快速开始

### 环境要求

- **Python** 3.10+（推荐 Conda 独立环境，如 `sup`）
- **Node.js** 18+
- **NVIDIA GPU + CUDA**（推荐；CPU 可跑部分功能但较慢）
- **PyTorch** 需单独安装（见下方「方式 B：Conda 环境 sup」或 `backend/requirements.txt` 头部说明）

### 1. 克隆仓库

```bash
git clone <your-repo-url> scene-understanding-platform
cd scene-understanding-platform
```

### 2. 后端

#### 方式 A：venv

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
bash scripts/fix_sup_opencv.sh
```

#### 方式 B：Conda 环境 `sup`（推荐）

```bash
conda create -n sup python=3.10 -y
conda activate sup
cd backend

# 1) PyTorch（按显卡/CUDA 选择；示例为 CUDA 11.8）
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1: 将 cu118 改为 cu121
# 仅 CPU:    pip install torch==2.2.2 torchvision==0.17.2

# 2) 其余 Python 依赖（含 SAM）
pip install -r requirements.txt

# 3) 修复 OpenCV / NumPy 兼容性（必做，勿用 conda install opencv）
bash scripts/fix_sup_opencv.sh
```

> **说明**：LightGlue 通过 `third_party/lightglue` 加入 `sys.path`，无需单独 `pip install`。

启动服务（**必须在 `backend` 目录下**）：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 <http://127.0.0.1:5173>。Vite 开发服务器将 `/api` 与 `/health` 代理到后端 `8000` 端口。

### 4. 验证

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0"}
```

- 交互式 API 文档：<http://127.0.0.1:8000/docs>
- OpenAPI JSON：<http://127.0.0.1:8000/openapi.json>

## 模型权重与数据集

以下内容**不包含在 Git 仓库中**，需自行准备（见 `.gitignore`）。

### 模型权重

| 模型 | 路径 |
|------|------|
| SALAD | `models/salad/dino_salad.ckpt` |
| NetVLAD | `models/netvlad/vgg16_netvlad.pth` |
| SAM (ViT-B) | `models/sam/sam_vit_b_01ec64.pth` |
| SuperPoint / LightGlue | `third_party/lightglue`（部分权重首次运行时自动下载） |

### 内置数据集路径

内置 CrossLoc / UAVD4L 根目录在 `backend/app/services/dataset_catalog.py` 中配置。默认示例如下，部署时请改为本机路径：

```
datasets/crossloc/
datasets/uavd4l/
```

目录结构：`{scene}/{train|test}/rgb/`。

### 运行时数据目录

| 路径 | 说明 |
|------|------|
| `backend/data/` | SQLite、`custom_datasets.json`、特征/分割输出 |
| `backend/uploads/` | 上传的 Query 图像 |

首次启动会自动创建 `data/` 并种子同步内置场景到 SQLite。

## 前端路由

| 路径 | 功能 |
|------|------|
| `/scenes` | 场景管理 |
| `/models` | 模型管理 |
| `/inference` | 单图推理 Hub |
| `/inference/retrieval` | 图像检索 |
| `/inference/matching` | 双图匹配可视化 |
| `/inference/feature-visualization` | 局部关键点可视化 |
| `/inference/interactive-segmentation` | 点提示交互分割 |
| `/scene-inference` | 场景推理 Hub |
| `/scene-inference/feature-extraction` | 批量特征提取 |
| `/scene-inference/semantic-segmentation` | 批量语义分割 |

## 项目结构

```
scene-understanding-platform/
├── README.md
├── 项目设计文档.md          # 架构与规划
├── backend/
│   ├── app/
│   │   ├── main.py              # 应用入口、启动时 DB 与 catalog 同步
│   │   ├── config.py            # 路径、设备、媒体浏览白名单
│   │   ├── api/v1/              # REST 路由
│   │   │   ├── scenes.py        # 场景 + 自定义数据集注册
│   │   │   ├── models.py
│   │   │   ├── inference.py     # 单图推理、特征可视化、交互分割
│   │   │   ├── benchmark.py     # 特征提取 / 语义分割异步任务
│   │   │   └── media.py         # 本地图像预览
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── dataset_catalog.py      # 内置数据集（硬编码）
│   │   │   ├── dataset_registry.py     # 自定义数据集持久化
│   │   │   ├── dataset_scanner.py      # 目录扫描与校验
│   │   │   ├── feature_extractors/     # SALAD、SuperPoint、SIFT、NetVLAD
│   │   │   ├── retrievers/             # SALAD、NetVLAD 检索
│   │   │   ├── matchers/               # LightGlue
│   │   │   ├── segmenters/             # SAM 交互分割
│   │   │   ├── semantic_segmenters/    # SAM 批量语义分割
│   │   │   ├── plugins/registry.py     # 插件注册表
│   │   │   └── benchmark_jobs.py       # 异步任务与进度
│   │   └── db/                  # SQLAlchemy 模型与会话
│   ├── scripts/
│   │   └── fix_sup_opencv.sh    # OpenCV / NumPy 环境修复
│   └── requirements.txt         # Python 依赖（含 sup 环境安装说明）
├── frontend/
│   └── src/
│       ├── pages/               # 各功能页面
│       ├── components/inference/
│       ├── config/inferenceHubModes.ts
│       └── hooks/useDatasetSceneSelection.ts
├── models/                      # 权重目录（需自行放置）
├── datasets/                    # 数据集目录（可选，与 catalog 配置一致）
└── third_party/                 # segment-anything、lightglue、salad、netvlad
```

## 数据集说明

### 双轨 Catalog

- **内置**：`dataset_catalog.py` 中的 CrossLoc、UAVD4L
- **自定义**：`backend/data/custom_datasets.json`，注册后合并进 `GET /scenes/catalog` 与推理用 `mock_datasets`

### 自定义数据集目录约定

在场景管理页选择数据集根目录（如 `.../Cambridge`），`family_id` 取目录 basename（slug 化）。每个**场景**子目录需同时包含 `train` 与 `test`，且每个 split 下必须有：

```
DatasetRoot/
├── SceneA/
│   ├── train/
│   │   ├── calibration/    # 内参
│   │   ├── poses/          # 外参
│   │   └── rgb/            # 图像（至少一张）
│   └── test/
│       ├── calibration/
│       ├── poses/
│       └── rgb/
└── SceneB/ ...
```

注册 API：`POST /api/v1/scenes/datasets/register`（可先 `POST .../scan` 预览）。

## 已接入模型

| model_id | 名称 | 能力 |
|----------|------|------|
| `salad` | SALAD | 特征提取、图像检索 |
| `netvlad` | NetVLAD | 特征提取、图像检索 |
| `superpoint` | SuperPoint | 特征提取、关键点检测 |
| `sift` | SIFT | 特征提取、关键点检测 |
| `lightglue` | LightGlue | 图像匹配 |
| `sam` | SAM (ViT-B) | 交互分割、语义分割 |

推理时插件会在首次调用时自动加载权重；模型管理页可对内置模型执行**卸载**（记录至 `data/removed_models.json`，重启或列表同步后可恢复未卸载项）。

## 插件化架构

```
model_id → plugins/registry.py → 具体 Plugin 单例
```

| 插件类型 | 目录 | 用途 |
|----------|------|------|
| `FeatureExtractor` | `feature_extractors/` | 场景批量特征提取 |
| `Retriever` | `retrievers/` | 单图 Top-K 检索 |
| `Matcher` | `matchers/` | 双图稀疏匹配 |
| `Segmenter` | `segmenters/` | 点提示交互分割 |
| `SemanticSegmenter` | `semantic_segmenters/` | 批量语义分割 |

场景推理任务在后台线程执行，通过 `job_id` 轮询进度（0–100%）。

## API 概览

前缀：`/api/v1`（健康检查为 `/health`）。

### 场景与数据集

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/scenes/catalog` | 数据集目录（内置 + 自定义 + 注册状态） |
| GET | `/scenes/datasets/browse` | 浏览允许的本地目录 |
| POST | `/scenes/datasets/scan` | 扫描预览（不写入） |
| POST | `/scenes/datasets/register` | 注册自定义数据集 |
| DELETE | `/scenes/datasets/{family_id}` | 删除自定义数据集 |
| GET/POST | `/scenes` | SQLite 场景列表 / 创建 |
| GET/PATCH/DELETE | `/scenes/{id}` | 场景 CRUD |

### 模型

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/POST | `/models` | 列表 / 注册 |
| GET | `/models/{id}` | 详情 |
| POST | `/models/{id}/load` | 预加载权重 |
| POST | `/models/{id}/unload` | 卸载权重 |
| DELETE | `/models/{id}` | 卸载并移除（内置模型写入 removed 记录） |

### 单图推理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/inference/datasets` | 可用于检索的数据集列表 |
| POST | `/inference/run` | 流水线推理（检索 / 匹配等） |
| POST | `/inference/feature-visualization` | 关键点 + 热力图 |
| POST | `/inference/interactive-segmentation/session` | 创建 SAM 会话 |
| POST | `/inference/interactive-segmentation/predict` | 点提示预测 Mask |

### 场景推理（Benchmark 任务）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/benchmarks/feature-extraction/run` | 启动特征提取 |
| GET | `/benchmarks/feature-extraction/{job_id}` | 查询进度 |
| POST | `/benchmarks/semantic-segmentation/run` | 启动语义分割 |
| GET | `/benchmarks/semantic-segmentation/{job_id}` | 查询进度 |

### 媒体

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/media/file?path=...` | 本地图像预览（路径需在白名单内） |

### 调用示例

**图像检索：**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/inference/run \
  -F "scene_id=nature" \
  -F "dataset_id=crossloc-nature-test" \
  -F "model_id=salad" \
  -F 'pipeline_stages=["image_retrieval"]' \
  -F "image=@query.jpg"
```

**批量特征提取：**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/benchmarks/feature-extraction/run \
  -H "Content-Type: application/json" \
  -d '{"model_id":"salad","dataset_id":"crossloc-nature-train","output_path":"data/features/crossloc-nature-train"}'
```

**批量语义分割：**

```bash
curl -X POST http://127.0.0.1:8000/api/v1/benchmarks/semantic-segmentation/run \
  -H "Content-Type: application/json" \
  -d '{"model_id":"sam","dataset_id":"crossloc-nature-train","output_path":"data/segmentation/crossloc-nature-train"}'
```

语义分割输出：`{stem}_mask.png` + `manifest.json`。

## 实现状态

| 能力 | 状态 |
|------|------|
| 场景管理（内置 + 自定义注册/删除） | ✅ |
| 模型管理（6 内置模型） | ✅ |
| SALAD / NetVLAD 图像检索 | ✅ |
| SuperPoint / SIFT 特征与可视化 | ✅ |
| LightGlue 双图匹配 | ✅ |
| SAM 点提示交互分割 | ✅ |
| 场景批量特征提取（真实进度） | ✅ |

## 环境排错

### OpenCV `findContours` 报错

Conda 自带的 OpenCV 4.13 与 NumPy 1.26 存在 ABI 不兼容。在已激活的 Conda 环境中执行：

```bash
cd backend && bash scripts/fix_sup_opencv.sh
```

或手动固定版本：

```bash
pip install "numpy>=1.24,<2"
pip install opencv-python-headless==4.10.0.84
```

### PyTorch / CUDA

请使用与显卡驱动匹配的 PyTorch 版本；推荐 `torch 2.2.2` + `torchvision 0.17.2` + `numpy<2`，避免 NumPy 2.x 导致部分扩展库异常。安装示例见上文「方式 B：Conda 环境 sup」。

### 自定义数据集浏览路径

注册时的目录浏览受 `backend/app/config.py` 中 `media_allowed_roots` 限制，部署时请加入本机数据集父目录。

## 仓库说明

以下内容默认**不提交**到 Git（见 `.gitignore`）：

- `backend/data/`、`backend/uploads/`
- `backend/.venv/`、`frontend/node_modules/`、`frontend/dist/`
- `.env`

上传 GitHub 前请确认未包含大体积权重、私有数据集路径或密钥。

## 更多文档

完整架构设计、定位流水线规划与模块演进见 [项目设计文档.md](./项目设计文档.md)。

## License

如无另行说明，本项目代码仅供学习与研究使用。第三方模型（SAM、LightGlue、SALAD、NetVLAD 等）与数据集（CrossLoc、UAVD4L 等）请遵循各自许可协议。
