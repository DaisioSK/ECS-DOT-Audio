# Dev Log

## 2025-11-14 Session

## TL;DR
- 通过 Docker + Makefile 搭建了统一的 CPU 训练环境（PyTorch、librosa、soundfile 等），Notebook 也能在容器中一键启动。
- 基于 ESC-50 构建了 glass vs background 数据流水线：路径解析、数据子集、Fold 统计、能量分析、峰值对齐、增强、缓存与 QA 试听。
- 现在的缓存策略使用 1s 窗 + 0.5s hop、70% 能量峰值阈值、前半段峰要求，生成的 mel `.npy` 与 QA wav 已可直接用于后续 CNN 训练。

## 项目状态概览
- **代码结构**：`src/` 包含配置 (`config.py`)、数据处理 (`data_utils.py`)、增强 (`augment.py` & `augment_pipeline.py`)、缓存 (`cache_utils.py`)、QA 相关 (`audio_qc.py`) 等模块，便于 Notebook 和脚本复用。
- **数据准备**：`train.ipynb` 从加载 ESC-50、分析玻璃能量、演示增强，到执行 smoke/full 缓存、生成 QA 播放列表，形成完整的数据准备链路。
- **产物**：`cache/mel64/` 下是高能窗的 log-mel 缓存与 QA wav；`cache_index_df` 描述每个窗口（标签、fold、clip/window ID、增强描述），可直接喂给训练脚本。
- **环境**：`Dockerfile` + `env.mk` 支撑构建、进入容器、启动 Notebook、运行测试；`.gitignore` 忽略了 `data/audio` 与 `cache/` 等大文件目录。

## 本次 Session 达成的目标
1. 在容器化环境中搭建 ESC-50 的数据探索与处理流水线，确保 reproducible。
2. 根据能量分析动态确定 1s 窗 + 0.5s hop，添加能量阈值和峰值要求，保证只保留真正有玻璃事件的窗口。
3. 设计增强策略（shift/gain、stretch/reverb、shift/mix、filter/gain），并写入缓存索引以供训练。
4. 搭建 QA 流程：把 mel `.npy` 近似还原为 wav，成对播放 base/aug，方便人工审听。

## 开发思路与关键改动
### 1. 环境与结构
- `Dockerfile`: 以 `python:3.10-slim` 为基底，加入 PyTorch CPU 轮、librosa/soundfile、lazygit 等工具；配套 `env.mk` 封装 build/rebuild/notebook/test 目标。
- `.gitignore`: 忽略 `data/audio/`、`cache/` 等重数据目录，防止二进制入仓。
- 目的：保持部署环境一致、仓库轻量，便于后续 CI/部署。

### 2. 数据模块化 (`src/`)
- `config.py`: 统一定义路径与默认常量，Notebook 可在运行时覆写。
- `data_utils.py`: 提供 `build_dataset`、`generate_aligned_windows` 等 helper。近期新增 `_iter_windows`、能量掩码 `_energy_mask`、峰值过滤逻辑（阈值/前半段约束），确保生成的窗口集中在玻璃能量段。
- `augment.py` + `augment_pipeline.py`: 基础增强原语 + 组合器，`run_pipeline` 会返回 `AugmentedWindow(audio, description)`，方便缓存时写入描述。
- `cache_utils.py`: 负责编排窗口生成、增强、缓存与索引；`CacheEntry` 记录 `clip_id/window_id`，`build_cache_index` 支持 energy threshold、peak ratio 等参数；QA 容易追溯 base/aug。
- `audio_qc.py`: `export_mel_to_wav` + `infer_base_mel_path`，用于 QA 试听。

### 3. Notebook 流程 (`train.ipynb`)
1. **导入 + 配置**：覆盖 `WINDOW_SECONDS=1.0`、`WINDOW_HOP=0.5`、`ENERGY_THRESHOLD=0.3`、`PEAK_RATIO_THRESHOLD=0.7` 等，确保 helper 使用当前 session 参数。
2. **数据加载与 Fold**：读取 `meta/esc50.csv`，构建 glass/background 子集，添加 `fold_id`，统计各类分布。
3. **随机可视化**：绘制波形+log-mel，直观检查数据质量。
4. **玻璃能量分析**：平滑 RMS、统计活跃时长/峰值位置、展示直方图与能量曲线，从数据得出“多数 <0.5s，极少数长尾”的结论。
5. **增强 Demo**：展示 `PIPELINE_PLAN` 的效果，生成峰值对齐窗 + 增强窗。
6. **缓存 (smoke/full)**：分别在小样本 & 全量数据上运行 `build_cache_index`，生成 `.npy` 文件和索引。（需运行后再继续 QA）
7. **QA 抽样/试听**：随机挑若干 glass 样本，导出 base/aug wav，Notebook 里成对播放，输出标签/文件路径供人工复核。

### 4. QA 试听改进
- 通过 `clip_id/window_id` 找到原始 base 窗，与增强窗一起写到 `cache/mel64/qa_audio/`；Notebook 中打印 `Label/Fold/Pipeline/Desc`，并按“Base/Aug”顺序播放器播放，便于对照。
- Mel→wav 仅用于 QA（Griffin-Lim 近似），训练中依旧使用高质量的 log-mel 特征。

### 使用示例 & 测试
1. **构建环境**：`make -f env.mk build`，然后 `make -f env.mk notebook` 启用容器 + Jupyter。
2. **运行 Notebook**：按顺序执行所有单元（smoke 缓存→全量缓存→QA 播放），生成 `cache/mel64/*.npy` 和 `cache_index_df`。QA wav 位于 `cache/mel64/qa_audio/`。
3. **快速验证**：QA 单元输出 base/aug 播放器，如需离线试听可直接双击生成的 wav。
4. **模块测试**：后续可针对 `src/` 做 unit test（例如验证 `generate_aligned_windows` 在不同阈值下的窗口数量），目前以 Notebook 运行结果为主要验证手段。

## TODO / Improvement
- **数据集类**：基于 `cache_index_df` 实现 `MelDataset` + `DataLoader`，按 fold 划分 train/val。
- **CNN Baseline**：搭建初版轻量 CNN（Conv+Pool+GAP+FC），在缓存特征上训练，记录准确率/F1。
- **量化/ONNX**：完成 QAT 或 PTQ，导出 INT8 ONNX，验证准确率再部署到 SoC。
- **增强配置管理**：将 `GLASS_PIPELINE_PLAN`、阈值等抽到 YAML/JSON，便于不同实验切换。
- **更丰富的 QA**：可选将原始 WAV 同步裁剪输出，方便肉耳对照；若需更高保真，可增加 Griffin-Lim 迭代或提升 mel 维度。
- **自动化脚本**：把 smoke/full 缓存、QA 抽样流程封装成 CLI，方便批处理或 CI 运行。


## 2025-11-15 Session (Capstone 数据准备收官)

### TL;DR
- `train.ipynb` 更名为 `prepare.ipynb`，专注数据准备；未来训练 Notebook 从输出索引起步。
- 数据管线升级：1s/0.5s 窗 + 80% 峰值阈值、六种增强组合、受控混音、背景池与 fold balancing。
- 新增 `src/datasets.py`（`balance_folds`, `MelDataset`）；Notebook 展示 base summary、fold 统计、背景 QA，支持导出 `balanced_index`。
- 环境修复：Dockerfile 锁 `numpy<2`、安装 `pyarrow`，解决 torch/Numpy 与 parquet 兼容。

### 项目状态
- **环境**：Docker image 含 PyTorch CPU 栈、numpy<2、pyarrow；`env.mk` 仍负责 build/run/notebook/test。
- **模块**：`src/data_utils`(峰值筛选)、`augement`/`augment_pipeline`(6 组合 + shift/mix 控制)、`cache_utils`(`source_type` + background pool)、`audio_qc`、`datasets`。
- **prepare.ipynb**：导入/配置覆盖 → 数据加载/Fold → 可视化 → 能量分析 → base summary → 增强 demo → 缓存 smoke/full → QA(base/aug/background) → fold balancing & dataset preview → balanced index 导出。
- **产物**：`cache/mel64/*.npy` + QA wav，`balanced_index_df` (CSV/Parquet)，供 `train.ipynb` 使用。

### 本次完成
1. Notebook 拆分 + 重命名，明确 prepare vs train 的职责。
2. 增强/缓存改进：shift 保留峰值、mix 使用正 SNR 和功率限制；PIPELINE_PLAN 扩展 6 组合（每 base 10 个增强）；背景池机制避免重复。
3. Base summary & QA：记录 offsets/总数，新增背景试听单元。
4. Dataset & Fold balancing：`balance_folds` 默认玻璃约 40%；`MelDataset` 负责加载 mel `.npy`。
5. Balanced index 导出：示例 `balanced_index_df.to_parquet('cache/index_balanced.parquet')`，train Notebook 直接读取。
6. 环境修复：Dockerfile 添加 `numpy<2`、`pyarrow`，避免 torch/Numpy 报错。

### 关键改动
- `augment.py`: `augment_time_shift` 零填充 + 限制峰位置；`mix_with_background` SNR=(3,9) & bg_max=0.1。
- `augment_pipeline.py`: 注册 6 种组合（shift_gain/stretch_reverb/shift_mix/filter_gain/gain_mix/stretch_filter）。
- `cache_utils.py`: `CacheEntry` 加 `clip_id/window_id/source_type`；背景窗全部入池，比例由 `balance_folds` 控制。
- `prepare.ipynb`: 配置打印、base summary（含总数/offset）、fold balancing & Dataset preview、背景 QA。
- `src/datasets.py`: `balance_folds` 调整玻璃:背景≈40:60；`MelDataset` 加载 mel `.npy`。
- `Dockerfile`: 加 `pyarrow`、锁 `numpy<2`。

### 使用示例
```bash
make -f env.mk build
make -f env.mk notebook
```
运行 `prepare.ipynb` → `balanced_index_df.to_parquet('cache/index_balanced.parquet', index=False)`。
在新的 `train.ipynb`：
```python
import pandas as pd
from src.datasets import MelDataset
index_df = pd.read_parquet('cache/index_balanced.parquet')
train_dataset = MelDataset(index_df[index_df['fold_id'].isin([1,2,3,4])])
val_dataset = MelDataset(index_df[index_df['fold_id']==5])
```

### TODO / Improvements
- 训练/验证：构建 CNN Baseline、交叉验证，并在 train Notebook 中记录指标。
- 量化/ONNX：完成 QAT/PTQ 并导出 INT8 模型。
- 配置抽象：将窗口、阈值、pipeline copy 等移至 YAML/JSON。
- 背景池优化：可按类别聚类、设最大样本数。
- 自动 QA：除音频播放外，可计算 embedding 或 Mel 距离辅助筛查。
