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

## 2025-11-17 Session

### 大图位置
- **Sprint**：Capstone Sprint #2「训练 & 评估」。目标：拿到可用 baseline、补齐评估/导出/推理链路，为 TinyML SoC demo 做准备。
- **Task**：Task-1 模型/训练模块化；Task-2 K-fold 评估与分析；Task-3 ONNX 导出与环境；Task-4 推理 demo。

### TL;DR
- 重写训练栈（TinyGlassNet + grad clip + class weight + K-fold），让 baseline 更贴近 MCU 约束且可重复。
- 自动化评估：运行全量 K-fold，输出指标表/可视化，自动挑选最佳折并导出统一 checkpoint & ONNX。
- 推理链路 ready：`infer.ipynb` + `src/inference.py` 演示 Torch vs ONNX，方便 QA/调参/现场 demo。

### 项目状态（宏观→微观）
- **宏观**：数据管线稳定，开始进入“模型训练、评估、导出、推理”闭环；后续重点转向量化与 MCU 验证。
- **训练组件**：
  - `src/models.TinyGlassNet`：3×Conv+ReLU+Pool + GAP + FC，参数受控、只用 MCU 支持的算子。
  - `src/training`：`train_model` + `run_kfold_training`，封装 ReduceLROnPlateau、class weight、grad clipping、lr logging。
  - `train.ipynb`：仅 orchestrate，执行 K-fold → 可视化 → 选最佳 fold → 混淆矩阵 → checkpoint/ONNX 输出。
- **推理组件**：
  - `src/inference.py`：加载 checkpoint/ONNX、批量 mel、Torch/ORT 前向统一接口。
  - `infer.ipynb`：抽样 mel 样本 → Torch/ONNX 推理 → 概率柱状图 → delta 验证。
- **环境**：Dockerfile 增装 `onnx`/`onnxruntime`；`.gitignore` 忽略任何 `.ipynb_checkpoints`；容器即可运行训练与推理。

### 本次 Session 达成
1. 实现 TinyGlassNet + 通用训练/评估模块；训练 loop 支持 LR 调整、Grad Clip、class weight。
2. 训练 Notebook 改为全 K-fold 流程：指标表/条形图、Loss/F1 曲线、最佳折选取、混淆矩阵。
3. 自动导出 `tinyglassnet_best.{pt,csv,onnx}`，infer Notebook 默认读取统一文件名。
4. `infer.ipynb` + `src/inference.py` 完成 Torch/ONNX 推理示例、概率可视化；用于 smoke/demo。
5. Dockerfile 加载 onnx 依赖，推理/导出无需额外 pip。

### 开发思路 & 关键改动
- **模型/训练**：TinyGlassNet 只用 Conv/Pool/ReLU/FC，保证可 ONNX → MCU；class weight= (background, glass)=(1.0,1.3) + grad clip=1.0 解决 fold 失衡/梯度尖峰；记录 lr、precision、recall、f1 供曲线分析。
- **K-fold**：`run_kfold_training` 按 fold 生成 checkpoint+history+metrics，`BEST_METRIC` 默认 F1 选最优；输出统一 `tinyglassnet_best.pt` 明确推理入口。
- **可视化**：K-fold 条形图（单色+顶部注数）替代 hue palette，避免 warning；confusion matrix 根据占比自动切换文字颜色，数字大而清晰；训练曲线帮助定位 epoch 调整点。
- **ONNX & 推理**：`export_to_onnx` opset=13，time 轴设动态；infer Notebook 对比 Torch/ONNX `Max prob delta` 作为导出验证。
- **环境**：Dockerfile 安装 onnxruntime，`.gitignore` 覆盖 checkpoint 目录，保证 repo 干净。

### Insight / 巧思
- K-fold 五折指标集中（recall 0.87–0.92，std≈0.02），验证了数据窗口/增强策略在正样上留足信息，class weight 起到了稳定作用。
- 统一输出 best checkpoint，减少后续脚本对 fold 的耦合；即使将来更换 BEST_METRIC 也能无缝覆盖。
- 推理 Notebook 直接对 mel `.npy` 做 batch 前向并对比 ORT，可作为导出 smoke 与 demo 入口，避免“结果黑盒”。

### 使用示例 / 验证
```bash
make -f env.mk build        # 构建 Docker
make -f env.mk notebook     # 容器内运行 train.ipynb（可先 SMOKE_TEST=True）
```
- 训练完成后生成 `cache/experiments/tinyglassnet_best.{pt,csv,onnx}`。
- 推理：在 Notebook 中执行 `infer.ipynb`，输出 Torch vs ONNX 概率差（应 ≈1e-5）与概率柱状图。

### TODO / Improvements
1. **量化/INT8**（继承）：加入 PTQ/QAT 流程、生成 INT8 ONNX，并评估精度损耗。
2. **Monte-Carlo 推理**：让 `infer.ipynb` 支持随机原始 WAV → pipeline → 推理，验证端到端鲁棒性。
3. **自动化测试**：补 `tests/`（windowing、balance_folds、MelDataset、run_kfold_training 分割、Torch vs ONNX delta 等）。
4. **标签扩展**：`datasets.py` 仍假设单正类，未来引入 gunshot 时需升级 label mapping 与 loss。
5. **Profiling**：记录 TinyGlassNet 参数量/MACs/内存 footprint，提前评估 MCU 可部署性。
6. **增强策略升级**（继承）：视训练表现，决定是否引入双窗口/多峰采样或更精细的背景混音。

## 2025-11-15 Code Review & Follow-up

### 结论
- 数据分布抽样已完成，人耳可辨的窗口质量总体可接受，当前“峰值放在前半段”策略继续保留作为 baseline。
- `generate_aligned_windows` 的 `extra_shifts` 兜底逻辑存在 `break` 过早退出的问题，实际并未尝试额外偏移；需在后续迭代中修复以恢复错位补偿能力。
- `augment_time_shift` 的“只允许向内移动”逻辑符合“峰值靠前”策略，但需要在未来保留少量随机 shift 以防模型过度依赖绝对位置（放入 improvement backlog，待首个训练结果后评估）。
- mix 增强阶段会对同一背景 clip 重复 `load_audio`，建议在 `_cache_glass_row` 内缓存背景 waveform 或引入 LRU，以减少 I/O 成本。
- `datasets.py` 目前假定只有一个正类，将来扩展 gunshot 时需要动态构建 `LABEL_TO_ID` 并支持多正类采样。
- 没有单元测试；建议至少补上 `generate_aligned_windows`、`balance_folds`、`MelDataset` 的轻量 pytest，以便后续 refactor 时有安全网。

### Action Items
1. 修复 `extra_shifts` 的 break 逻辑，确保错位兜底真正生效。
2. 为 mix 背景采样添加 waveform 缓存，降低缓存阶段的磁盘负载。
3. `datasets.py` 改为根据 `POSITIVE_LABELS` 动态生成 label→id 映射，提前兼容多任务。
4. 规划最小单测集（windowing、fold balance、dataset getitem），待 baseline 训练前后择机补齐。
5. 观察首轮训练结果后，再决定是否引入“双窗口/多峰”增强策略。


## 2025-12-09 10:59:25 +08 Session (Capstone case study & event detection)

### 大图位置
- **Sprint**：Capstone Sprint #3「事件检测验证」。聚焦真实/合成长音频上的玻璃事件检测体验与评估。
- **Task**：Task-1 构建可控混音流水线；Task-2 滑窗推理 + 事件合并评估；Task-3 体验回放与可视化。

### TL;DR
- 新增事件级推理 helper（背景床 + 随机叠加 glass + SNR 控制 + 非静音分段），支持真实场景的长音频事件检测。
- `case_study.ipynb` 重建：生成 60s 背景床，随机放置/重叠玻璃片段，自动切分真值事件，滑窗推理（Torch/ONNX）、事件合并、P/R/F1、可视化、试听。
- 真值标注从“整段 clip”改为“非静音子段”，减少空白/多事件误差；混音允许多 SNR，评估更贴近现场噪声。

### 项目状态（宏观→微观）
- **宏观**：数据准备、训练、推理链路已成型；新增事件检测验证路径，补齐长音频滑窗与事件评估。
- **混音与真值**：可构建 40–60s 背景床，随机顺序/间隔叠加 glass，自动非静音分段生成真值（含 SNR）。
- **推理与评估**：滑窗 log-mel (1s/0.5s) → TinyGlassNet Torch/ONNX → 阈值合并 → 容差匹配，输出 TP/FP/FN、P/R/F1，附试听和时间轴可视化。
- **素材**：`data/external/glass_ext_01..05.*` + ESC-50 非玻璃片段作为背景池。

### 本次完成
1. 新建事件检测 helper（`src/event_detection.py`）：背景床构建、随机叠加、SNR 调制、非静音分段、滑窗生成、事件合并/匹配。
2. 重建 `case_study.ipynb`：参数化混音/评估，支持试听与可视化；默认 60s 底床、随机起点/间隔、SNR 3–9dB、允许轻度重叠。
3. 统一外部玻璃片段命名（`glass_ext_01..05`），确保批处理与日志可读。

### 开发思路与关键改动
- 背景→痛点→方案：原版事件集中在前半，真值粗糙（整段 clip），背景过大可能淹没 glass。新方案随机排序/起点/间隔（含负间隔允许重叠），叠加时按目标 SNR 缩放玻璃，非静音分段提取真实事件（避免空白/多事件偏差）。
- `src/event_detection.py`：
  - `build_background_bed`：循环拼接背景池至目标时长，15ms 淡入淡出。
  - `mix_glass_on_bed`：随机顺序、随机起点/间隔、可负间隔（轻度重叠）；按 `SNR_RANGE_DB` 与背景 RMS 调整玻璃能量；非静音分段 (`librosa.effects.split`) 生成精确事件，记录 SNR。
  - 其它：`detect_events_in_clip`、`_scale_to_snr`、保留滑窗/合并/匹配/推理封装。
- Notebook (`case_study.ipynb`)：
  - 配置参数化：背景增益、玻璃增益、起点/间隔范围、SNR 范围、分段阈值/最小时长、阈值/合并间隔/容差。
  - 流程：收集素材→背景床→叠加并切分真值→试听+真值打印（含 SNR）→滑窗 mel→Torch/ONNX 推理→事件合并/评估→时间轴可视化。

### Insight / 巧思
- 真值用“非静音子段”替代整段，解决 clip 内静音/多击碎导致的误差，评估更公平。
- 随机时间戳+可重叠间隔让事件分布均衡，避免模型只在前半段被测试。
- SNR 控制在混音时完成（对玻璃缩放），可便捷生成不同难度场景；打印 SNR 便于关联误报/漏报。

### 使用示例 / 验证
```bash
make -f env.mk notebook   # 容器内打开 case_study.ipynb
```
按序运行 notebook：
1) 构造 60s 背景床，叠加 5 个 glass（随机顺序/间隔/重叠，SNR 3–9dB），保存 `cache/case_study/mix.wav`。
2) 试听混音，查看真值事件起止（含 SNR）。
3) 滑窗 mel → Torch/ONNX 推理 → 事件合并/评估（TP/FP/FN、P/R/F1）→ 概率时间轴标注 GT/Pred。

### TODO / Improvement
- 阈值与窗口：基于分段真值再调优阈值/merge_gap/tolerance，分 SNR 桶输出命中率。
- 更逼真背景：支持不同类型背景的最大占比/长度约束，避免单类背景占主导。
- 质量指标：增加延迟统计（预测事件中心 vs 真值中心偏移均值/分位数）。
- 自动化：封装 case study 为 CLI（给定参数自动生成混音、跑推理、输出指标）。此前 TODO（训练/量化/配置抽象/QA 自动化）仍需延续。


## 2025-12-09 11:42:37 +08 Session (Engineering hardening: config, run_id, CLI)

### 大图位置
- **Sprint**：Capstone Sprint #3「事件检测验证」持续工程化。
- **Task**：Task-Eng-1 参数集中化与配置快照；Task-Eng-2 防覆盖 run_id 输出；Task-Eng-3 提供 CLI 便于批量/CI 运行。

### TL;DR
- 集中 case study 默认参数/路径/版本到 `config.py`，避免散落硬编码。
- 事件检测 helper 重构：抽取交叉淡入工具、支持外部 RNG 注入，减少重复。
- 新增 `case_study_cli.py`：生成 run_id 目录，保存 config/结果/mix.wav，便于复现与批量跑。
- Notebook 同步：引用集中 defaults，保存 `run_config.json`，统一 RNG_SEED，输出路径标准化。

### 本次完成
1. `src/config.py`：新增 `CASE_STUDY_DEFAULTS`、`CASE_STUDY_DIR`、`CASE_STUDY_SCHEMA_VERSION`，集中背景增益/间隔/SNR/阈值等默认值与路径。
2. `src/event_detection.py` 重构：
   - 提取 `_crossfade_blend` 复用淡入淡出/交叉淡入，减少重复。
   - `mix_glass_on_bed` 支持外部 RNG 注入（或 seed），保持 SNR 缩放/非静音分段逻辑。
3. `case_study.ipynb` 重建：
   - 读取 config defaults，记录 `run_config.json`，统一 RNG_SEED。
   - 输出路径从 `CASE_STUDY_DIR` 取得，仍生成 `mix.wav` 等结果。
4. 新增 `src/case_study_cli.py`：
   - `python -m src.case_study_cli --seed 42 --config my_cfg.json --output cache/case_study_runs`。
   - 自动生成 run_id 子目录，落盘 config、mix.wav、results.json，避免覆盖，便于批量/CI。

### 开发思路与关键改动
- 背景→痛点：硬编码分散、输出易覆盖、随机性不可追溯。
- 方案：集中 defaults + 版本号，run_id 输出目录，配置/结果快照；Helper 抽象交叉淡入、统一 RNG 注入。
- 实现：config 添加 defaults/schema；event_detection 抽 `_crossfade_blend`、rng 参数；notebook 消费集中参数并写 config；CLI 路径/seed/run_id 打通。

### 使用示例 / 验证
- CLI：`python -m src.case_study_cli --seed 123 --config my_cfg.json --output cache/case_study_runs`
  - 输出：`cache/case_study_runs/run_<timestamp>/` 下含 `run_config.json`、`mix.wav`、`results.json`。
- Notebook：`make -f env.mk notebook` 打开 `case_study.ipynb`，参数默认取自 config，生成 `cache/case_study/run_config.json` 与 `mix.wav`。

### TODO / Improvement（继承）
- 阈值/merge/tolerance 基于分段真值调优，分 SNR 桶报表。
- 背景池约束与 SNR 分桶延迟统计。
- CLI 增加阈值搜索/多 run 批量；单测/CI/lint 待补；自动化导出/评估脚本仍可推进。
