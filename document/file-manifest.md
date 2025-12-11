# File Manifest

```text
.
├── Dockerfile                     # 基于 python:3.10-slim 的 CPU 镜像，预装 PyTorch/onnxruntime/librosa/pyarrow 等训练推理依赖
├── env.mk                         # make 配方：镜像 build/rebuild、启动容器/Notebook、运行测试的统一入口
├── dev_log.md                     # 全项目开发历程与里程碑总结，含每次 session 的目标、改动、TODO
├── case_study.ipynb               # 多标签事件级混音与滑窗推理 Notebook，含可视化/试听/指标汇总
├── prepare.ipynb                  # 早期单标签数据准备流水线（加载/能量分析/缓存/QA），保留兼容
├── prepare_new.ipynb              # 主力多源多标签准备：去重/重采样/分折/能量与分窗 QA/增强/缓存/平衡索引导出
├── train.ipynb                    # K 折训练与可视化，输出最佳 checkpoint 和 ONNX，含指标表/曲线/混淆矩阵
├── infer.ipynb                    # 读取 mel 样本，Torch vs ONNX 前向对比、概率柱状图与 delta 校验
├── cache/                         # 运行后生成的特征与中间产物：mel64 .npy、QA wav、索引 CSV/Parquet 等（不列举具体文件）
├── cache_old/                     # 旧版本缓存的备份/归档，防止覆盖
├── data/                          # 原始与清洗后数据根目录（不展开具体音频）
│   ├── esc50/                     # ESC-50 数据集原始文件、官方 meta/README/LICENSE 等资源
│   ├── freesound/                 # freesound 抓取的玻璃音频存放处及获取说明
│   ├── gunshot_kaggle/            # Kaggle 枪声数据存放处及获取说明
│   ├── meta/                      # 统一 schema 的元数据 CSV 与构建脚本
│   │   ├── esc50.csv              # ESC-50 统一字段 meta：filepath/label/source/fold/时长/格式/md5
│   │   ├── gunshot_kaggle.csv     # 枪声音频 meta：含 weapon_id 等 extra_meta 字段
│   │   ├── freesound.csv          # freesound 玻璃音频 meta，符合统一 schema
│   │   └── build_freesound_meta.py # 从 freesound 音频生成/更新 meta CSV 的脚本
│   └── esc50/...                  # 其余示例/说明/图片等辅助文件
├── tools/                         # 实用脚本集合
│   ├── convert_to_wav.py          # 通用音频转 WAV 的 CLI，递归扫描多格式，支持可选重采样与下混，保持目录结构
│   └── resample_audio.py          # 批量重采样到 22.05k mono 并写入新 meta 的脚本，便于离线预处理/去除依赖
└── src/                           # 核心库代码（Notebook 与 CLI 共享）
    ├── __init__.py                # 包初始化
    ├── config.py                  # 全局路径/标签空间/采样率/mel 参数与 case study 默认配置
    ├── data_utils.py              # 音频加载、重采样/下混、静音裁剪、窗口生成与能量/峰值筛选、mel 计算
    ├── augment.py                 # 基础增强算子：时移/增益/伸缩/混音/滤波等
    ├── augment_pipeline.py        # 增强组合计划与执行器，返回带描述的 AugmentedWindow
    ├── cache_utils.py             # orchestrator：按窗口与增强生成 log-mel、写缓存与索引、导出 QA 音频
    ├── datasets.py                # 多标签 MelDataset、batch padding、fold 平衡/采样辅助
    ├── folds.py                   # 折分与平衡相关的工具函数（补充 datasets 逻辑）
    ├── meta_utils.py              # 统一 meta 加载/去重/映射、分层折分、标签 ID 附加、多标签平衡
    ├── models.py                  # TinyGlassNet 等轻量 CNN 模型定义，适配 ONNX/Edge
    ├── training.py                # 训练与 K 折评估循环，含 class weight、grad clip、LR 调度、指标记录与导出
    ├── inference.py               # Torch/ONNX 推理统一接口，批量前向与概率输出
    ├── event_detection.py         # 长音频背景床构建、随机叠加事件、滑窗推理、事件合并/匹配与指标
    ├── case_study_cli.py          # 事件检测 CLI：多标签混音、推理、评估与报表输出
    ├── audio_qc.py                # 将 mel 近似还原/导出为 wav 供 QA 试听，路径推断辅助
    ├── export.py                  # 模型导出/ONNX 相关辅助函数
    ├── metrics.py                 # 分类/检测指标计算、混淆矩阵等工具
    └── viz.py                     # 波形、mel、时间轴与概率可视化工具，服务 Notebook/报告
```
