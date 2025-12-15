# Module Interfaces（src/）
按“文件 → 签名 → 描述”三级结构列出 `src/` 下所有类与函数，涵盖输入/输出要点、示例与使用场景，语言尽量口语化。

## config.py
- `locate_project_root(start: Path | None = None, marker: str = "data") -> Path`
  - 自下而上寻找包含 marker 的目录并返回项目根，找不到就抛错。常用于脚本启动时定位根目录，避免相对路径出错。
- 常量：`PROJECT_ROOT`, `DATA_ROOT`, `CACHE_DIR`, `SR`, `N_MELS`, `N_FFT`, `HOP_LENGTH`, `WINDOW_SECONDS`, `WINDOW_HOP`, `TARGET_LABELS`, `LABEL_TO_ID`, `NUM_CLASSES`, `BACKGROUND_LABEL`, `CASE_STUDY_*` 等。
  - 统一的路径、标签空间、采样率和窗口参数，Notebook 与脚本都会读这些默认值。
- 数据与缓存：`META_FILES`（多源 meta 列表）、`RAW_AUDIO_ROOTS`（源到原始音频根的映射，兼容缺失 filepath 场景）、`CACHE_ROOT`/`CACHE_MEL64`（缓存根及默认 mel64 路径），`AUDIO_DIR`/`META_FILE` 作为 legacy 兼容。

## audio_qc.py
- `mel_db_to_waveform(mel_db: np.ndarray, sr: int = SR, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_iter: int = 32) -> np.ndarray`
  - 把 log-mel(dB) 近似还原成波形，用于快速试听。例：`y = mel_db_to_waveform(mel)`.
- `export_mel_to_wav(mel_path: Path, out_path: Path, sr: int = SR, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_iter: int = 32) -> Path`
  - 读取保存的 `.npy` mel，重构并写成 wav，返回输出路径。例：`export_mel_to_wav(Path("cache/...npy"), Path("qa.wav"))`.
- `infer_base_mel_path(path: Path, clip_id: str, window_id: str) -> Path`
  - 根据 clip/window id 从增强样本推断原始 base 窗路径，方便对照 QA。

## augment.py
- `augment_time_shift(y: np.ndarray, sr: int = SR, max_shift: float = 0.15) -> np.ndarray`
  - 随机平移窗口，同时限制峰值不跑出窗外，用于轻量时移扰动。
- `augment_time_stretch(y: np.ndarray, rate_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray`
  - 拉伸/压缩时长并填补到原长度，模拟节奏变化。
- `augment_gain(y: np.ndarray, db_range: Tuple[float, float] = (-5.0, 5.0)) -> np.ndarray`
  - 随机增减响度，丰富音量分布。
- `mix_with_background(y: np.ndarray, background: np.ndarray, snr_db_range: Tuple[float, float] = (3.0, 9.0), bg_max_ratio: float = 0.1) -> np.ndarray`
  - 按随机 SNR 混入背景并限制背景能量，提升噪声鲁棒性。例：`mix_with_background(fg, bg)`.
- `apply_simple_reverb(y: np.ndarray, decay: float = 0.3, delay_ms: int = 50) -> np.ndarray`
  - 简单多 tap 回声，模拟房间混响。
- `apply_simple_filter(y: np.ndarray, cutoff: float = 4000.0, sr: int = SR, kind: str = 'lowpass') -> np.ndarray`
  - 4 阶巴特沃斯低/高通，调节频带。

## augment_pipeline.py
- `PIPELINE_REGISTRY: Dict[str, Sequence[str]]`
  - 内置 6 种增强组合（shift_gain、stretch_reverb、shift_mix、filter_gain、gain_mix、stretch_filter）。
- `AugmentedWindow(audio: np.ndarray, description: str)`
  - 增强结果容器，保存波形和文字描述。
- `apply_pipeline(y: np.ndarray, pipeline: Sequence[str>, background: np.ndarray | None = None) -> np.ndarray`
  - 按序执行原语，遇到 mix 必须给背景。例：`apply_pipeline(win, ["time_shift","mix"], bg)`.
- `run_pipeline(y: np.ndarray, pipeline_name: str, background: np.ndarray | None = None) -> AugmentedWindow`
  - 按名称调用注册组合，返回带描述的结果。例：`run_pipeline(win, "shift_mix", background=bg)`.
- `choose_pipeline(name: str | None = None) -> Sequence[str>`
  - 随机或按名选择 pipeline，便于随机增强。

## cache_utils.py
- `CacheEntry(path: str, labels: List[str], label_ids: List[int], label: str, fold_id: int, source_filename: str, clip_id: str, window_id: str, pipeline_name: str, augment_desc: str, source_type: str)`
  - 缓存索引的统一行结构，包含路径、标签 ID、折号、来源等。
- `_resolve_clip_info(row: pd.Series) -> (clip_id: str, source_name: str)`
  - 优先从 `filepath` 推导 clip_id/文件名，缺失时回退 `filename`，避免新 meta 无 filename 时崩溃。
- `_save_window(window: np.ndarray, label: str, clip_id: str, fold_id: int, suffix: str, cache_dir: Path) -> Path`
  - 计算 mel 并写 `.npy`，返回保存路径（内部使用）。
- `sample_background_chunk(dataset_df: pd.DataFrame, length: int, rng=None) -> np.ndarray`
  - 随机挑一段背景并裁剪到指定长度，用于 mix。
- `_encode_labels(raw_label: str | Sequence[str] | None) -> (List[str], List[int], str)`
  - 标签去重/过滤背景，返回标签列表、ID 列表和主标签。
- `_cache_glass_row(...) -> (List[CacheEntry], int)`
  - 对正类生成 base+增强窗口并写盘，返回索引条目和 base 数量。
- `_cache_background_row(...) -> List[CacheEntry]`
  - 背景滑窗写盘，返回索引条目。
- `build_cache_index(dataset_df: pd.DataFrame, pipeline_plan=None, cache_dir=CACHE_DIR, align_labels=None, extra_shifts=None, energy_threshold=0.2, peak_ratio_threshold=0.8, front_peak_ratio=0.5, seed=SEED) -> pd.DataFrame`
  - 主入口：遍历数据集生成 mel 缓存与索引 DataFrame。例：`index_df = build_cache_index(df, cache_dir=Path("cache/mel64_multi"))`，供训练/推理使用。

## data_utils.py
- `build_dataset(meta_df: pd.DataFrame, positive_map=None, background_label=BACKGROUND_LABEL, background_multiplier=BACKGROUND_MULTIPLIER, seed=42) -> pd.DataFrame`
  - 从 meta 采样正例/背景，写入 `target_label`，背景量按倍率控制。
- `audio_path(row: pd.Series) -> Path`
  - 解析音频路径（优先 filepath，其次 raw_filepath，最后 AUDIO_DIR/filename；都缺失时抛 KeyError）。
- `load_audio(row: pd.Series, sr: int = SR) -> (np.ndarray, int)`
  - 加载音频并重采样到目标采样率，单声道输出。
- `trim_silence(y: np.ndarray, sr: int, top_db: float = 20.0, min_keep_seconds: float = 0.0) -> np.ndarray`
  - 按能量阈值裁静音段，可设置最短保留片段。
- `log_mel_spectrogram(y: np.ndarray, sr: int, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH, n_mels: int = N_MELS) -> np.ndarray`
  - 计算 log-mel 特征。
- `_iter_windows(y: np.ndarray, sr: int, window_seconds=WINDOW_SECONDS, hop_seconds=WINDOW_HOP) -> Iterable[(np.ndarray, int)]`
  - 滑窗生成器，返回窗口及起始样本（内部）。
- `sliding_windows(y: np.ndarray, sr: int, window_seconds=WINDOW_SECONDS, hop_seconds=WINDOW_HOP) -> Iterable[np.ndarray]`
  - 公开滑窗，只给窗口。
- `center_peak_window(y: np.ndarray, sr: int, window_seconds=WINDOW_SECONDS, hop: int = HOP_LENGTH, shift_seconds: float = 0.0) -> np.ndarray`
  - 围绕 RMS 峰值取窗，可再加偏移。
- `_energy_mask(y: np.ndarray, sr: int, window_seconds: float, hop_seconds: float, threshold_ratio: float) -> np.ndarray`
  - 根据相对能量生成布尔掩码（内部）。
- `generate_aligned_windows(row: pd.Series, align_labels: List[str], extra_shifts=None, energy_threshold=0.2, peak_ratio_threshold=0.7, front_peak_ratio=0.5, trim_silence_before=False, trim_top_db=20.0, trim_min_keep_seconds=0.0, debug=False, label_params=None, debug_sink: list | None = None) -> List[np.ndarray]`
  - 正类：对裁静音后的波形滑窗，按峰值占比和峰值位置筛选；背景：按能量掩码筛窗；两者都在 `debug_sink` 记录“原始时间轴上的起止、状态、原因、峰值占比/位置”，并在无窗时用中心峰值兜底。默认参数从 `config.WINDOW_PARAMS` 读取（glass/gunshot/background 三套），`label_params` 可进一步覆写，`extra_shifts` 支持兜底偏移；`debug` 仅为兼容保留，不再打印。例：`logs=[]; wins = generate_aligned_windows(row, ["glass","gunshot"], label_params={"gunshot":{"peak_ratio_threshold":0.6}}, debug_sink=logs)`.
- `generate_aligned_windows_legacy(...)`
  - 旧版逻辑，保留向后兼容（可参考源码，推荐使用新版）。

## datasets.py
- `balance_folds(index_df: pd.DataFrame, target_ratio: float = 0.4, random_state: int = 42) -> pd.DataFrame`
  - 按折平衡正样/背景比例（多标签看 label_ids），返回新索引。
- `MelEntry(path: Path, label_ids: List[int>)`
  - Dataset 用的条目结构。
- `MelDataset(index_df: pd.DataFrame, max_items: int | None = None)`
  - `__getitem__` 返回 `(mel[1,64,T], multi_hot targets)`；可截取前 max_items。例：`ds = MelDataset(df)`.
- `load_index_df(index_path: str | Path) -> pd.DataFrame`
  - 读取 CSV 或 Parquet 索引。
- `subset_by_folds(index_df: pd.DataFrame, folds: Sequence[int]) -> pd.DataFrame`
  - 选取指定折。
- `pad_mel_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]], max_frames: int | None = None) -> (torch.Tensor, torch.Tensor)`
  - 批量 pad/crop 到统一时间长度。例：`collate_fn=lambda b: pad_mel_batch(b, 200)`.
- `build_dataloaders(index_df, train_folds, val_folds, batch_size=64, num_workers=0, smoke_limit=None, random_state=42, collate_max_frames=None) -> (DataLoader, DataLoader)`
  - 生成训练/验证 DataLoader，支持 smoke 抽样与截帧。例：`train_loader, val_loader = build_dataloaders(df, [1,2,3,4], [5])`.

## event_detection.py
- `ClipSpec(path: Path, label: str, gain_db: float = 0.0)`
  - 描述待混入的剪辑及增益。
- `GroundTruthEvent(label: str, start: float, end: float, source: str, snr_db: float | None = None)`
  - 真值事件：标签、起止时间、来源、可选 SNR。
- `_apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray`
  - 按 dB 放大/衰减（内部）。
- `_apply_fade(audio: np.ndarray, fade_samples: int) -> np.ndarray`
  - 做淡入淡出（内部）。
- `_crossfade_blend(timeline: List[np.ndarray], waveform: np.ndarray, crossfade: int) -> (int, np.ndarray)`
  - 与上一段交叉淡化，返回重叠长度与剩余波形（内部）。
- `_rms(x: np.ndarray) -> float`
  - RMS 计算（内部）。
- `_scale_to_snr(fg: np.ndarray, bg: np.ndarray, snr_db: float) -> np.ndarray`
  - 将前景缩放到目标 SNR（内部）。
- `detect_events_in_clip(audio: np.ndarray, sr: int, top_db: float = 35.0, min_event_dur: float = 0.08, max_event_dur: float | None = None) -> List[Tuple[float, float]]`
  - 用非静音分段找事件，太长的分段会被切块。
- `compose_timeline(clips: Sequence[ClipSpec], sr: int = SR, crossfade_ms: float = 15.0, normalize: bool = True) -> (np.ndarray, List[GroundTruthEvent])`
  - 串接多个剪辑（可交叉淡化），返回合成音频和真值列表。常用于背景床或顺序拼接。
- `build_background_bed(clips: Sequence[ClipSpec], sr: int = SR, target_duration: float | None = None, crossfade_ms: float = 15.0, normalize: bool = True) -> np.ndarray`
  - 循环拼背景到目标时长，输出归一化床噪。
- `mix_glass_on_bed(background_bed: np.ndarray, glass_clips: Sequence[ClipSpec], sr: int = SR, start_offset_range=(0.5,5.0), gap_range=(-0.5,8.0), crossfade_ms=15.0, snr_range_db=(3.0,9.0), split_top_db=35.0, min_event_dur=0.08, max_event_dur=None, segment_on_original=True, seed=42, rng=None) -> (np.ndarray, List[GroundTruthEvent])`
  - 随机顺序/间隔将前景叠加到背景，控制 SNR/淡化/分段，返回混音和真值。用于 case study 合成。
- `sliding_log_mel_windows(y: np.ndarray, sr: int = SR, window_seconds=WINDOW_SECONDS, hop_seconds=WINDOW_HOP, min_coverage=0.7, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS) -> (torch.Tensor, List[Tuple[float, float]])`
  - 将长音频切成 mel 批次和对应时间跨度。例：`batch, spans = sliding_log_mel_windows(audio, sr)`。
- `merge_events(window_spans: Sequence[Tuple[float, float]], glass_probs: Sequence[float], threshold: float = 0.5, merge_gap: float = 0.25) -> List[dict]`
  - 把连续高概率窗口合并成事件块。
- `match_events(pred_events: Sequence[dict], gt_events: Sequence[GroundTruthEvent], tolerance: float = 0.5) -> dict`
  - 基于时间容差计算 TP/FP/FN 和 P/R/F1。
- `predict_glass_probs(batch: torch.Tensor, spans, model=None, session=None, device="cpu") -> (List[float], InferenceResult)`
  - Torch 或 ONNX 推理，取玻璃列概率。例：`probs, res = predict_glass_probs(batch, spans, model=my_model)`.
- `predict_label_probs(batch: torch.Tensor, spans, labels, model=None, session=None, device="cpu") -> (dict[str, List[float]], InferenceResult)`
  - 多标签概率按标签返回。例：`probs, _ = predict_label_probs(batch, spans, TARGET_LABELS, session=ort_sess)`.
- `match_events_with_pairs(pred_events, gt_events, tolerance=0.5) -> (pairs, unmatched_gt, unmatched_pred)`
  - 返回匹配对（含延迟）和未匹配索引，兼容 dict 形式的预测输入。
- `bucket_recall_by_snr(gt_events, matched_gt_indices) -> dict`
  - 按 SNR 桶统计召回。
- `bucket_delay(pairs) -> dict`
  - 将匹配延迟分桶计数。
- `smooth_probabilities(probs, kernel_size=1) -> List[float]`
  - 对概率做滑动平均，kernel=1 时不变。

## export.py
- `export_to_onnx(model: torch.nn.Module, example_input: torch.Tensor, onnx_path: str | Path, opset: int = 13) -> Path`
  - 导出模型为 ONNX，设置动态 batch/time，返回保存路径。例：`export_to_onnx(model, torch.randn(1,1,64,128), "cache/model.onnx")`。

## folds.py
- `assign_folds(dataset_df: pd.DataFrame, fold_column: str = "fold") -> pd.Series`
  - 读取指定列作为折号并转 int。
- `fold_summary(dataset_df: pd.DataFrame, fold_column: str = "fold", label_column: str = "target_label") -> pd.DataFrame`
  - 折×标签计数表，便于检查分布。

## inference.py
- `InferenceResult(logits: torch.Tensor, probs: torch.Tensor, preds: torch.Tensor)`
  - 推理输出容器，包含 logits、sigmoid 概率和阈值化预测。
- `load_torch_checkpoint(checkpoint_path, device="cpu", num_classes=None) -> (TinyGlassNet, Dict)`
  - 载入 checkpoint，构建模型并返回载荷。例：`model, payload = load_torch_checkpoint("tinyglassnet_best.pt")`。
- `load_mel_batch(index_df, max_items=None) -> torch.Tensor`
  - 从索引加载 mel `.npy` 堆成 batch。例：`batch = load_mel_batch(df.head(8))`。
- `run_torch_inference(model: TinyGlassNet, batch: torch.Tensor, device="cpu", threshold=0.5) -> InferenceResult`
  - Torch 前向 + sigmoid + 阈值化。
- `create_onnx_session(onnx_path, providers=None) -> ort.InferenceSession`
  - 创建 ONNX Runtime 会话，默认 CPUExecutionProvider。
- `run_onnx_inference(session: ort.InferenceSession, batch: torch.Tensor, threshold=0.5) -> InferenceResult`
  - ONNX 前向 + 阈值化，常用于导出校验或部署。

## metrics.py
- `confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 2, normalize: bool = False) -> np.ndarray`
  - 支持多热/单 logit 的混淆矩阵计算，可选择归一化。
- `plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], normalize: bool = False, ax: plt.Axes | None = None) -> plt.Axes`
  - 画混淆矩阵热力图并标注数值。例：`plot_confusion_matrix(cm, ["bg","glass"])`。

## models.py
- `TinyGlassNet(in_channels: int = 1, base_channels: int = 16, num_classes: int = NUM_CLASSES)`
  - 3×Conv+ReLU+Pool + GAP + FC 的轻量 CNN，`forward(x) -> logits`，易导出 ONNX，适合 Edge。
- `count_parameters(model: nn.Module) -> int`
  - 计算可训练参数总数。
- `TrainingArtifacts(history: list[dict], best_state_dict: dict)`
  - 训练产物容器，包含历史记录与最佳状态。

## training.py
- `EpochResult(loss: float, accuracy: float, precision: float, recall: float, f1: float)`
  - 单 epoch 指标容器。
- `_compute_binary_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float>`
  - 计算宏 P/R/F1 和 subset acc（内部）。
- `_run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None, grad_clip_norm: float | None = None) -> (EpochResult, torch.Tensor, torch.Tensor)`
  - 单轮训练/验证核心，自动处理 CrossEntropy+多热回退 BCE，返回指标、预测、真值。
- `train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device, scheduler=None, early_stopping_patience=5, grad_clip_norm=None) -> TrainingArtifacts`
  - 训练主循环，含早停与 LR 调度。例：`artifacts = train_model(model, tr_loader, val_loader, 30, criterion, optim, device)`。
- `evaluate_model(model, loader, criterion, device) -> Dict[str, float>`
  - 验证集快速评估。
- `run_kfold_training(k, fold_ids, index_df, build_loaders_fn, model_builder, criterion_builder, optimizer_builder, scheduler_builder=None, device=None, output_dir=None, epochs=20, early_stopping=5, grad_clip_norm=None, **loader_kwargs) -> List[Dict]`
  - K 折 orchestrator：构建 loader/模型、训练并可保存各折 checkpoint。

## viz.py
- `plot_wave_and_mel(row: pd.Series | None = None, y: np.ndarray | None = None, sr: int | None = None, title: str | None = None) -> None`
  - 绘制波形与 log-mel（可传 meta 行或直接 (y, sr)），可自定义标题。适合质检和演示。

## case_study_cli.py
- `_load_config(path: Path | None) -> dict[str, Any]`
  - 读取 YAML/JSON 配置（内部用）。
- `run_case_study(cfg_path: Path | None, output_dir: Path | None, seed: int | None)`
  - 运行多标签混音→滑窗推理→指标输出，生成报表/音频。例：`run_case_study(Path("config.yaml"), Path("out"), seed=123)`.
- `main()`
  - CLI 入口，支持 `python -m src.case_study_cli`。

## export.py / tools 提示
- 已在 export.py 节列出导出接口；工具脚本（`tools/convert_to_wav.py`, `tools/resample_audio.py`）为 CLI，签名见各文件 argparse 定义，调用示例：
  - `python tools/convert_to_wav.py INPUT_DIR --out-dir OUT --sr 22050 --mono`
  - `python tools/resample_audio.py --out-root cache/data_resampled --out-meta-dir data/meta_resampled --sr 22050`
