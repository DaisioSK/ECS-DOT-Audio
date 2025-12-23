"""Training utilities for CNN baseline."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import TrainingArtifacts


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def _compute_binary_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute subset accuracy and macro precision/recall/F1 for multi-label."""
    preds = preds.detach().cpu().float()
    targets = targets.detach().cpu().float()
    if preds.numel() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    eps = 1e-8
    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)
    precision_c = tp / (tp + fp + eps)
    recall_c = tp / (tp + fn + eps)
    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + eps)
    macro_precision = precision_c.mean().item()
    macro_recall = recall_c.mean().item()
    macro_f1 = f1_c.mean().item()
    subset_acc = (preds == targets).all(dim=1).float().mean().item()
    return {
        "accuracy": subset_acc,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }


def _run_epoch(model: nn.Module,
               loader: DataLoader,
               criterion: nn.Module,
               device: torch.device,
               optimizer: torch.optim.Optimizer | None = None,
               grad_clip_norm: float | None = None) -> Tuple[EpochResult, torch.Tensor, torch.Tensor]:
    """Run one train/val epoch."""
    is_train = optimizer is not None
    model.train(mode=is_train)
    total_loss = 0.0
    preds: List[torch.Tensor] = []
    trues: List[torch.Tensor] = []
    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            # Use provided criterion; if CrossEntropy is given but targets are multi-hot, fall back to BCE.
            if isinstance(criterion, nn.CrossEntropyLoss) and targets.dim() > 1:
                if logits.shape != targets.shape:
                    # Align logits to target shape when num_classes differs (e.g., default model classes vs config)
                    if logits.shape[1] > targets.shape[1]:
                        logits = logits[:, :targets.shape[1]]
                    elif logits.shape[1] < targets.shape[1]:
                        targets = targets[:, :logits.shape[1]]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            else:
                loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(logits.detach())
            pred_labels = (probs >= 0.5).float()
            preds.append(pred_labels)
            trues.append(targets.detach().float())
    preds_cat = torch.cat(preds) if preds else torch.empty(0)
    trues_cat = torch.cat(trues) if trues else torch.empty(0)
    metrics = _compute_binary_metrics(preds_cat, trues_cat)
    avg_loss = total_loss / max(len(loader.dataset), 1)
    result = EpochResult(loss=avg_loss, **metrics)
    return result, preds_cat, trues_cat


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
                early_stopping_patience: int | None = 5,
                grad_clip_norm: float | None = None,
                best_key: str = "f1",
                maximize: bool | None = None,
                top_k: int = 1,
                select_last_ratio: float = 0.1,
                train_loss_tolerance: float = 0.2) -> TrainingArtifacts:
    """Train model with early stopping and metric logging."""
    history: List[Dict] = []
    best_state = None
    maximize = True if maximize is None else maximize
    if best_key.lower() == "loss":
        maximize = False
    best_score = -float("inf") if maximize else float("inf")
    early_stop_enabled = early_stopping_patience is not None and early_stopping_patience > 0
    patience = early_stopping_patience if early_stop_enabled else float("inf")
    epochs_without_improve = 0
    candidate_states: List[Dict] = []

    def _compare(new_score: float, ref_score: float) -> bool:
        return new_score > ref_score if maximize else new_score < ref_score

    last_window = max(1, int(math.ceil(epochs * max(select_last_ratio, 0.0))))

    for epoch in range(1, epochs + 1):
        train_result, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )
        val_result, val_preds, val_targets = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        val_metrics = {
            "loss": val_result.loss,
            "accuracy": val_result.accuracy,
            "precision": val_result.precision,
            "recall": val_result.recall,
            "f1": val_result.f1,
        }
        val_score = val_metrics.get(best_key, val_result.f1)
        history_entry = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_acc": train_result.accuracy,
            "train_f1": train_result.f1,
            "val_loss": val_result.loss,
            "val_acc": val_result.accuracy,
            "val_precision": val_result.precision,
            "val_recall": val_result.recall,
            "val_f1": val_result.f1,
            "lr": current_lr,
        }
        history.append(history_entry)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_result.loss)
        elif scheduler is not None:
            scheduler.step()
        history_entry["lr"] = optimizer.param_groups[0]["lr"]

        improved = _compare(val_score, best_score)
        if improved:
            best_score = val_score
            best_state = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "val_metrics": val_metrics,
                "score": val_score,
                "score_key": best_key,
                "epoch": epoch,
                "predictions": val_preds.cpu(),
                "targets": val_targets.cpu(),
                "train_loss": train_result.loss,
            }
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        candidate_states.append({
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "val_metrics": val_metrics,
            "score": val_score,
            "score_key": best_key,
            "epoch": epoch,
            "train_loss": train_result.loss,
            "predictions": val_preds.cpu(),
            "targets": val_targets.cpu(),
        })
        if len(candidate_states) > last_window:
            candidate_states.pop(0)
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_result.loss:.4f} train_acc={train_result.accuracy:.3f} "
            f"train_f1={train_result.f1:.3f} val_loss={val_result.loss:.4f} "
            f"val_acc={val_result.accuracy:.3f} val_f1={val_result.f1:.3f} "
            f"lr={current_lr:.6f}"
        )
        if early_stop_enabled and epochs_without_improve >= patience:
            print("Early stopping triggered.")
            break
    if not candidate_states:
        candidate_states = [best_state] if best_state is not None else []

    filtered = candidate_states
    if candidate_states:
        losses = [c.get("train_loss", float("inf")) for c in candidate_states]
        median_loss = float(torch.tensor(losses).median().item()) if losses else float("inf")
        max_loss = median_loss * (1.0 + max(train_loss_tolerance, 0.0))
        filtered = [c for c in candidate_states if c.get("train_loss", float("inf")) <= max_loss]
        if not filtered:
            filtered = candidate_states

    default_score = -float("inf") if maximize else float("inf")
    filtered.sort(key=lambda s: s.get("score", default_score), reverse=maximize)
    top_states = filtered[:max(1, top_k)]
    best_state = top_states[0] if top_states else best_state
    if best_state is None:
        best_state = {
            "model": model.state_dict(),
            "val_metrics": {},
            "score": None,
            "score_key": best_key,
            "epoch": epoch,
        }
    return TrainingArtifacts(history=history, best_state_dict=best_state, top_states=top_states)


def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> Dict[str, float]:
    """Evaluate trained model on a loader."""
    result, _, _ = _run_epoch(model, loader, criterion, device, optimizer=None)
    return {
        "loss": result.loss,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
    }


def run_kfold_training(k: int,
                       fold_ids: Sequence[int],
                       index_df,
                       build_loaders_fn,
                       model_builder,
                       criterion_builder,
                       optimizer_builder,
                       scheduler_builder=None,
                       device: torch.device | None = None,
                       output_dir: str | Path | None = None,
                       epochs: int = 20,
                       early_stopping: int = 5,
                       grad_clip_norm: float | None = None,
                       best_key: str = "f1",
                       maximize: bool | None = None,
                       top_k_checkpoints: int = 1,
                       **loader_kwargs) -> List[Dict]:
    """Run K-fold training, returning metrics per fold."""
    device = device or torch.device("cpu")
    fold_records: List[Dict] = []
    for fold in fold_ids:
        val_folds = (fold,)
        train_folds = tuple(sorted(set(fold_ids) - set(val_folds)))
        print(f"=== Fold {fold}: train={train_folds} val={val_folds} ===")
        train_loader, val_loader = build_loaders_fn(
            index_df,
            train_folds=train_folds,
            val_folds=val_folds,
            **loader_kwargs,
        )
        model = model_builder().to(device)
        criterion = criterion_builder()
        optimizer = optimizer_builder(model.parameters())
        scheduler = scheduler_builder(optimizer) if scheduler_builder else None
        artifacts = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            early_stopping_patience=early_stopping,
            grad_clip_norm=grad_clip_norm,
            best_key=best_key,
            maximize=maximize,
            top_k=top_k_checkpoints,
        )
        best_state = artifacts.best_state_dict
        metrics = best_state.get("val_metrics", {}).copy()
        top_states = artifacts.top_states
        record = {
            "fold": fold,
            "train_folds": train_folds,
            "metrics": metrics,
            "history": artifacts.history,
            "best_state": best_state,
            "top_states": top_states,
            "checkpoint_path": None,
        }
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            num_classes = getattr(model, "classifier", None)
            num_classes = getattr(num_classes, "out_features", None) if num_classes is not None else None
            # Save top-k checkpoints, keep the best path for backward compatibility.
            saved_paths = []
            for rank, state in enumerate(top_states, start=1):
                ckpt_path = output_dir / f"tinyglassnet_fold{fold}_top{rank}.pt"
                torch.save(
                    {
                        "model_state_dict": state["model"],
                        "val_metrics": state.get("val_metrics", {}),
                        "fold": fold,
                        "train_folds": train_folds,
                        "num_classes": num_classes,
                        "epoch": state.get("epoch"),
                        "score": state.get("score"),
                        "score_key": state.get("score_key", best_key),
                    },
                    ckpt_path,
                )
                saved_paths.append(str(ckpt_path))
                if rank == 1:
                    record["checkpoint_path"] = str(ckpt_path)
            record["top_checkpoint_paths"] = saved_paths
            print(f"Saved fold top-{len(saved_paths)} checkpoints to {output_dir}")
        fold_records.append(record)
    return fold_records


__all__ = ["train_model", "evaluate_model", "run_kfold_training"]
