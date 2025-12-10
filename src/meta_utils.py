"""Helpers to load/merge meta CSVs and build balanced folds for multi-label tasks."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import BACKGROUND_LABEL, LABEL_TO_ID, TARGET_LABELS


def load_meta_files(meta_files: Sequence[str | Path]) -> pd.DataFrame:
    """Load and concatenate meta CSV files sharing the unified schema."""
    parts = []
    for path in meta_files:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    return df


def assign_folds_if_missing(df: pd.DataFrame, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Fill missing fold_id via stable hash on filepath."""
    out = df.copy()
    miss_mask = out["fold_id"].isna() | (out["fold_id"] == "") | (out["fold_id"] == " ")

    def _hash_to_fold(val: str) -> int:
        h = hashlib.md5((val + str(seed)).encode("utf-8")).hexdigest()
        return (int(h, 16) % k) + 1

    out.loc[miss_mask, "fold_id"] = out.loc[miss_mask, "filepath"].apply(_hash_to_fold)
    out["fold_id"] = out["fold_id"].astype(int)
    return out


def map_canonical_labels(df: pd.DataFrame,
                         label_map: Dict[str, str],
                         target_labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Map raw labels to canonical labels; non-mapped keep original."""
    target_labels = target_labels or TARGET_LABELS
    out = df.copy()
    out["canonical_label"] = out["label"].map(label_map).fillna(out["label"])
    out["is_target"] = out["canonical_label"].isin(target_labels)
    return out


def _compute_bg_weights(bg_df: pd.DataFrame, label_col: str = "canonical_label") -> np.ndarray:
    counts = bg_df[label_col].value_counts()
    weights = bg_df[label_col].map(lambda c: 1.0 / counts.get(c, 1))
    weights = weights / weights.sum()
    return weights.values


def balance_folds_multi(df: pd.DataFrame,
                        target_labels: Sequence[str],
                        ratios: Dict[str, float],
                        seed: int = 42) -> pd.DataFrame:
    """Balance each fold to approximate target label ratios."""
    rng = np.random.default_rng(seed)
    target_ratio_total = sum(ratios.get(lbl, 0.0) for lbl in target_labels)
    bg_ratio = ratios.get("background", 0.0)

    balanced_parts: List[pd.DataFrame] = []
    for fold_id, fold_df in df.groupby("fold_id"):
        pos_df = fold_df[fold_df["canonical_label"].isin(target_labels)]
        bg_pool = fold_df[~fold_df["canonical_label"].isin(target_labels)]
        pos_total = len(pos_df)
        if pos_total == 0:
            # No positives, take all background (or skip)
            balanced_parts.append(bg_pool.copy())
            continue
        desired_bg = int(round(pos_total * (bg_ratio / max(target_ratio_total, 1e-8))))
        desired_bg = min(desired_bg, len(bg_pool))
        if desired_bg > 0 and len(bg_pool) > 0:
            weights = _compute_bg_weights(bg_pool)
            sample_idx = rng.choice(len(bg_pool), size=desired_bg, replace=False, p=weights)
            bg_sample = bg_pool.iloc[sorted(sample_idx)].copy()
        else:
            bg_sample = pd.DataFrame(columns=bg_pool.columns)
        balanced_parts.append(pd.concat([pos_df, bg_sample], ignore_index=True))

    if not balanced_parts:
        return df.copy()
    out = pd.concat(balanced_parts, ignore_index=True)
    return out


def attach_label_ids(df: pd.DataFrame, target_labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Add labels/label_ids columns for downstream multi-label datasets."""
    target_labels = target_labels or TARGET_LABELS
    out = df.copy()

    def _labels_for_row(row):
        lab = row["canonical_label"]
        return [lab] if lab in target_labels else []

    out["labels"] = out.apply(_labels_for_row, axis=1)
    out["label_ids"] = out["labels"].apply(lambda labs: [LABEL_TO_ID[l] for l in labs if l in LABEL_TO_ID])
    out["label"] = out["canonical_label"]
    return out


def filter_by_audio_attrs(df: pd.DataFrame,
                          allowed_sr: Sequence[int] | None = None,
                          allowed_channels: Sequence[int] | None = None,
                          min_duration: float | None = None,
                          max_duration: float | None = None) -> pd.DataFrame:
    """Filter meta rows by sr/channels/duration constraints."""
    out = df.copy()
    if allowed_sr:
        out = out[out["sr"].isin(allowed_sr)]
    if allowed_channels:
        out = out[out["channels"].isin(allowed_channels)]
    if min_duration is not None:
        out = out[out["duration_sec"] >= min_duration]
    if max_duration is not None:
        out = out[out["duration_sec"] <= max_duration]
    return out


__all__ = [
    "load_meta_files",
    "assign_folds_if_missing",
    "map_canonical_labels",
    "balance_folds_multi",
    "attach_label_ids",
    "filter_by_audio_attrs",
    "stratified_folds",
]


def deduplicate_meta(df: pd.DataFrame, subset: Sequence[str] | None = None) -> pd.DataFrame:
    """Drop duplicate rows based on md5/filepath."""
    subset = subset or ["md5", "filepath"]
    existing = [col for col in subset if col in df.columns]
    if not existing:
        return df.copy()
    return df.drop_duplicates(subset=existing, keep="first").reset_index(drop=True)


def _extract_weapon_id(row: pd.Series) -> str:
    """Parse weapon_id from extra_meta or parent folder fallback."""
    extra = row.get("extra_meta", "")
    weapon = None
    if isinstance(extra, str) and "weapon_id=" in extra:
        try:
            weapon = extra.split("weapon_id=")[-1].split(",")[0]
        except Exception:
            weapon = None
    if not weapon:
        # fallback: parent folder name
        weapon = Path(str(row.get("filepath", ""))).parent.name
    return weapon


def sample_gunshot_even(df: pd.DataFrame,
                        target_label: str = "gunshot",
                        total: int = 60,
                        seed: int = 42) -> pd.DataFrame:
    """Evenly sample gunshot clips across weapon_id to reach target total."""
    gun_df = df[df["canonical_label"] == target_label].copy()
    if gun_df.empty:
        return gun_df
    gun_df["weapon_id"] = gun_df.apply(_extract_weapon_id, axis=1)
    groups = list(gun_df.groupby("weapon_id"))
    rng = np.random.default_rng(seed)
    per_group = max(1, total // len(groups))
    sampled_parts: List[pd.DataFrame] = []
    remaining = total
    for weapon, g in groups:
        take = min(len(g), per_group)
        sampled = g.sample(n=take, random_state=seed) if take < len(g) else g
        sampled_parts.append(sampled)
        remaining -= len(sampled)
    # If total not reached due to short groups, top up from leftovers
    if remaining > 0:
        leftovers = pd.concat([g for _, g in groups], ignore_index=True)
        already_idx = pd.concat(sampled_parts).index
        leftovers = leftovers[~leftovers.index.isin(already_idx)]
        if len(leftovers) > 0:
            topup = leftovers.sample(n=min(remaining, len(leftovers)), random_state=seed)
            sampled_parts.append(topup)
    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    return sampled_df


def stratified_folds(df: pd.DataFrame,
                     k: int = 5,
                     seed: int = 42,
                     group_key: str = "canonical_label",
                     sub_key: str | None = None,
                     fold_column: str = "fold_id") -> pd.DataFrame:
    """Assign folds with stratification by label and optional subgroup, aiming for even counts."""
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []

    for _, gdf in df.groupby(group_key):
        if sub_key and sub_key in gdf.columns:
            # Interleave sub-groups to keep subcategory balanced
            pools = []
            for _, sdf in gdf.groupby(sub_key):
                idx = np.arange(len(sdf))
                rng.shuffle(idx)
                pools.append(sdf.iloc[idx].copy())
            ordered_rows = []
            # Round-robin pull one sample per subgroup until all empty
            while any(len(p) > 0 for p in pools):
                for p in pools:
                    if len(p) > 0:
                        ordered_rows.append(p.iloc[0])
                        p.drop(p.index[0], inplace=True)
            gdf_ordered = pd.DataFrame(ordered_rows)
        else:
            idx = np.arange(len(gdf))
            rng.shuffle(idx)
            gdf_ordered = gdf.iloc[idx].copy()

        fold_ids = (np.arange(len(gdf_ordered)) % k) + 1
        gdf_ordered[fold_column] = fold_ids
        parts.append(gdf_ordered)

    if not parts:
        return df.copy()
    return pd.concat(parts, ignore_index=True)
