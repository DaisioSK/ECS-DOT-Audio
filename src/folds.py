"""Utilities for assigning cross-validation folds."""
from __future__ import annotations

import pandas as pd


def assign_folds(dataset_df: pd.DataFrame, fold_column: str = "fold") -> pd.Series:
    """Return fold id per row, defaulting to metadata fold column."""
    if fold_column not in dataset_df.columns:
        raise KeyError(f"Column '{fold_column}' missing from dataset_df")
    return dataset_df[fold_column].astype(int)


def fold_summary(dataset_df: pd.DataFrame,
                 fold_column: str = "fold",
                 label_column: str = "target_label") -> pd.DataFrame:
    """Aggregate counts per fold/label for inspection."""
    if fold_column not in dataset_df or label_column not in dataset_df:
        raise KeyError("Required columns missing for fold summary")
    return (
        dataset_df.groupby([fold_column, label_column])
        .size()
        .reset_index(name="count")
    )


def summarize_windows(win_df: pd.DataFrame, label_col: str = "target_label") -> dict:
    """Summaries for window-level df: per-fold totals and clip-level counts."""
    if win_df.empty:
        return {"clips": pd.DataFrame(), "per_fold": {}, "per_fold_label": pd.DataFrame()}
    clips = (
        win_df.groupby(["clip_id", "fold_id", label_col])
        .size()
        .reset_index(name="n_windows")
    )
    per_fold = win_df.groupby("fold_id").size().to_dict()
    per_fold_label = (
        win_df.pivot_table(index="fold_id", columns=label_col, values="window_id", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    return {"clips": clips, "per_fold": per_fold, "per_fold_label": per_fold_label}


def rebalance_folds_by_windows(
    win_df: pd.DataFrame,
    target_per_fold: int | None = None,
    label_col: str = "target_label",
    preserve_label_presence: bool = True,
    sub_label_col: str | None = None,
    preserve_sub_label_presence: bool = True,
    tolerance_ratio: float = 0.1,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Greedy rebalance: move whole clips between folds to reduce window-count variance.

    Args:
        win_df: window-level df with columns ['clip_id','fold_id',label_col,'window_id']
        target_per_fold: desired windows per fold (int). Default: total/k.
        label_col: column name for label.
        preserve_label_presence: avoid moving a clip if it would remove the last clip of that label in a fold.
        tolerance_ratio: allowed deviation (e.g., 0.1 -> +/-10%).
    Returns:
        (rebalanced_df, moves) where moves is a list of {clip_id, from, to, n_windows, label}
    """
    if win_df.empty:
        return win_df.copy(), []

    k = win_df["fold_id"].nunique()
    total = len(win_df)
    target = target_per_fold or int(round(total / k))
    tol = max(1, int(round(target * tolerance_ratio)))

    group_cols = ["clip_id", "fold_id", label_col] + ([sub_label_col] if sub_label_col else [])
    clips = win_df.groupby(group_cols).size().reset_index(name="n_windows")

    label_presence = (
        clips.groupby(["fold_id", label_col])
        .size()
        .reset_index(name="n_clips")
        .pivot_table(index="fold_id", columns=label_col, values="n_clips", fill_value=0)
    )
    sub_presence = None
    if sub_label_col:
        sub_presence = (
            clips.groupby(["fold_id", sub_label_col])
            .size()
            .reset_index(name="n_clips")
            .pivot_table(index="fold_id", columns=sub_label_col, values="n_clips", fill_value=0)
        )

    fold_counts = win_df.groupby("fold_id").size().to_dict()
    moves: list[dict] = []
    moved_once: set[str] = set()

    def over_under():
        over = sorted(
            [f for f, c in fold_counts.items() if c > target + tol],
            key=lambda f: fold_counts[f],
            reverse=True,
        )
        under = sorted(
            [f for f, c in fold_counts.items() if c < target - tol],
            key=lambda f: fold_counts[f],
        )
        return over, under

    over, under = over_under()
    max_iter = max(1, len(clips) * 2)
    iters = 0
    while over and under and iters < max_iter:
        iters += 1
        src = over[0]
        dst = under[0]
        cand = clips[clips["fold_id"] == src]
        # avoid moving the same clip repeatedly
        cand = cand[~cand["clip_id"].isin(moved_once)]
        if cand.empty:
            over.pop(0)
            over, under = over_under()
            continue
        # pick candidate that best reduces imbalance (min max deviation after move)
        best_row = None
        best_score = float("inf")
        moved = False
        for _, row in cand.iterrows():
            clip_id, fold_id, lbl, n_win = row["clip_id"], row["fold_id"], row[label_col], row["n_windows"]
            sub_lbl = row[sub_label_col] if sub_label_col and sub_label_col in row else None
            if preserve_label_presence and label_presence.loc[src, lbl] <= 1:
                continue
            if preserve_sub_label_presence and sub_label_col and sub_presence is not None:
                if sub_presence.loc[src].get(sub_lbl, 0) <= 1:
                    continue
            new_src = fold_counts[src] - n_win
            new_dst = fold_counts[dst] + n_win
            score = max(abs(new_src - target), abs(new_dst - target))
            if score < best_score:
                best_score = score
                best_row = row
        if best_row is not None:
            clip_id, lbl, n_win = best_row["clip_id"], best_row[label_col], best_row["n_windows"]
            clips.loc[(clips["clip_id"] == clip_id), "fold_id"] = dst
            fold_counts[src] -= n_win
            fold_counts[dst] += n_win
            label_presence.loc[src, lbl] -= 1
            label_presence.loc[dst, lbl] = label_presence.loc[dst, lbl] + 1
            moves.append({"clip_id": clip_id, "from": src, "to": dst, "n_windows": int(n_win), "label": lbl})
            moved_once.add(clip_id)
            moved = True
        if not moved:
            over.pop(0)
        over, under = over_under()

    moved_clip_ids = {m["clip_id"]: m["to"] for m in moves}
    out = win_df.copy()
    if moved_clip_ids:
        out["fold_id"] = out["clip_id"].map(moved_clip_ids).fillna(out["fold_id"]).astype(int)
    return out, moves
