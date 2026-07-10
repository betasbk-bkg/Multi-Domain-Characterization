"""Common metrics for Paper B dataset admission and pilot analysis.

This module contains dataset-agnostic functions only. Dataset-specific parsing
must live in adapters/ and output the standardized format described in
03_standardized_data_schema/STANDARD_DATA_FORMAT.md.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import math
import numpy as np
import pandas as pd


def normalize_distribution(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {str(k): float(v) / float(total) for k, v in counts.items()}


def majority_label(labels: Sequence[str]) -> str:
    if len(labels) == 0:
        raise ValueError("empty label sequence")
    counts = Counter(map(str, labels))
    # deterministic tie-break for reproducibility
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p) | set(q)
    p_arr = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    q_arr = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    p_arr = p_arr / max(p_arr.sum(), eps)
    q_arr = q_arr / max(q_arr.sum(), eps)
    m = 0.5 * (p_arr + q_arr)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2((a[mask] + eps) / (b[mask] + eps))))
    return 0.5 * kl(p_arr, m) + 0.5 * kl(q_arr, m)


def performance_accuracy(labels_by_item: pd.DataFrame, gold: pd.DataFrame, N: int, rng: np.random.Generator) -> float:
    gold_map = dict(zip(gold["item_id"].astype(str), gold["gold_label"].astype(str)))
    hits = []
    for item_id, grp in labels_by_item.groupby("item_id"):
        item_id = str(item_id)
        if item_id not in gold_map or len(grp) < N:
            continue
        sampled = rng.choice(grp["label"].astype(str).to_numpy(), size=N, replace=False)
        hits.append(majority_label(sampled) == gold_map[item_id])
    if not hits:
        return float("nan")
    return float(np.mean(hits))


def performance_distribution_recovery(labels: pd.DataFrame, N: int, rng: np.random.Generator, metric: str = "jsd") -> float:
    scores = []
    for item_id, grp in labels.groupby("item_id"):
        arr = grp["label"].astype(str).to_numpy()
        if len(arr) < 2 * N or len(arr) < 10:
            continue
        perm = rng.permutation(len(arr))
        # split-half: reference and query are disjoint
        ref_idx = perm[: len(arr)//2]
        query_idx = perm[len(arr)//2:]
        if len(query_idx) < N or len(ref_idx) < 2:
            continue
        query_sample = rng.choice(arr[query_idx], size=N, replace=False)
        ref_dist = normalize_distribution(Counter(arr[ref_idx]))
        sample_dist = normalize_distribution(Counter(query_sample))
        if metric == "tv":
            dist = total_variation(sample_dist, ref_dist)
        else:
            dist = js_divergence(sample_dist, ref_dist)
        scores.append(1.0 - dist)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def bootstrap_curve(labels: pd.DataFrame, gold: pd.DataFrame | None, mode: str, Ns: Sequence[int], B: int, seed: int, metric: str = "jsd") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = labels.copy()
    labels["item_id"] = labels["item_id"].astype(str)
    labels["label"] = labels["label"].astype(str)
    label_values = sorted(labels["label"].unique())
    label_to_code = {label: i for i, label in enumerate(label_values)}
    n_label_values = len(label_values)
    labels["_label_code"] = labels["label"].map(label_to_code).astype(int)
    item_labels = {
        item_id: grp["_label_code"].to_numpy(dtype=np.int16)
        for item_id, grp in labels.groupby("item_id", sort=False)
    }
    items = np.array(list(item_labels.keys()), dtype=object)
    Ns = [int(n) for n in Ns]
    rows = []
    if mode == "gold_accuracy":
        if gold is None:
            raise ValueError("gold dataframe is required for gold_accuracy mode")
        gold_labels = gold["gold_label"].astype(str)
        unknown_gold = sorted(set(gold_labels) - set(label_to_code))
        if unknown_gold:
            next_code = n_label_values
            for label in unknown_gold:
                label_to_code[label] = next_code
                next_code += 1
            n_label_values = next_code
        gold_map = dict(zip(gold["item_id"].astype(str), gold_labels.map(label_to_code).astype(int)))
        prob = np.full(len(items), 1.0 / len(items))
        ns_arr = np.asarray(Ns, dtype=int)
        for b in range(B):
            item_weights = rng.multinomial(len(items), prob)
            sums_arr = np.zeros(len(Ns), dtype=float)
            counts_arr = np.zeros(len(Ns), dtype=float)
            for item_id, weight in zip(items, item_weights):
                if weight <= 0:
                    continue
                if item_id not in gold_map:
                    continue
                arr = item_labels[item_id]
                valid_mask = ns_arr <= len(arr)
                if not np.any(valid_mask):
                    continue
                max_valid_n = int(ns_arr[valid_mask].max())
                sample = rng.choice(arr, size=max_valid_n, replace=False)
                onehot = np.zeros((max_valid_n, n_label_values), dtype=np.int16)
                onehot[np.arange(max_valid_n), sample.astype(int)] = 1
                cumulative = np.cumsum(onehot, axis=0)
                majorities = np.argmax(cumulative[ns_arr[valid_mask] - 1], axis=1)
                hits = (majorities == gold_map[item_id]).astype(float)
                sums_arr[valid_mask] += float(weight) * hits
                counts_arr[valid_mask] += float(weight)
            for idx, N in enumerate(Ns):
                val = sums_arr[idx] / counts_arr[idx] if counts_arr[idx] else float("nan")
                rows.append({"bootstrap": b, "N": int(N), "C": val})
    elif mode == "reference_distribution":
        prob = np.full(len(items), 1.0 / len(items))
        ns_arr = np.asarray(Ns, dtype=int)
        for b in range(B):
            item_weights = rng.multinomial(len(items), prob)
            sums_arr = np.zeros(len(Ns), dtype=float)
            counts_arr = np.zeros(len(Ns), dtype=float)
            for item_id, weight in zip(items, item_weights):
                if weight <= 0:
                    continue
                arr = item_labels[item_id]
                if len(arr) < 10:
                    continue
                valid_mask = (2 * ns_arr) <= len(arr)
                if not np.any(valid_mask):
                    continue
                perm = rng.permutation(len(arr))
                ref = arr[perm[: len(arr)//2]]
                query = arr[perm[len(arr)//2:]]
                if len(ref) < 2 or len(query) < int(ns_arr[valid_mask].min()):
                    continue
                ref_counts = np.bincount(ref, minlength=n_label_values).astype(float)
                ref_probs = ref_counts / max(ref_counts.sum(), 1e-12)
                query_perm = rng.permutation(len(query))
                max_valid_n = int(ns_arr[valid_mask].max())
                query_sample = query[query_perm[:max_valid_n]].astype(int)
                onehot = np.zeros((max_valid_n, n_label_values), dtype=float)
                onehot[np.arange(max_valid_n), query_sample] = 1.0
                cumulative = np.cumsum(onehot, axis=0)
                q_counts = cumulative[ns_arr[valid_mask] - 1]
                q_probs = q_counts / ns_arr[valid_mask, None]
                if metric == "tv":
                    dist_vals = 0.5 * np.abs(q_probs - ref_probs[None, :]).sum(axis=1)
                else:
                    mix = 0.5 * (q_probs + ref_probs[None, :])
                    q_terms = np.where(q_probs > 0, q_probs * np.log2((q_probs + 1e-12) / (mix + 1e-12)), 0.0)
                    r_terms = np.where(ref_probs[None, :] > 0, ref_probs[None, :] * np.log2((ref_probs[None, :] + 1e-12) / (mix + 1e-12)), 0.0)
                    dist_vals = 0.5 * q_terms.sum(axis=1) + 0.5 * r_terms.sum(axis=1)
                scores = 1.0 - dist_vals
                sums_arr[valid_mask] += float(weight) * scores
                counts_arr[valid_mask] += float(weight)
            for idx, N in enumerate(Ns):
                val = sums_arr[idx] / counts_arr[idx] if counts_arr[idx] else float("nan")
                rows.append({"bootstrap": b, "N": int(N), "C": val})
    else:
        raise ValueError(f"unknown mode: {mode}")
    return pd.DataFrame(rows)


def summarize_curve(curve: pd.DataFrame) -> pd.DataFrame:
    return curve.groupby("N", as_index=False).agg(
        C_mean=("C", "mean"),
        C_lo=("C", lambda x: np.nanpercentile(x, 2.5)),
        C_hi=("C", lambda x: np.nanpercentile(x, 97.5)),
        valid=("C", lambda x: np.isfinite(x).sum()),
    )


def utility_optimum(Ns: Sequence[int], C: Sequence[float], lambdas: Sequence[float], N_budget: float) -> pd.DataFrame:
    Ns_arr = np.asarray(Ns, dtype=float)
    C_arr = np.asarray(C, dtype=float)
    # normalize to [0,1] using observed span unless a fitted C_ref is supplied elsewhere
    denom = np.nanmax(C_arr) - np.nanmin(C_arr)
    if not np.isfinite(denom) or denom <= 1e-12:
        C_norm = np.zeros_like(C_arr)
    else:
        C_norm = (C_arr - np.nanmin(C_arr)) / denom
    rows = []
    for lam in lambdas:
        U = lam * C_norm - (1.0 - lam) * (Ns_arr / float(N_budget))
        idx = int(np.nanargmax(U))
        rows.append({"lambda": float(lam), "N_star": int(Ns_arr[idx]), "U_star": float(U[idx])})
    return pd.DataFrame(rows)
