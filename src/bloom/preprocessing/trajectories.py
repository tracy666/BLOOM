"""Build complete probabilistic pseudo-trajectories from sequential OT couplings.

The source algorithm retains the union of top-k and cumulative-mass candidates,
uses deterministic global-row tie breaking or seeded stochastic sampling,
stitches backward/forward paths through each anchor, and records anchor support.
Input couplings and observation rows must come from the same AnnData row order.
"""

from __future__ import annotations

import os
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import argparse
import yaml

_HAVE_YAML = True
try:
    _ = yaml.dump
except Exception:
    _HAVE_YAML = False


@dataclass
class Config:
    """Explicit configuration consumed by the preserved trajectory algorithm."""

    ADATA_OBS_PATH: str
    transitions_dir: str
    time_key: str
    celltype_key: str
    edge_mode: str
    top_k: int
    cum_mass: float
    min_mass: float
    max_edges_per_row: Optional[int]
    stable_tie_break: bool
    seed: int
    stochastic: bool
    temperature: float
    trajectories_per_start_cell: int
    hard_celltype: bool
    switch_penalty: float
    switch_count_penalty: float
    flip_flop_penalty: float
    use_anchors: bool
    anchor_times: Optional[List[str]]
    anchor_stride: int
    anchors_direction: str
    trajectories_per_anchor: Optional[int]
    max_start_cells_per_anchor: Optional[int]
    min_anchor_support: int
    out_dir: str
    save_npy: bool
    save_jsonl: bool
    save_partial_each_anchor: bool


def _time_sort_key_robust(t: str) -> Tuple:
    """
    This key tries:
      1) pure float
      2) prefix+float patterns like E14.5, P0, Day12, etc.
      3) fallback to string

    The exact ordering of prefixes is adjustable; current heuristic:
      E < P < others
    """
    s = str(t).strip()

    try:
        return (0, float(s))
    except Exception:
        pass

    m = re.match(r"^([A-Za-z]+)?\s*([+-]?\d+(?:\.\d+)?)\s*$", s)
    if m is not None:
        prefix = (m.group(1) or "").upper()
        num = float(m.group(2))

        prefix_rank = {
            "E": 0,
            "P": 1,
        }.get(prefix, 9)

        return (1, prefix_rank, num, prefix, s)

    return (2, s)


def sorted_times_from_obs(obs: pd.DataFrame, time_key: str) -> List[str]:
    ts = obs[time_key].astype(str).unique().tolist()
    ts.sort(key=_time_sort_key_robust)
    return ts


def matrix_path(transitions_dir: Path, t0: str, t1: str) -> Path:
    return transitions_dir / f"start_{t0}_end_{t1}_transport_matrix.npy"


def load_matrix(transitions_dir: Path, t0: str, t1: str) -> np.ndarray:
    fn = matrix_path(transitions_dir, t0, t1)
    if not fn.exists():
        raise FileNotFoundError(f"Missing OT matrix: {fn}")
    P = np.load(fn)
    if P.ndim != 2:
        raise ValueError(f"Expected 2D matrix at {fn}, got shape {P.shape}")
    return P


def _safe_choice_index(rng: np.random.Generator, p: np.ndarray) -> int:
    """Sample index robustly from p (handles NaN/inf/neg); fallback to uniform."""
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)

    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        return int(rng.integers(0, len(p)))

    p = p / s
    p = np.clip(p, 0.0, 1.0)
    s2 = float(p.sum())
    if not np.isfinite(s2) or s2 <= 0.0:
        return int(rng.integers(0, len(p)))
    p = p / s2
    return int(rng.choice(len(p), p=p))


def _argmax_stable_by_global_id(next_nodes: np.ndarray, w: np.ndarray, eps: float = 0.0) -> int:
    """
    Deterministic argmax:
      - find max weight (within eps tolerance)
      - among ties, choose the smallest GLOBAL cell id
    Returns index j into next_nodes.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.size == 0:
        return 0
    mx = float(np.max(w))
    if not np.isfinite(mx):
        return int(np.argmin(next_nodes)) if next_nodes.size else 0
    if eps > 0:
        mask = w >= mx - float(eps)
    else:
        mask = w == mx
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return int(np.argmax(w))
    tied = next_nodes[idx]
    j = int(idx[int(np.argmin(tied))])  # smallest global id among ties
    return j


def _apply_type_dynamics_penalty(
    base_w: np.ndarray,
    *,
    prev_type: int,
    prev_prev_type: Optional[int],
    tgt_types: np.ndarray,
    n_switches_so_far: int,
    switch_penalty: float,
    switch_count_penalty: float,
    flip_flop_penalty: float,
    temperature: float,
) -> np.ndarray:
    """
    Modify weights to discourage switching and flip-flops.
    Returned vector is normalized if possible; if sum<=0, returns unnormalized (caller handles).
    """
    w = np.asarray(base_w, dtype=np.float64).copy()
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0] = 0.0

    switch = tgt_types != int(prev_type)

    if switch_penalty and switch_penalty > 0:
        w[switch] *= math.exp(-float(switch_penalty))
    if switch_count_penalty and switch_count_penalty > 0:
        w[switch] *= math.exp(-float(switch_count_penalty) * float(n_switches_so_far))
    if flip_flop_penalty and flip_flop_penalty > 0 and prev_prev_type is not None:
        flip = (tgt_types == int(prev_prev_type)) & (tgt_types != int(prev_type))
        w[flip] *= math.exp(-float(flip_flop_penalty))
    if temperature and temperature > 0 and float(temperature) != 1.0:
        w = np.power(w, 1.0 / float(temperature))

    s = float(w.sum())
    if not np.isfinite(s) or s <= 0.0:
        return w
    return w / s


def _select_targets_for_row(row: np.ndarray, *, cfg: "Config") -> np.ndarray:
    """
    Deterministic edge selection for a single source row.

    Returns:
      js_loc: np.ndarray of target LOCAL indices (sorted by descending mass; ties stable)
    """
    if row.size == 0:
        return np.array([], dtype=np.int32)

    row = np.asarray(row, dtype=np.float64)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    row[row < 0] = 0.0

    mode = str(cfg.edge_mode).lower()
    if mode not in ("filtered", "all_nonzero"):
        raise ValueError(f"edge_mode must be 'filtered' or 'all_nonzero', got {cfg.edge_mode!r}")

    if mode == "all_nonzero":
        if float(cfg.min_mass) > 0.0:
            idx = np.flatnonzero(row >= float(cfg.min_mass)).astype(np.int32, copy=False)
        else:
            idx = np.flatnonzero(row > 0.0).astype(np.int32, copy=False)

        if idx.size == 0:
            return np.array([], dtype=np.int32)

        val = row[idx]
        if cfg.stable_tie_break:
            order = np.lexsort((idx, -val))  # (-val desc, idx asc)
        else:
            order = np.argsort(val)[::-1]
        idx = idx[order]

        if cfg.max_edges_per_row is not None and idx.size > int(cfg.max_edges_per_row):
            idx = idx[: int(cfg.max_edges_per_row)]
        return idx.astype(np.int32, copy=False)

    if float(cfg.min_mass) > 0.0:
        cand = np.flatnonzero(row >= float(cfg.min_mass)).astype(np.int32, copy=False)
        if cand.size == 0:
            return np.array([], dtype=np.int32)
    else:
        cand = np.arange(row.size, dtype=np.int32)

    cand_val = row[cand]

    if cfg.stable_tie_break:
        order = np.lexsort((cand, -cand_val))
    else:
        order = np.argsort(cand_val)[::-1]
    cand = cand[order]
    cand_val = cand_val[order]

    keep = np.zeros(cand.size, dtype=bool)

    if cfg.top_k is not None and int(cfg.top_k) > 0:
        keep[: min(int(cfg.top_k), cand.size)] = True

    if cfg.cum_mass is not None and 0.0 < float(cfg.cum_mass) < 1.0:
        total = float(cand_val.sum())
        if total > 0:
            csum = np.cumsum(cand_val)
            j = int(np.searchsorted(csum, float(cfg.cum_mass) * total, side="left"))
            keep[: min(j + 1, cand.size)] = True

    if not keep.any():
        j0 = int(np.argmax(row))
        if float(row[j0]) > 0:
            return np.array([j0], dtype=np.int32)
        return np.array([], dtype=np.int32)

    idx = cand[keep]
    val = row[idx]
    if cfg.stable_tie_break:
        order2 = np.lexsort((idx, -val))
    else:
        order2 = np.argsort(val)[::-1]
    idx = idx[order2]

    if cfg.max_edges_per_row is not None and idx.size > int(cfg.max_edges_per_row):
        idx = idx[: int(cfg.max_edges_per_row)]
    return idx.astype(np.int32, copy=False)


def _normalize_rows(P: np.ndarray) -> np.ndarray:
    """Row-normalize robustly (NaN/inf/neg => 0)."""
    P = np.asarray(P, dtype=np.float64)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P[P < 0] = 0.0
    rs = P.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    return P / rs


def build_step_adjacency(
    P: np.ndarray,
    src_global: np.ndarray,
    tgt_global: np.ndarray,
    src_type_codes: np.ndarray,
    tgt_type_codes: np.ndarray,
    *,
    cfg: "Config",
    desc: str = "",
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Build adjacency for one step:
      adj[src_global_id] = array of tgt_global_ids
      wts[src_global_id] = array of normalized weights aligned with adj[src]
    """
    if P.shape != (len(src_global), len(tgt_global)):
        raise ValueError(
            f"Shape mismatch: P{P.shape} vs src {len(src_global)} tgt {len(tgt_global)}"
        )

    adj: Dict[int, np.ndarray] = {}
    wts: Dict[int, np.ndarray] = {}

    for i_loc in tqdm(range(P.shape[0]), desc=desc, leave=False):
        row = P[i_loc]
        js_loc = _select_targets_for_row(row, cfg=cfg)
        if js_loc.size == 0:
            continue

        if cfg.hard_celltype:
            st = int(src_type_codes[i_loc])
            mask = tgt_type_codes[js_loc] == st
            js_loc = js_loc[mask]
            if js_loc.size == 0:
                continue

        tgt_g = tgt_global[js_loc].astype(np.int64, copy=False)

        w = np.asarray(row[js_loc], dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w[w < 0] = 0.0
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0.0:
            w = np.ones_like(w, dtype=np.float32) / len(w)
        else:
            w = (w / s).astype(np.float32)

        src_g = int(src_global[i_loc])
        adj[src_g] = tgt_g
        wts[src_g] = w

    return adj, wts


def step_outdegree_stats(
    *,
    step_adj: List[Dict[int, np.ndarray]],
    src_cells_by_step: List[np.ndarray],
    label: str,
) -> List[Dict[str, Any]]:
    """
    For each step:
      - avg out-degree
      - 95th percentile out-degree
      - fraction of src nodes with >=1 outgoing edge
    """
    out: List[Dict[str, Any]] = []
    for i, (adj, src_cells) in enumerate(zip(step_adj, src_cells_by_step)):
        src_cells = np.asarray(src_cells, dtype=np.int64)
        if src_cells.size == 0:
            out.append(
                dict(
                    step=i,
                    label=label,
                    n_src=0,
                    n_with_edge=0,
                    avg_out_degree=0.0,
                    p95_out_degree=0.0,
                    frac_with_edge=0.0,
                )
            )
            continue

        deg = np.fromiter(
            (len(adj.get(int(c), ())) for c in src_cells), dtype=np.int64, count=src_cells.size
        )
        n_src = int(src_cells.size)
        n_with = int((deg > 0).sum())
        out.append(
            dict(
                step=i,
                label=label,
                n_src=n_src,
                n_with_edge=n_with,
                avg_out_degree=float(deg.mean()),
                p95_out_degree=float(np.percentile(deg, 95)),
                frac_with_edge=float(n_with / n_src) if n_src > 0 else 0.0,
            )
        )
    return out


def anchor_support_histogram(support_map: Dict[Tuple[int, ...], Set[str]]) -> Dict[str, int]:
    if not support_map:
        return {}
    ks = sorted({len(s) for s in support_map.values()})
    return {str(k): int(sum(1 for s in support_map.values() if len(s) == k)) for k in ks}


def anchor_support_summary(support_map: Dict[Tuple[int, ...], Set[str]]) -> Dict[str, Any]:
    n = int(len(support_map))
    if n == 0:
        return dict(n_unique_sequences=0, n_support_eq1=0, n_support_ge2=0, frac_ge2=0.0)
    eq1 = int(sum(1 for s in support_map.values() if len(s) == 1))
    ge2 = int(sum(1 for s in support_map.values() if len(s) >= 2))
    return dict(
        n_unique_sequences=n,
        n_support_eq1=eq1,
        n_support_ge2=ge2,
        frac_ge2=float(ge2 / n),
    )


def _sample_forward_from_anchor(
    *,
    anchor_idx: int,
    times: List[str],
    forward_steps: List[Dict[int, np.ndarray]],
    weight_steps: List[Dict[int, np.ndarray]],
    celltype_codes_global: np.ndarray,
    start_cells: np.ndarray,
    cfg: "Config",
    rng: np.random.Generator,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """
    Returns:
      paths_by_start[start_cell] = [start_cell, ..., cell_at_end]
      stats dict (counts + failure rates)
    """
    paths_by_start: Dict[int, List[int]] = {}
    last_step = len(times) - 1

    n_start = int(len(start_cells))
    n_completed = 0
    n_failed_no_edge = 0
    n_failed_weight_mismatch = 0

    for start in start_cells.tolist():
        start = int(start)
        path = [start]
        cur = start
        ok = True

        prev_type = int(celltype_codes_global[cur])
        prev_prev_type: Optional[int] = None
        n_switches = 0

        for step in range(anchor_idx, last_step):
            nxts = forward_steps[step].get(cur, None)
            if nxts is None or len(nxts) == 0:
                ok = False
                n_failed_no_edge += 1
                break

            w = weight_steps[step].get(cur, None)
            if w is None or len(w) != len(nxts):
                ok = False
                n_failed_weight_mismatch += 1
                break

            if cfg.stochastic:
                p = np.asarray(w, dtype=np.float64)
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                p = np.clip(p, 0.0, None)
                s = float(p.sum())
                p = (np.ones_like(p) / len(p)) if (not np.isfinite(s) or s <= 0) else (p / s)

                if not cfg.hard_celltype:
                    tgt_types = celltype_codes_global[nxts]
                    p = _apply_type_dynamics_penalty(
                        p,
                        prev_type=prev_type,
                        prev_prev_type=prev_prev_type,
                        tgt_types=tgt_types,
                        n_switches_so_far=n_switches,
                        switch_penalty=cfg.switch_penalty,
                        switch_count_penalty=cfg.switch_count_penalty,
                        flip_flop_penalty=cfg.flip_flop_penalty,
                        temperature=cfg.temperature,
                    )
                    s2 = float(p.sum())
                    p = (np.ones_like(p) / len(p)) if (not np.isfinite(s2) or s2 <= 0) else (p / s2)

                j = _safe_choice_index(rng, p)
                nxt = int(nxts[j])
            else:
                j = _argmax_stable_by_global_id(nxts, w, eps=0.0)
                nxt = int(nxts[j])

            nxt_type = int(celltype_codes_global[nxt])
            if nxt_type != prev_type:
                n_switches += 1
            prev_prev_type = prev_type
            prev_type = nxt_type

            cur = nxt
            path.append(cur)

        if ok and len(path) == (len(times) - anchor_idx):
            paths_by_start[start] = path
            n_completed += 1

    stats = dict(
        n_start=n_start,
        n_completed=n_completed,
        n_failed_no_edge=n_failed_no_edge,
        n_failed_weight_mismatch=n_failed_weight_mismatch,
        fail_rate_no_edge=(n_failed_no_edge / n_start if n_start > 0 else 0.0),
        fail_rate_weight_mismatch=(n_failed_weight_mismatch / n_start if n_start > 0 else 0.0),
        fail_rate_total=(
            (n_failed_no_edge + n_failed_weight_mismatch) / n_start if n_start > 0 else 0.0
        ),
    )
    return paths_by_start, stats


def _sample_backward_from_anchor(
    *,
    anchor_idx: int,
    times: List[str],
    backward_steps: List[Dict[int, np.ndarray]],
    backward_weight_steps: List[Dict[int, np.ndarray]],
    celltype_codes_global: np.ndarray,
    start_cells: np.ndarray,
    cfg: "Config",
    rng: np.random.Generator,
) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """
    Returns:
      paths_by_start[start_cell] = [start_cell, ..., cell_at_t0]  (anchor->past order)
    """
    paths_by_start: Dict[int, List[int]] = {}

    n_start = int(len(start_cells))
    n_completed = 0
    n_failed_no_edge = 0
    n_failed_weight_mismatch = 0

    for start in start_cells.tolist():
        start = int(start)
        path = [start]
        cur = start
        ok = True

        prev_type = int(celltype_codes_global[cur])
        prev_prev_type: Optional[int] = None
        n_switches = 0

        for step in range(anchor_idx - 1, -1, -1):
            nxts = backward_steps[step].get(cur, None)
            if nxts is None or len(nxts) == 0:
                ok = False
                n_failed_no_edge += 1
                break

            w = backward_weight_steps[step].get(cur, None)
            if w is None or len(w) != len(nxts):
                ok = False
                n_failed_weight_mismatch += 1
                break

            if cfg.stochastic:
                p = np.asarray(w, dtype=np.float64)
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                p = np.clip(p, 0.0, None)
                s = float(p.sum())
                p = (np.ones_like(p) / len(p)) if (not np.isfinite(s) or s <= 0) else (p / s)

                if not cfg.hard_celltype:
                    tgt_types = celltype_codes_global[nxts]
                    p = _apply_type_dynamics_penalty(
                        p,
                        prev_type=prev_type,
                        prev_prev_type=prev_prev_type,
                        tgt_types=tgt_types,
                        n_switches_so_far=n_switches,
                        switch_penalty=cfg.switch_penalty,
                        switch_count_penalty=cfg.switch_count_penalty,
                        flip_flop_penalty=cfg.flip_flop_penalty,
                        temperature=cfg.temperature,
                    )
                    s2 = float(p.sum())
                    p = (np.ones_like(p) / len(p)) if (not np.isfinite(s2) or s2 <= 0) else (p / s2)

                j = _safe_choice_index(rng, p)
                nxt = int(nxts[j])
            else:
                j = _argmax_stable_by_global_id(nxts, w, eps=0.0)
                nxt = int(nxts[j])

            nxt_type = int(celltype_codes_global[nxt])
            if nxt_type != prev_type:
                n_switches += 1
            prev_prev_type = prev_type
            prev_type = nxt_type

            cur = nxt
            path.append(cur)

        if ok and len(path) == (anchor_idx + 1):
            paths_by_start[start] = path
            n_completed += 1

    stats = dict(
        n_start=n_start,
        n_completed=n_completed,
        n_failed_no_edge=n_failed_no_edge,
        n_failed_weight_mismatch=n_failed_weight_mismatch,
        fail_rate_no_edge=(n_failed_no_edge / n_start if n_start > 0 else 0.0),
        fail_rate_weight_mismatch=(n_failed_weight_mismatch / n_start if n_start > 0 else 0.0),
        fail_rate_total=(
            (n_failed_no_edge + n_failed_weight_mismatch) / n_start if n_start > 0 else 0.0
        ),
    )
    return paths_by_start, stats


def _merge_backward_forward(
    backward_path: Optional[List[int]], forward_path: Optional[List[int]]
) -> Optional[List[int]]:
    """
    backward_path: anchor->...->t0 order
    forward_path : anchor->...->t_end order
    output: t0..anchor..t_end
    """
    if backward_path is None and forward_path is None:
        return None
    if backward_path is None:
        return forward_path[:]
    if forward_path is None:
        return list(reversed(backward_path))
    if backward_path[0] != forward_path[0]:
        return None
    left = list(reversed(backward_path))
    right = forward_path[1:]
    return left + right


def _compute_anchor_indices(
    times: List[str], *, anchor_times: Optional[List[str]], anchor_stride: int
) -> List[int]:
    if anchor_times is None:
        return list(range(0, len(times), max(1, int(anchor_stride))))
    for t in anchor_times:
        if t not in times:
            raise ValueError(f"anchor_time {t} not in times")
    return [times.index(t) for t in anchor_times]


def build_trajectories(cfg: "Config"):
    out_dir = Path(
        os.path.join(
            cfg.out_dir,
            f"celltype_{cfg.celltype_key}"
            f"_edge_{cfg.edge_mode}"
            f"_topk_{cfg.top_k}"
            f"_cummass_{cfg.cum_mass}"
            f"_minmass_{cfg.min_mass}"
            f"_maxrow_{cfg.max_edges_per_row}"
            f"_stochastic_{cfg.stochastic}"
            f"_hardcell_{cfg.hard_celltype}"
            f"_anchordir_{cfg.anchors_direction}"
            f"_maxstartcell_{cfg.max_start_cells_per_anchor}"
            f"_minanchorsupp_{cfg.min_anchor_support}"
            f"_seed_{cfg.seed}",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    transitions_dir = Path(cfg.transitions_dir)

    if not transitions_dir.exists():
        raise FileNotFoundError(f"[config] transitions_dir does not exist: {transitions_dir}")

    rng = np.random.default_rng(int(cfg.seed))
    np.random.seed(int(cfg.seed))  # safety if any legacy np.random is used

    print(f"[load] obs: {cfg.ADATA_OBS_PATH}")
    if not os.path.exists(cfg.ADATA_OBS_PATH):
        raise FileNotFoundError(f"[config] ADATA_OBS_PATH does not exist: {cfg.ADATA_OBS_PATH}")
    obs = pd.read_parquet(cfg.ADATA_OBS_PATH)

    if cfg.time_key not in obs.columns:
        raise KeyError(f"{cfg.time_key} not found in obs columns")
    if cfg.celltype_key not in obs.columns:
        raise KeyError(f"{cfg.celltype_key} not found in obs columns")

    times = sorted_times_from_obs(obs, cfg.time_key)
    print(f"[time] Found {len(times)} timepoints: {times[0]} .. {times[-1]}")

    time_series = obs[cfg.time_key].astype(str)
    time_to_idx = {t: i for i, t in enumerate(times)}
    time_idx_global = time_series.map(time_to_idx).to_numpy(dtype=np.int32)

    celltype_cat = obs[cfg.celltype_key].astype("category")
    celltype_codes_global = celltype_cat.cat.codes.to_numpy(dtype=np.int32)
    celltype_names = list(celltype_cat.cat.categories)
    print(f"[celltype] {cfg.celltype_key}: {len(celltype_names)} categories")

    cells_by_time: List[np.ndarray] = []
    typecodes_by_time: List[np.ndarray] = []
    for ti in range(len(times)):
        idx = np.flatnonzero(time_idx_global == ti).astype(np.int64)
        cells_by_time.append(idx)
        typecodes_by_time.append(celltype_codes_global[idx])

    forward_steps: List[Dict[int, np.ndarray]] = []
    weight_steps: List[Dict[int, np.ndarray]] = []
    print("[build] forward adjacency...")
    for i in tqdm(range(len(times) - 1), desc="forward steps"):
        t0, t1 = times[i], times[i + 1]
        P = load_matrix(transitions_dir, t0, t1)

        fwd, wts = build_step_adjacency(
            P,
            src_global=cells_by_time[i],
            tgt_global=cells_by_time[i + 1],
            src_type_codes=typecodes_by_time[i],
            tgt_type_codes=typecodes_by_time[i + 1],
            cfg=cfg,
            desc=f"fwd {t0}->{t1}",
        )
        forward_steps.append(fwd)
        weight_steps.append(wts)

        print(f"  step {t0}->{t1}: src_with_edges={len(fwd)}/{len(cells_by_time[i])}")

    backward_steps: List[Dict[int, np.ndarray]] = []
    backward_weight_steps: List[Dict[int, np.ndarray]] = []
    print("[build] backward adjacency (transpose + row-normalize)...")
    for i in tqdm(range(len(times) - 1), desc="backward steps"):
        t0, t1 = times[i], times[i + 1]
        P = load_matrix(transitions_dir, t0, t1)
        Pb = _normalize_rows(P.T)

        bwd, bwts = build_step_adjacency(
            Pb,
            src_global=cells_by_time[i + 1],
            tgt_global=cells_by_time[i],
            src_type_codes=typecodes_by_time[i + 1],
            tgt_type_codes=typecodes_by_time[i],
            cfg=cfg,
            desc=f"bwd {t1}->{t0}",
        )
        backward_steps.append(bwd)
        backward_weight_steps.append(bwts)

        print(f"  step {t1}->{t0}: src_with_edges={len(bwd)}/{len(cells_by_time[i+1])}")

    forward_src_cells_by_step = [cells_by_time[i] for i in range(len(times) - 1)]
    backward_src_cells_by_step = [cells_by_time[i + 1] for i in range(len(times) - 1)]
    forward_graph_stats = step_outdegree_stats(
        step_adj=forward_steps, src_cells_by_step=forward_src_cells_by_step, label="forward"
    )
    backward_graph_stats = step_outdegree_stats(
        step_adj=backward_steps, src_cells_by_step=backward_src_cells_by_step, label="backward"
    )

    if cfg.anchors_direction not in ("forward", "backward", "both"):
        raise ValueError(
            f"anchors_direction must be 'forward'|'backward'|'both', got {cfg.anchors_direction!r}"
        )

    if not cfg.use_anchors:
        anchor_indices = [0]
        cfg.anchors_direction = "forward"
    else:
        anchor_indices = _compute_anchor_indices(
            times, anchor_times=cfg.anchor_times, anchor_stride=cfg.anchor_stride
        )

    if int(cfg.min_anchor_support) < 1:
        raise ValueError("min_anchor_support must be >= 1")

    support_map: Dict[Tuple[int, ...], Set[str]] = {}

    forward_tot = dict(n_start=0, n_completed=0, n_failed_no_edge=0, n_failed_weight_mismatch=0)
    backward_tot = dict(n_start=0, n_completed=0, n_failed_no_edge=0, n_failed_weight_mismatch=0)

    processed_anchors: List[str] = []

    print("[paths] processing anchors...")
    for ai, a_idx in enumerate(tqdm(anchor_indices, desc="anchors")):
        a_time = str(times[a_idx])
        processed_anchors.append(a_time)

        anchor_cells = cells_by_time[a_idx].astype(np.int64, copy=False)

        if cfg.trajectories_per_anchor is None:
            start_cells = anchor_cells
        else:
            n = min(int(cfg.trajectories_per_anchor), len(anchor_cells))
            start_cells = rng.choice(anchor_cells, size=n, replace=False)

        if cfg.max_start_cells_per_anchor is not None:
            cap = int(cfg.max_start_cells_per_anchor)
            if len(start_cells) > cap:
                start_cells = rng.choice(start_cells, size=cap, replace=False)

        start_cells = np.sort(start_cells.astype(np.int64, copy=False))

        f_map: Dict[int, List[int]] = {}
        b_map: Dict[int, List[int]] = {}

        if cfg.anchors_direction in ("forward", "both") and a_idx < (len(times) - 1):
            f_map, f_stats = _sample_forward_from_anchor(
                anchor_idx=a_idx,
                times=times,
                forward_steps=forward_steps,
                weight_steps=weight_steps,
                celltype_codes_global=celltype_codes_global,
                start_cells=start_cells,
                cfg=cfg,
                rng=rng,
            )
            for k in forward_tot:
                forward_tot[k] += int(f_stats.get(k, 0))

        if cfg.anchors_direction in ("backward", "both") and a_idx > 0:
            b_map, b_stats = _sample_backward_from_anchor(
                anchor_idx=a_idx,
                times=times,
                backward_steps=backward_steps,
                backward_weight_steps=backward_weight_steps,
                celltype_codes_global=celltype_codes_global,
                start_cells=start_cells,
                cfg=cfg,
                rng=rng,
            )
            for k in backward_tot:
                backward_tot[k] += int(b_stats.get(k, 0))

        reps = max(1, int(cfg.trajectories_per_start_cell))
        for cell0 in start_cells.tolist():
            cell0 = int(cell0)

            bp0 = b_map.get(cell0, None)
            fp0 = f_map.get(cell0, None)

            if cfg.anchors_direction == "forward" and fp0 is None:
                continue
            if cfg.anchors_direction == "backward" and bp0 is None:
                continue
            if cfg.anchors_direction == "both" and (bp0 is None or fp0 is None):
                continue

            full0 = _merge_backward_forward(bp0, fp0)
            if full0 is None:
                continue
            if cfg.anchors_direction == "both" and len(full0) != len(times):
                continue

            seq0 = tuple(int(x) for x in full0)
            support_map.setdefault(seq0, set()).add(a_time)

            if cfg.stochastic and reps > 1:
                one = np.array([cell0], dtype=np.int64)
                for _r in range(reps - 1):
                    bp = bp0
                    fp = fp0
                    if cfg.anchors_direction in ("backward", "both") and a_idx > 0:
                        btmp, _ = _sample_backward_from_anchor(
                            anchor_idx=a_idx,
                            times=times,
                            backward_steps=backward_steps,
                            backward_weight_steps=backward_weight_steps,
                            celltype_codes_global=celltype_codes_global,
                            start_cells=one,
                            cfg=cfg,
                            rng=rng,
                        )
                        bp = btmp.get(cell0, None)
                    if cfg.anchors_direction in ("forward", "both") and a_idx < (len(times) - 1):
                        ftmp, _ = _sample_forward_from_anchor(
                            anchor_idx=a_idx,
                            times=times,
                            forward_steps=forward_steps,
                            weight_steps=weight_steps,
                            celltype_codes_global=celltype_codes_global,
                            start_cells=one,
                            cfg=cfg,
                            rng=rng,
                        )
                        fp = ftmp.get(cell0, None)

                    if cfg.anchors_direction == "both" and (bp is None or fp is None):
                        continue
                    full = _merge_backward_forward(bp, fp)
                    if full is None:
                        continue
                    if cfg.anchors_direction == "both" and len(full) != len(times):
                        continue
                    seq = tuple(int(x) for x in full)
                    support_map.setdefault(seq, set()).add(a_time)

        if cfg.save_partial_each_anchor:
            (out_dir / "partial_state.json").write_text(
                json.dumps(
                    {
                        "processed_anchors": processed_anchors,
                        "n_unique_sequences_so_far": int(len(support_map)),
                        "last_anchor_time": a_time,
                        "config": cfg.__dict__,
                    },
                    indent=2,
                )
            )

    support_hist = anchor_support_histogram(support_map)
    support_sum = anchor_support_summary(support_map)

    kept = [seq for seq, supp in support_map.items() if len(supp) >= int(cfg.min_anchor_support)]
    kept.sort(key=lambda s: (-len(support_map[s]), s))  # deterministic ordering

    trajs = [list(seq) for seq in kept]

    def _rate(d: Dict[str, int]) -> Dict[str, Any]:
        n_start = int(d["n_start"])
        n_failed = int(d["n_failed_no_edge"]) + int(d["n_failed_weight_mismatch"])
        return dict(
            n_start=n_start,
            n_completed=int(d["n_completed"]),
            n_failed_no_edge=int(d["n_failed_no_edge"]),
            n_failed_weight_mismatch=int(d["n_failed_weight_mismatch"]),
            fail_rate_no_edge=(float(d["n_failed_no_edge"]) / n_start if n_start > 0 else 0.0),
            fail_rate_weight_mismatch=(
                float(d["n_failed_weight_mismatch"]) / n_start if n_start > 0 else 0.0
            ),
            fail_rate_total=(float(n_failed) / n_start if n_start > 0 else 0.0),
        )

    path_completion = dict(
        forward=_rate(forward_tot),
        backward=_rate(backward_tot),
    )

    meta = {
        "mode": "paths",
        "config": cfg.__dict__,
        "times": times,
        "n_unique_sequences": int(len(support_map)),
        "n_kept_sequences": int(len(trajs)),
        "min_anchor_support": int(cfg.min_anchor_support),
        "support_histogram": support_hist,
        "anchor_support_summary": support_sum,
        "graph_stats": {
            "forward": forward_graph_stats,
            "backward": backward_graph_stats,
        },
        "path_completion": path_completion,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[save] meta.json: {out_dir / 'meta.json'}")

    if cfg.save_npy:
        np.save(
            out_dir / "trajectories.npy",
            np.array([np.array(p, dtype=np.int64) for p in trajs], dtype=object),
        )
        print(f"[save] trajectories.npy: {out_dir / 'trajectories.npy'}")

        supp_list = [sorted(list(support_map[tuple(p)])) for p in trajs]
        np.save(out_dir / "trajectory_support_anchors.npy", np.array(supp_list, dtype=object))
        print(
            f"[save] trajectory_support_anchors.npy: {out_dir / 'trajectory_support_anchors.npy'}"
        )

    if cfg.save_jsonl:
        fn = out_dir / "trajectories.jsonl"
        with fn.open("w") as f:
            for p in trajs:
                supp = sorted(list(support_map[tuple(p)]))
                f.write(json.dumps({"path": [int(x) for x in p], "anchors": supp}) + "\n")
        print(f"[save] trajectories.jsonl: {fn}")

    config_filename = out_dir / "config.yaml"
    try:
        if not _HAVE_YAML:
            raise RuntimeError("PyYAML not available")

        with open(config_filename, "w") as f:
            yaml.dump(cfg.__dict__, f, default_flow_style=False)
        print(f"[save] Configuration saved to {config_filename}")
    except Exception:
        config_filename_json = out_dir / "config.json"
        with open(config_filename_json, "w") as f:
            json.dump(cfg.__dict__, f, indent=2)
        print(f"[save] Configuration saved to {config_filename_json}")

    print("[report] n_unique_sequences:", int(len(support_map)))
    print("[report] n_kept_sequences  :", int(len(trajs)))
    print("[report] anchor_support_summary:")
    print(json.dumps(support_sum, indent=2))
    print("[report] path_completion:")
    print(json.dumps(path_completion, indent=2))
    print("Done.")


def output_directory(cfg):
    """Return the source-defined directory for this exact trajectory recipe."""
    return Path(
        os.path.join(
            cfg.out_dir,
            f"celltype_{cfg.celltype_key}_edge_{cfg.edge_mode}_topk_{cfg.top_k}_cummass_{cfg.cum_mass}_minmass_{cfg.min_mass}_maxrow_{cfg.max_edges_per_row}_stochastic_{cfg.stochastic}_hardcell_{cfg.hard_celltype}_anchordir_{cfg.anchors_direction}_maxstartcell_{cfg.max_start_cells_per_anchor}_minanchorsupp_{cfg.min_anchor_support}_seed_{cfg.seed}",
        )
    )


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=None,
        help="Optional copy of trajectories.npy at the training-input path",
    )

    parser.add_argument(
        "--ADATA_OBS_PATH", type=str, required=True, help="Path to the AnnData dataset obs file"
    )

    parser.add_argument(
        "--transitions_dir",
        type=str,
        required=True,
        help="Directory containing the OT transition matrices",
    )

    parser.add_argument(
        "--transitions_origin",
        type=str,
        default="X_pca",
        choices=["X_pca", "X_scVI"],
        help="Origin of the OT transition matrices (used for subdirectory naming)",
    )

    parser.add_argument("--time_key", type=str, default="time", help="Column name in obs for time")

    parser.add_argument(
        "--celltype_key",
        type=str,
        default="class_label",
        help="Column name in obs for cell type (e.g. 'class_label' or 'subclass_label')",
    )

    parser.add_argument(
        "--edge_mode",
        type=str,
        default="filtered",
        choices=["filtered", "all_nonzero"],
        help="Edge selection mode: 'filtered' (union of top_k and cum_mass after min_mass) or 'all_nonzero' (keep all nonzero edges, optionally with min_mass threshold)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="For 'filtered' edge_mode: number of top edges to keep per row",
    )
    parser.add_argument(
        "--cum_mass",
        type=float,
        default=0.95,
        help="For 'filtered' edge_mode: cumulative mass threshold to keep edges per row",
    )
    parser.add_argument(
        "--min_mass",
        type=float,
        default=0.0,
        help="Minimum mass threshold for edges (applies to both modes)",
    )
    parser.add_argument(
        "--max_edges_per_row",
        type=int,
        default=None,
        help="Maximum number of edges to keep per row (None means no cap)",
    )

    parser.add_argument(
        "--stable_tie_break",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use deterministic stable sorting for tie-breaking (recommended)",
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Whether to sample trajectories stochastically (if False, uses deterministic max with stable tie-breaking)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for stochastic sampling (only used if --stochastic is set)",
    )
    parser.add_argument(
        "--trajectories_per_start_cell",
        type=int,
        default=1,
        help="Number of trajectories to sample per start cell (only meaningful if --stochastic is set)",
    )
    parser.add_argument(
        "--hard_celltype",
        action="store_true",
        default=False,
        help="Whether to enforce hard cell type constraints (only allow edges between same cell type)",
    )
    parser.add_argument(
        "--switch_penalty",
        type=float,
        default=2.0,
        help="Penalty factor for switching cell types in stochastic sampling (only used if --stochastic is set and --hard_celltype is not set)",
    )
    parser.add_argument(
        "--switch_count_penalty",
        type=float,
        default=0.5,
        help="Additional penalty factor that increases with the number of switches so far (only used if --stochastic is set and --hard_celltype is not set)",
    )
    parser.add_argument(
        "--flip_flop_penalty",
        type=float,
        default=3.0,
        help="Additional penalty factor for flip-flopping back to a previously visited cell type (only used if --stochastic is set and --hard_celltype is not set)",
    )

    parser.add_argument(
        "--use_anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use anchors for path sampling (if False, will treat all cells at the earliest time as a single anchor)",
    )

    parser.add_argument(
        "--anchor_times",
        type=str,
        default=None,
        help="Comma-separated list of specific time points to use as anchors (e.g. 'E14.5,P0'). If not set, will use all time points as potential anchors.",
    )
    parser.add_argument(
        "--anchor_stride",
        type=int,
        default=1,
        help="Stride for selecting anchors from the time points (e.g. 2 means every other time point)",
    )
    parser.add_argument(
        "--anchors_direction",
        type=str,
        default="both",
        choices=["forward", "backward", "both"],
        help="Direction for anchor-based sampling: 'forward' (sample forward trajectories from anchors), 'backward' (sample backward trajectories from anchors), or 'both' (sample both forward and backward and stitch)",
    )
    parser.add_argument(
        "--trajectories_per_anchor",
        type=int,
        default=None,
        help="Number of trajectories to sample per anchor cell (only meaningful if --stochastic is set). If not set, will sample trajectories_per_start_cell per anchor cell.",
    )
    parser.add_argument(
        "--max_start_cells_per_anchor",
        type=int,
        default=None,
        help="Maximum number of start cells to sample per anchor (set to None to use all anchor cells)",
    )
    parser.add_argument(
        "--min_anchor_support",
        type=int,
        default=2,
        help="Minimum number of anchors supporting a trajectory for it to be kept in the final output",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Base output directory to save results (final output will be in a subdirectory with key settings)",
    )

    parser.add_argument(
        "--save_npy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save trajectories and supports in .npy format",
    )
    parser.add_argument(
        "--save_jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save trajectories and supports in .jsonl format (one JSON object per line)",
    )
    parser.add_argument(
        "--save_partial_each_anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save a partial_state.json checkpoint after processing each anchor (useful for long runs)",
    )

    parser_argument = parser.parse_args()
    parser_argument_dict = vars(parser_argument)

    cfg = Config(
        ADATA_OBS_PATH=parser_argument_dict["ADATA_OBS_PATH"],
        transitions_dir=os.path.join(
            parser_argument_dict["transitions_dir"], parser_argument_dict["transitions_origin"]
        ),
        time_key=parser_argument_dict["time_key"],
        celltype_key=parser_argument_dict["celltype_key"],
        edge_mode=parser_argument_dict["edge_mode"],
        top_k=parser_argument_dict["top_k"],
        cum_mass=parser_argument_dict["cum_mass"],
        min_mass=parser_argument_dict["min_mass"],
        max_edges_per_row=parser_argument_dict["max_edges_per_row"],
        stable_tie_break=parser_argument_dict["stable_tie_break"],
        seed=parser_argument_dict["seed"],
        stochastic=parser_argument_dict["stochastic"],
        temperature=parser_argument_dict["temperature"],
        trajectories_per_start_cell=parser_argument_dict["trajectories_per_start_cell"],
        hard_celltype=parser_argument_dict["hard_celltype"],
        switch_penalty=parser_argument_dict["switch_penalty"],
        switch_count_penalty=parser_argument_dict["switch_count_penalty"],
        flip_flop_penalty=parser_argument_dict["flip_flop_penalty"],
        use_anchors=parser_argument_dict["use_anchors"],
        anchor_times=(
            None
            if parser_argument_dict["anchor_times"] is None
            else [
                t.strip() for t in str(parser_argument_dict["anchor_times"]).split(",") if t.strip()
            ]
        ),
        anchor_stride=parser_argument_dict["anchor_stride"],
        anchors_direction=parser_argument_dict["anchors_direction"],
        trajectories_per_anchor=parser_argument_dict["trajectories_per_anchor"],
        max_start_cells_per_anchor=parser_argument_dict["max_start_cells_per_anchor"],
        min_anchor_support=parser_argument_dict["min_anchor_support"],
        out_dir=os.path.join(
            parser_argument_dict["out_dir"], parser_argument_dict["transitions_origin"]
        ),
        save_npy=parser_argument_dict["save_npy"],
        save_jsonl=parser_argument_dict["save_jsonl"],
        save_partial_each_anchor=parser_argument_dict["save_partial_each_anchor"],
    )

    print("--------------- Configuration ---------------")
    for k in sorted(parser_argument_dict.keys()):
        print(f"{k:>30}: {parser_argument_dict[k]}")
    print("---------------------------------------------")

    build_trajectories(cfg)
    if parser_argument.canonical_output is not None:
        import shutil

        saved = np.load(output_directory(cfg) / "trajectories.npy", allow_pickle=True)
        if saved.size == 0:
            raise RuntimeError(
                "No full trajectories passed the anchor-support filter; inspect meta.json and revise the explicit recipe before training"
            )
        destination = parser_argument.canonical_output
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output_directory(cfg) / "trajectories.npy", destination)


if __name__ == "__main__":
    main()
