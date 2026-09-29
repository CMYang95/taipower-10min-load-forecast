"""分開預測：每天各自 origin、train<該日、只預測 24h（horizon=1）。

不做一次 origin 連續預測 72h。
Slot 回看預設同 cluster（同 weekday）日。
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from domain.cluster_assign import CLUSTER_NAMES, assign_cluster
from domain.dow_data_split import cluster_id_for_day, same_cluster_days_before
from domain.slot_adapt import apply_slot_adapt, estimate_slot_gap, metrics, origin_curve
from domain.train_dow_xgb import fit_pack_at_cutoff, predict_horizon_proto


def roll_daily_proto_curves(
    *,
    load10: pd.Series,
    holidays: pd.DataFrame,
    origins: list[pd.Timestamp],
    max_days_per_cluster: int,
    valid_start: pd.Timestamp,
    test_start: pd.Timestamp,
    quiet: bool = False,
) -> pd.DataFrame:
    """每個 origin：train<origin，只預測該日 24h。"""
    rows = []
    if not quiet:
        print(f"  分開日預測原點 {len(origins)}（各 origin 僅 h=1）", flush=True)
    for i, origin in enumerate(origins):
        if (not quiet) and i % 10 == 0:
            print(f"    {i+1}/{len(origins)} {pd.Timestamp(origin).date()}", flush=True)
        origin = pd.Timestamp(origin).normalize()
        pack = fit_pack_at_cutoff(
            load10=load10,
            holidays=holidays,
            train_cutoff=origin,
            max_days_per_cluster=max_days_per_cluster,
            quiet=True,
        )
        pred = predict_horizon_proto(pack["profiles"], pack["clusters"], origin, horizon_days=1)
        day = origin
        if pred.empty:
            continue
        if day < valid_start:
            split = "pre_valid"
        elif day < test_start:
            split = "valid"
        else:
            split = "test"
        cid = cluster_id_for_day(day, pack["clusters"])
        tmp = pred.rename("load_10").reset_index()
        tmp.columns = ["date_time", "load_10"]
        tmp["origin"] = day
        tmp["horizon"] = 1
        tmp["split"] = split
        tmp["cluster_id"] = cid
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def predict_separate_days(
    *,
    load10: pd.Series,
    holidays: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    max_days_per_cluster: int,
    quiet: bool = False,
) -> pd.Series:
    """評測窗內每一天獨立 fit + 預測 24h，再拼接。"""
    pieces: list[pd.Series] = []
    for i, day in enumerate(target_dates):
        day = pd.Timestamp(day).normalize()
        if not quiet:
            print(f"  D{i+1} {_day_label(day)} train<{day.date()} → 預測 24h", flush=True)
        pack = fit_pack_at_cutoff(
            load10=load10,
            holidays=holidays,
            train_cutoff=day,
            max_days_per_cluster=max_days_per_cluster,
            quiet=quiet,
        )
        pred = predict_horizon_proto(pack["profiles"], pack["clusters"], day, horizon_days=1)
        pieces.append(pred)
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def predict_dow_proto_resid_separate_days(
    *,
    actual: pd.Series,
    load10: pd.Series,
    holidays: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    max_days_per_cluster: int,
    idx_days: set,
    valid_start: pd.Timestamp,
    lookback_days: int = 2,
    beta: float = 1.0,
    quiet: bool = False,
) -> tuple[pd.Series, dict[pd.Timestamp, np.ndarray]]:
    """dow_proto：同群中位數原型 + 滾動日曆 D-1..D-N slot 殘差（actual-pred）補正。

    返回 (補正後預測, resid_by_day)；resid_by_day[D] 為該日使用的 gap，shape=(144,)。
    """
    pieces: list[pd.Series] = []
    resid_by_day: dict[pd.Timestamp, np.ndarray] = {}
    for i, day in enumerate(target_dates):
        day = pd.Timestamp(day).normalize()
        if not quiet:
            print(
                f"  D{i+1} {_day_label(day)} proto+resid lb={lookback_days} beta={beta}",
                flush=True,
            )
        pack = fit_pack_at_cutoff(
            load10=load10,
            holidays=holidays,
            train_cutoff=day,
            max_days_per_cluster=max_days_per_cluster,
            quiet=True,
        )
        pred = predict_horizon_proto(pack["profiles"], pack["clusters"], day, horizon_days=1)

        lb_origins = [
            day - pd.Timedelta(days=j)
            for j in range(1, int(lookback_days) + 1)
            if (day - pd.Timedelta(days=j)).normalize() in idx_days
        ]
        curves = roll_daily_proto_curves(
            load10=load10,
            holidays=holidays,
            origins=list(lb_origins),
            max_days_per_cluster=max_days_per_cluster,
            valid_start=valid_start,
            test_start=day + pd.Timedelta(days=1),
            quiet=True,
        )
        # same_cluster_days=None → 日曆 D-1..D-lookback
        gap, used = estimate_slot_gap(actual, curves, day, lookback_days, same_cluster_days=None)
        resid_by_day[day] = np.asarray(gap, dtype=float)
        if not quiet:
            print(
                f"    resid refs={[str(d.date()) for d in lb_origins]} days_used={used} "
                f"mean_gap={float(np.nanmean(gap)):.1f}",
                flush=True,
            )
        pieces.append(apply_slot_adapt(pred, gap, beta))
    return (pd.concat(pieces) if pieces else pd.Series(dtype=float)), resid_by_day


def apply_slot_separate_days(
    *,
    actual: pd.Series,
    load10: pd.Series,
    holidays: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    max_days_per_cluster: int,
    lookback_days: int,
    beta: float,
    idx_days: set,
    valid_start: pd.Timestamp,
    same_dow_only: bool = True,
    quiet: bool = False,
) -> tuple[pd.Series, int]:
    """分開日預測 + 各日各自 slot gap（預設同 cluster 回看）。"""
    clusters_all = assign_cluster(load10.index, holidays)
    slot_pieces: list[pd.Series] = []
    total_used = 0
    for day in target_dates:
        day = pd.Timestamp(day).normalize()
        pack = fit_pack_at_cutoff(
            load10=load10,
            holidays=holidays,
            train_cutoff=day,
            max_days_per_cluster=max_days_per_cluster,
            quiet=True,
        )
        pred = predict_horizon_proto(pack["profiles"], pack["clusters"], day, horizon_days=1)
        cid = cluster_id_for_day(day, pack["clusters"])

        if same_dow_only:
            same_days = same_cluster_days_before(day, cid, clusters_all, lookback_days)
            lb_origins = [d for d in same_days if d.normalize() in idx_days]
        else:
            lb_origins = [
                day - pd.Timedelta(days=i)
                for i in range(1, lookback_days + 1)
                if (day - pd.Timedelta(days=i)).normalize() in idx_days
            ]

        curves = roll_daily_proto_curves(
            load10=load10,
            holidays=holidays,
            origins=list(lb_origins),
            max_days_per_cluster=max_days_per_cluster,
            valid_start=valid_start,
            test_start=day + pd.Timedelta(days=1),
            quiet=True,
        )
        gap, used = estimate_slot_gap(
            actual,
            curves,
            day,
            lookback_days,
            same_cluster_days=lb_origins if same_dow_only else None,
        )
        total_used = max(total_used, used)
        if not quiet:
            print(
                f"  slot {day.date()} cid={cid} lookback={lookback_days} "
                f"same_dow={same_dow_only} days_used={used}",
                flush=True,
            )
        slot_pieces.append(apply_slot_adapt(pred, gap, beta))
    return pd.concat(slot_pieces) if slot_pieces else pd.Series(dtype=float), total_used


def select_slot_on_valid_daily(
    actual: pd.Series,
    curves: pd.DataFrame,
    *,
    valid_start: pd.Timestamp,
    test_start: pd.Timestamp,
    lookbacks: list[int],
    betas: list[float],
    clusters: pd.Series | None = None,
    same_dow_only: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """valid 期：每日 origin 只評估該日 24h；slot 預設同 cluster 回看。"""
    valid_days = sorted(
        {
            pd.Timestamp(o).normalize()
            for o in curves["origin"].unique()
            if valid_start <= pd.Timestamp(o).normalize() < test_start
        }
    )
    rows = []
    best = None
    for lb, beta in product(lookbacks, betas):
        maes = []
        mapes = []
        for day in valid_days:
            base = origin_curve(curves, day, 1)
            if base.empty:
                continue
            y = actual.reindex(base.index)
            if y.notna().sum() < len(base) // 2:
                continue
            same_days = None
            if same_dow_only and clusters is not None:
                cid = cluster_id_for_day(day, clusters)
                same_days = same_cluster_days_before(day, cid, clusters, lb)
            gap, used = estimate_slot_gap(
                actual, curves, day, lb, same_cluster_days=same_days
            )
            if used == 0:
                continue
            adapted = apply_slot_adapt(base, gap, beta)
            m = metrics(y.to_numpy(), adapted.to_numpy())
            if np.isfinite(m["MAE"]):
                maes.append(m["MAE"])
                mapes.append(m["MAPE"])
        mean_mae = float(np.mean(maes)) if maes else float("inf")
        mean_mape = float(np.mean(mapes)) if mapes else float("nan")
        rec = {
            "lookback_days": lb,
            "beta": beta,
            "n_days": len(maes),
            "valid_mean_MAE": mean_mae,
            "valid_mean_MAPE": mean_mape,
        }
        rows.append(rec)
        if best is None or mean_mae < best["valid_mean_MAE"]:
            best = rec
    if best is None:
        best = {
            "lookback_days": 2,
            "beta": 0.3,
            "n_days": 0,
            "valid_mean_MAE": float("nan"),
            "valid_mean_MAPE": float("nan"),
        }
    return best, pd.DataFrame(rows)


def _day_label(day: pd.Timestamp) -> str:
    day = pd.Timestamp(day).normalize()
    return f"{day.date()} ({CLUSTER_NAMES.get(int(day.dayofweek), '?')})"
