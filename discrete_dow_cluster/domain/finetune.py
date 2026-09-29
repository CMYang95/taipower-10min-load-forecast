"""Fine-tuning：同 weekday 原型重擬 / 每 DOW 獨立 XGB；分日 24h + valid 選 K。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from domain.cluster_assign import CLUSTER_NAMES
from domain.dow_xgb import apply_xgb_residual
from domain.slot_adapt import metrics
from domain.train_dow_xgb import fit_pack_at_cutoff, predict_horizon_proto


def _day_label(day: pd.Timestamp) -> str:
    day = pd.Timestamp(day).normalize()
    return f"{day.date()} ({CLUSTER_NAMES.get(int(day.dayofweek), '?')})"


def finetune_proto_separate_days(
    *,
    load10: pd.Series,
    holidays: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    k_same_dow: int,
    max_days_per_cluster: int = 16,
    quiet: bool = False,
) -> pd.Series:
    """分日預測：每日用最近 K 個同 weekday 日重擬中位數原型。"""
    pieces: list[pd.Series] = []
    for i, day in enumerate(target_dates):
        day = pd.Timestamp(day).normalize()
        if not quiet:
            print(
                f"  FT-proto D{i+1} {_day_label(day)} K={k_same_dow} train<{day.date()}",
                flush=True,
            )
        pack = fit_pack_at_cutoff(
            load10=load10,
            holidays=holidays,
            train_cutoff=day,
            max_days_per_cluster=max_days_per_cluster,
            n_same_dow_days=int(k_same_dow),
            quiet=quiet,
        )
        pred = predict_horizon_proto(pack["profiles"], pack["clusters"], day, horizon_days=1)
        pieces.append(pred)
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def finetune_xgb_separate_days(
    *,
    load10: pd.Series,
    holidays: pd.DataFrame,
    weather: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    k_same_dow: int,
    max_days_per_cluster: int = 16,
    xgb_train_days: int = 365,
    min_samples: int = 200,
    xgb_params: dict | None = None,
    weather_lag_days: int = 1,
    quiet: bool = False,
) -> pd.Series:
    """分日預測：每日用 K 同 weekday 原型 + 同 cluster XGB 殘差。"""
    from domain.dow_data_split import cluster_id_for_day
    from domain.dow_xgb import fit_xgb_for_cluster

    pieces: list[pd.Series] = []
    for i, day in enumerate(target_dates):
        day = pd.Timestamp(day).normalize()
        if not quiet:
            print(
                f"  FT-xgb D{i+1} {_day_label(day)} K={k_same_dow} train<{day.date()}",
                flush=True,
            )
        pack = fit_pack_at_cutoff(
            load10=load10,
            holidays=holidays,
            train_cutoff=day,
            max_days_per_cluster=max_days_per_cluster,
            n_same_dow_days=int(k_same_dow),
            weather=weather,
            quiet=quiet,
        )
        cid = cluster_id_for_day(day, pack["clusters"])
        # 只訓目標日所屬 cluster，避免 8 群全訓
        load_by = pack.get("load_by_cluster") or {}
        weather_by = pack.get("weather_by_cluster") or {}
        model = fit_xgb_for_cluster(
            cid,
            load_sub=load_by.get(cid, pd.Series(dtype=float)),
            weather_sub=weather_by.get(cid, pd.DataFrame()),
            profile=pack["profiles"].get(cid),
            load10=load10,
            clusters=pack["clusters"],
            train_cutoff=pack["train_cutoff"],
            k_same_dow=int(k_same_dow),
            xgb_train_days=xgb_train_days,
            min_samples=min_samples,
            xgb_params=xgb_params,
            weather_lag_days=weather_lag_days,
            quiet=quiet,
        )
        models = {c: None for c in range(8)}
        models[cid] = model
        base = predict_horizon_proto(pack["profiles"], pack["clusters"], day, horizon_days=1)
        pred = apply_xgb_residual(
            base,
            profiles=pack["profiles"],
            clusters=pack["clusters"],
            xgb_models=models,
            load10=load10,
            weather=weather if weather is not None else pd.DataFrame(),
            weather_lag_days=weather_lag_days,
        )
        pieces.append(pred)
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)


def select_finetune_on_valid(
    *,
    actual: pd.Series,
    load10: pd.Series,
    holidays: pd.DataFrame,
    weather: pd.DataFrame,
    valid_days: list[pd.Timestamp],
    ks: list[int],
    method: str = "proto",
    max_days_per_cluster: int = 16,
    xgb_train_days: int = 365,
    min_samples: int = 200,
    xgb_params: dict | None = None,
    weather_lag_days: int = 1,
    quiet: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """valid 期掃 K=1..7，按 mean MAE 選最優。method: proto | xgb。"""
    rows = []
    best = None
    for k in ks:
        maes = []
        mapes = []
        for day in valid_days:
            day = pd.Timestamp(day).normalize()
            if method == "xgb":
                pred = finetune_xgb_separate_days(
                    load10=load10,
                    holidays=holidays,
                    weather=weather,
                    target_dates=[day],
                    k_same_dow=k,
                    max_days_per_cluster=max_days_per_cluster,
                    xgb_train_days=xgb_train_days,
                    min_samples=min_samples,
                    xgb_params=xgb_params,
                    weather_lag_days=weather_lag_days,
                    quiet=quiet,
                )
            else:
                pred = finetune_proto_separate_days(
                    load10=load10,
                    holidays=holidays,
                    target_dates=[day],
                    k_same_dow=k,
                    max_days_per_cluster=max_days_per_cluster,
                    quiet=quiet,
                )
            if pred.empty:
                continue
            y = actual.reindex(pred.index)
            if y.notna().sum() < len(pred) // 2:
                continue
            m = metrics(y.to_numpy(), pred.to_numpy())
            if np.isfinite(m["MAE"]):
                maes.append(m["MAE"])
                mapes.append(m["MAPE"])
        mean_mae = float(np.mean(maes)) if maes else float("inf")
        mean_mape = float(np.mean(mapes)) if mapes else float("nan")
        rec = {
            "method": f"ft_{method}",
            "k_same_dow": k,
            "n_days": len(maes),
            "valid_mean_MAE": mean_mae,
            "valid_mean_MAPE": mean_mape,
        }
        rows.append(rec)
        if best is None or mean_mae < best["valid_mean_MAE"]:
            best = rec
        print(
            f"  valid FT-{method} K={k} mean_MAE={mean_mae:.1f} n={len(maes)}",
            flush=True,
        )
    if best is None:
        best = {
            "method": f"ft_{method}",
            "k_same_dow": 3,
            "n_days": 0,
            "valid_mean_MAE": float("nan"),
            "valid_mean_MAPE": float("nan"),
        }
    return best, pd.DataFrame(rows)
