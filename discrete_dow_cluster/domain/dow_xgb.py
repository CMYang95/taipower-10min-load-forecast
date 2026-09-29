"""每 DOW cluster 獨立 XGB 殘差：訓練集僅同 weekday/holiday，含天氣特徵。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

from domain.dow_data_split import CLUSTER_IDS, cluster_day_list, cluster_id_for_day

SLOTS_DAY = 144
FEATURE_COLS = [
    "slot",
    "slot_sin",
    "slot_cos",
    "base_hat",
    "T_mean",
    "humidity",
    "wind_speed",
    "sunshine",
    "load_lag_same_dow",  # 上一同群日同 slot
]


def _slot_of(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 6 + index.minute // 10).astype(int)


def _weather_at(weather: pd.DataFrame, ts: pd.Timestamp, lag_days: int = 1) -> dict[str, float]:
    """取 ts - lag_days 的天氣；缺則 0。"""
    cols = ["T_mean", "humidity", "wind_speed", "sunshine"]
    out = {c: 0.0 for c in cols}
    if weather is None or weather.empty:
        return out
    t = pd.Timestamp(ts) - pd.Timedelta(days=int(lag_days))
    if t in weather.index:
        row = weather.loc[t]
        for c in cols:
            if c in weather.columns:
                v = row[c] if not isinstance(row, pd.DataFrame) else row[c].iloc[0]
                out[c] = float(v) if pd.notna(v) else 0.0
        return out
    # 最近前值
    w = weather.loc[weather.index <= t]
    if w.empty:
        return out
    row = w.iloc[-1]
    for c in cols:
        if c in weather.columns:
            v = row[c]
            out[c] = float(v) if pd.notna(v) else 0.0
    return out


def _same_dow_lag_load(
    load10: pd.Series,
    clusters: pd.Series,
    ts: pd.Timestamp,
    cluster_id: int,
) -> float:
    """上一同 cluster 日、同 slot 的負載。"""
    day = pd.Timestamp(ts).normalize()
    days = cluster_day_list(clusters, cluster_id, before=day)
    if not days:
        return float("nan")
    prev = days[0]
    slot = int(ts.hour * 6 + ts.minute // 10)
    prev_ts = prev + pd.Timedelta(minutes=10 * slot)
    if prev_ts in load10.index:
        v = load10.loc[prev_ts]
        return float(v) if pd.notna(v) else float("nan")
    return float("nan")


def build_cluster_feature_matrix(
    index: pd.DatetimeIndex,
    *,
    base_hat: pd.Series,
    load10: pd.Series,
    clusters: pd.Series,
    weather: pd.DataFrame,
    cluster_id: int,
    weather_lag_days: int = 1,
) -> pd.DataFrame:
    """為給定時刻建特徵（僅用於該 cluster 的點）。"""
    rows = []
    slots = _slot_of(index)
    for i, ts in enumerate(index):
        bh = base_hat.reindex([ts]).iloc[0] if ts in base_hat.index else np.nan
        if not np.isfinite(bh):
            # try positional
            try:
                bh = float(base_hat.loc[ts])
            except Exception:
                continue
        if not np.isfinite(bh):
            continue
        w = _weather_at(weather, ts, weather_lag_days)
        lag = _same_dow_lag_load(load10, clusters, ts, cluster_id)
        if not np.isfinite(lag):
            lag = float(bh)
        s = int(slots[i])
        rows.append(
            {
                "ts": ts,
                "slot": float(s),
                "slot_sin": float(np.sin(2 * np.pi * s / SLOTS_DAY)),
                "slot_cos": float(np.cos(2 * np.pi * s / SLOTS_DAY)),
                "base_hat": float(bh),
                "T_mean": w["T_mean"],
                "humidity": w["humidity"],
                "wind_speed": w["wind_speed"],
                "sunshine": w["sunshine"],
                "load_lag_same_dow": lag,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["ts", *FEATURE_COLS])
    return pd.DataFrame(rows)


def _xgb_params(params: dict | None) -> dict:
    p = params or {}
    return dict(
        n_estimators=int(p.get("n_estimators", 200)),
        max_depth=int(p.get("max_depth", 4)),
        learning_rate=float(p.get("learning_rate", 0.05)),
        subsample=float(p.get("subsample", 0.8)),
        colsample_bytree=float(p.get("colsample_bytree", 0.8)),
        min_child_weight=float(p.get("min_child_weight", 8)),
        random_state=int(p.get("seed", 42)),
        n_jobs=-1,
        objective="reg:squarederror",
        tree_method="hist",
    )


def fit_xgb_for_cluster(
    cluster_id: int,
    *,
    load_sub: pd.Series,
    weather_sub: pd.DataFrame,
    profile: np.ndarray,
    load10: pd.Series,
    clusters: pd.Series,
    train_cutoff: pd.Timestamp,
    k_same_dow: int | None = None,
    xgb_train_days: int = 365,
    min_samples: int = 200,
    xgb_params: dict | None = None,
    weather_lag_days: int = 1,
    quiet: bool = False,
) -> Any | None:
    """在同 cluster 子序列上訓殘差 XGB；樣本不足返回 None。"""
    if xgb is None:
        if not quiet:
            print("  xgboost 未安裝，跳過 XGB fine-tune", flush=True)
        return None

    train_cutoff = pd.Timestamp(train_cutoff).normalize()
    days = cluster_day_list(clusters, cluster_id, before=train_cutoff)
    if k_same_dow is not None:
        use_days = set(days[: int(k_same_dow)])
    else:
        # 日曆窗內的同群日
        start = train_cutoff - pd.Timedelta(days=int(xgb_train_days))
        use_days = {d for d in days if d >= start}

    if not use_days:
        if not quiet:
            print(f"  XGB cluster={cluster_id} 無訓練日", flush=True)
        return None

    # 訓練 index：同群日全部 10min 點
    mask_days = pd.Series(
        [pd.Timestamp(t).normalize() in use_days for t in load_sub.index],
        index=load_sub.index,
    )
    train_load = load_sub.loc[mask_days]
    if len(train_load) < int(min_samples):
        if not quiet:
            print(
                f"  XGB cluster={cluster_id} n={len(train_load)} < {min_samples} → fallback proto",
                flush=True,
            )
        return None

    # base_hat = 原型按 slot 貼到訓練時刻
    slots = _slot_of(train_load.index)
    base_vals = np.asarray(profile, dtype=float)[slots]
    base_hat = pd.Series(base_vals, index=train_load.index, name="base_hat")
    residual = (train_load.astype(float) - base_hat).rename("resid")

    feat = build_cluster_feature_matrix(
        train_load.index,
        base_hat=base_hat,
        load10=load10,
        clusters=clusters,
        weather=weather_sub if weather_sub is not None else pd.DataFrame(),
        cluster_id=cluster_id,
        weather_lag_days=weather_lag_days,
    )
    if feat.empty:
        return None
    feat = feat.set_index("ts")
    y = residual.reindex(feat.index)
    x = feat[FEATURE_COLS]
    m = y.notna() & np.isfinite(y.to_numpy()) & x.notna().all(axis=1)
    if int(m.sum()) < int(min_samples):
        if not quiet:
            print(
                f"  XGB cluster={cluster_id} valid n={int(m.sum())} < {min_samples} → fallback",
                flush=True,
            )
        return None

    model = xgb.XGBRegressor(**_xgb_params(xgb_params))
    model.fit(x.loc[m], y.loc[m])
    if not quiet:
        print(f"  XGB cluster={cluster_id} n={int(m.sum())} days={len(use_days)}", flush=True)
    return model


def fit_all_cluster_xgb(
    *,
    pack: dict,
    load10: pd.Series,
    weather: pd.DataFrame,
    k_same_dow: int | None = None,
    xgb_train_days: int = 365,
    min_samples: int = 200,
    xgb_params: dict | None = None,
    weather_lag_days: int = 1,
    quiet: bool = False,
) -> dict[int, Any]:
    """對 pack 內各 cluster 訓 XGB；返回 {cid: model|None}。"""
    models: dict[int, Any] = {}
    load_by = pack.get("load_by_cluster") or {}
    weather_by = pack.get("weather_by_cluster") or {}
    profiles = pack["profiles"]
    clusters = pack["clusters"]
    cutoff = pack["train_cutoff"]
    for cid in CLUSTER_IDS:
        models[cid] = fit_xgb_for_cluster(
            cid,
            load_sub=load_by.get(cid, pd.Series(dtype=float)),
            weather_sub=weather_by.get(cid, pd.DataFrame()),
            profile=profiles.get(cid, np.zeros(SLOTS_DAY)),
            load10=load10,
            clusters=clusters,
            train_cutoff=cutoff,
            k_same_dow=k_same_dow,
            xgb_train_days=xgb_train_days,
            min_samples=min_samples,
            xgb_params=xgb_params,
            weather_lag_days=weather_lag_days,
            quiet=quiet,
        )
    return models


def apply_xgb_residual(
    base: pd.Series,
    *,
    profiles: dict[int, np.ndarray],
    clusters: pd.Series,
    xgb_models: dict[int, Any],
    load10: pd.Series,
    weather: pd.DataFrame,
    weather_lag_days: int = 1,
) -> pd.Series:
    """對 base（原型預測）逐日用對應 cluster 的 XGB 加殘差。"""
    if not xgb_models or all(m is None for m in xgb_models.values()):
        return base.copy()

    out = base.astype(float).copy()
    # 按日分組
    days = pd.DatetimeIndex(base.index.normalize()).unique()
    for day in days:
        day = pd.Timestamp(day).normalize()
        mask = (base.index >= day) & (base.index < day + pd.Timedelta(days=1))
        part = base.loc[mask]
        if part.empty:
            continue
        cid = cluster_id_for_day(day, clusters)
        model = xgb_models.get(cid)
        if model is None:
            continue
        feat = build_cluster_feature_matrix(
            part.index,
            base_hat=part,
            load10=load10,
            clusters=clusters,
            weather=weather if weather is not None else pd.DataFrame(),
            cluster_id=cid,
            weather_lag_days=weather_lag_days,
        )
        if feat.empty:
            continue
        feat = feat.set_index("ts")
        x = feat[FEATURE_COLS]
        pred_resid = model.predict(x)
        out.loc[feat.index] = part.reindex(feat.index).to_numpy(dtype=float) + pred_resid
    out.name = "load_10"
    return out
