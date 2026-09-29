"""Fully Discrete：星期／假日硬分群 + 各群 slot 中位數原型（無 S1/XGB 連續殘差）。

預測日屬哪一群 → 整日套該群歷史中位數曲線（144 點）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from domain.cluster_assign import HOLIDAY_CLUSTER, assign_cluster, cluster_counts
from infra.data_io import (
    FREQ,
    SLOTS_DAY,
    align_to_10min,
    coverage_report,
    existing_path,
    load_10min_load,
    load_daily_temp_10min,
    load_holidays,
    load_odwo_10min,
    load_solar_10min,
    load_station_weather_10min,
    merge_weather,
)
from domain.dow_data_split import (
    buckets_summary,
    cluster_id_for_day,
    split_series_by_cluster,
    split_weather_by_cluster,
)

# 套件根（settings.yaml / data / outputs）
ROOT = Path(__file__).resolve().parent.parent

CLUSTER_IDS = list(range(8))


def load_settings(path: Path | None = None) -> dict:
    p = path or (ROOT / "settings.yaml")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict:
    raw = dict(cfg.get("paths") or {})
    out = {}
    for k, v in raw.items():
        if v is None or v == "":
            out[k] = None
            continue
        path = Path(str(v))
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        out[k] = path
    out["output_dir"] = Path(out.get("output_dir") or (ROOT / "outputs")).resolve()
    return out


def _asfreq(s: pd.Series) -> pd.Series:
    s = s.sort_index()
    return s[~s.index.duplicated(keep="last")].asfreq(FREQ)


def _fill(s: pd.Series) -> pd.Series:
    return s.interpolate(method="time", limit=24).ffill().bfill()


def _slot_of_index(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 6 + index.minute // 10).astype(int)


def fit_median_profiles(
    load: pd.Series,
    clusters: pd.Series,
    train_cutoff: pd.Timestamp,
    *,
    max_days_per_cluster: int = 16,
    n_same_dow_days: int | None = None,
    quiet: bool = False,
) -> dict[int, np.ndarray]:
    """每個 cluster → shape (144,) slot 中位數。

    只用 cutoff 前最近 N 個該群日（同 weekday / holiday），絕不跨群混用。
    n_same_dow_days 若給定（fine-tune），覆寫 max_days_per_cluster。
    """
    train_cutoff = pd.Timestamp(train_cutoff).normalize()
    n_days = int(n_same_dow_days) if n_same_dow_days is not None else int(max_days_per_cluster)
    hist = load.loc[load.index < train_cutoff].astype(float)
    cl = clusters.reindex(hist.index)
    profiles: dict[int, np.ndarray] = {}

    recent = hist.loc[hist.index >= train_cutoff - pd.Timedelta(days=28)]
    global_med = np.full(SLOTS_DAY, np.nan, dtype=float)
    if len(recent):
        slots_g = _slot_of_index(recent.index)
        for s in range(SLOTS_DAY):
            vals = recent.to_numpy()[slots_g == s]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                global_med[s] = float(np.median(vals))
    global_med = pd.Series(global_med).ffill().bfill().fillna(0.0).to_numpy(dtype=float)

    for cid in CLUSTER_IDS:
        mask = (cl == cid).fillna(False)
        sub = hist.loc[mask]
        if sub.empty:
            profiles[cid] = global_med.copy()
            if not quiet:
                print(f"  proto cluster={cid} empty → recent global median", flush=True)
            continue
        days = sorted(pd.DatetimeIndex(sub.index.normalize()).unique(), reverse=True)
        use_days = set(days[:n_days])
        day_norm = sub.index.normalize()
        sub = sub.loc[[d in use_days for d in day_norm]]
        slots = _slot_of_index(sub.index)
        prof = np.full(SLOTS_DAY, np.nan, dtype=float)
        for s in range(SLOTS_DAY):
            vals = sub.to_numpy()[slots == s]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                prof[s] = float(np.median(vals))
        miss = ~np.isfinite(prof)
        if miss.any():
            prof[miss] = global_med[miss]
        profiles[cid] = prof.astype(float)
        if not quiet:
            print(
                f"  proto cluster={cid} last {len(use_days)} same-dow days median "
                f"(level~{float(np.nanmedian(prof)):.0f} MW)",
                flush=True,
            )
    return profiles


def predict_horizon_proto(
    profiles: dict[int, np.ndarray],
    clusters: pd.Series,
    origin: pd.Timestamp,
    horizon_days: int,
) -> pd.Series:
    """依每日 cluster 貼中位數原型（一天一型，非連續殘差模型）。"""
    origin = pd.Timestamp(origin).normalize()
    pieces = []
    for d in range(horizon_days):
        day = origin + pd.Timedelta(days=d)
        idx = pd.date_range(day, periods=SLOTS_DAY, freq=FREQ)
        cid = cluster_id_for_day(day, clusters)
        prof = profiles.get(cid, profiles.get(0))
        pieces.append(pd.Series(prof, index=idx, name="load_10"))
    return pd.concat(pieces)


def predict_horizon_proto_xgb(
    profiles: dict[int, np.ndarray],
    clusters: pd.Series,
    origin: pd.Timestamp,
    horizon_days: int,
    *,
    xgb_models: dict,
    load10: pd.Series,
    weather: pd.DataFrame,
    weather_lag_days: int = 1,
) -> pd.Series:
    """原型 + 同 DOW XGB 殘差校正。"""
    from domain.dow_xgb import apply_xgb_residual

    base = predict_horizon_proto(profiles, clusters, origin, horizon_days)
    return apply_xgb_residual(
        base,
        profiles=profiles,
        clusters=clusters,
        xgb_models=xgb_models,
        load10=load10,
        weather=weather,
        weather_lag_days=weather_lag_days,
    )


def fit_pack_at_cutoff(
    *,
    load10: pd.Series,
    holidays: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    holiday_id: int = HOLIDAY_CLUSTER,
    max_days_per_cluster: int = 16,
    n_same_dow_days: int | None = None,
    weather: pd.DataFrame | None = None,
    quiet: bool = False,
) -> dict:
    train_cutoff = pd.Timestamp(train_cutoff).normalize()
    clusters = assign_cluster(load10.index, holidays, holiday_id=holiday_id)
    profiles = fit_median_profiles(
        load10,
        clusters,
        train_cutoff,
        max_days_per_cluster=max_days_per_cluster,
        n_same_dow_days=n_same_dow_days,
        quiet=quiet,
    )
    counts = cluster_counts(clusters.loc[clusters.index < train_cutoff])
    hist_mask = load10.index < train_cutoff
    load_hist = load10.loc[hist_mask]
    cl_hist = clusters.loc[hist_mask]
    load_by_cluster = split_series_by_cluster(load_hist, cl_hist)
    weather_by_cluster: dict[int, pd.DataFrame] = {cid: pd.DataFrame() for cid in CLUSTER_IDS}
    if weather is not None and not weather.empty:
        w_hist = weather.loc[weather.index < train_cutoff]
        weather_by_cluster = split_weather_by_cluster(w_hist, clusters.reindex(w_hist.index))
    return {
        "train_cutoff": train_cutoff,
        "clusters": clusters,
        "profiles": profiles,
        "cluster_counts": counts,
        "hist_base": _fill(load10),
        "max_days_per_cluster": max_days_per_cluster,
        "n_same_dow_days": n_same_dow_days,
        "load_by_cluster": load_by_cluster,
        "weather_by_cluster": weather_by_cluster,
        "buckets_summary": buckets_summary(load_by_cluster, weather_by_cluster),
    }


def roll_proto_curves(
    pack: dict,
    origins: list[pd.Timestamp],
    *,
    horizon_days: int,
    valid_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    print(f"  滾動原點 {len(origins)}（median proto）", flush=True)
    for i, origin in enumerate(origins):
        if i % 10 == 0:
            print(f"    {i+1}/{len(origins)} {pd.Timestamp(origin).date()}", flush=True)
        pred = predict_horizon_proto(pack["profiles"], pack["clusters"], origin, horizon_days)
        day0 = pd.Timestamp(origin).normalize()
        for h in range(horizon_days):
            day = day0 + pd.Timedelta(days=h)
            part = pred.loc[(pred.index >= day) & (pred.index < day + pd.Timedelta(days=1))]
            if part.empty:
                continue
            if day < valid_start:
                split = "pre_valid"
            elif day < test_start:
                split = "valid"
            else:
                split = "test"
            cid = cluster_id_for_day(day, pack["clusters"])
            tmp = part.rename("load_10").reset_index()
            tmp.columns = ["date_time", "load_10"]
            tmp["origin"] = day0
            tmp["horizon"] = h + 1
            tmp["split"] = split
            tmp["cluster_id"] = cid
            rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_common_data(cfg: dict, paths: dict) -> dict:
    load_path = paths.get("load_csv")
    if load_path is None or not Path(load_path).exists():
        raise FileNotFoundError(
            f"找不到負載 CSV：{load_path}（請在 settings.yaml 的 paths.load_csv 指定正確路徑）"
        )
    load_csv = Path(load_path)
    print(f"load={load_csv}", flush=True)
    load10 = _asfreq(align_to_10min(load_10min_load(load_csv), "load"))
    weather = merge_weather(
        load10.index,
        load_odwo_10min(existing_path(paths.get("odwo0011_csv"))),
        load_station_weather_10min(existing_path(paths.get("weather_stations_dir"))),
        load_solar_10min(existing_path(paths.get("energy_csv"))),
        load_daily_temp_10min(existing_path(paths.get("weather_daily_csv")), load10.index),
    )
    holidays = load_holidays(existing_path(paths.get("holidays_csv")))
    if not weather.empty:
        print("weather coverage:", flush=True)
        for k, v in coverage_report(weather).items():
            print(f"  {k}: {v}", flush=True)
    else:
        print("weather coverage: 無可用天氣資料", flush=True)
    return {
        "load10": load10,
        "weather": weather,
        "holidays": holidays,
        "actual": load10.astype(float),
    }
