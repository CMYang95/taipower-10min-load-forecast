"""負載與天氣按 cluster（Mon–Sun + Holiday）嚴格分桶；同 weekday 日檢索。"""

from __future__ import annotations

import pandas as pd

from domain.cluster_assign import CLUSTER_NAMES, HOLIDAY_CLUSTER, assign_cluster

CLUSTER_IDS = list(range(8))


def split_series_by_cluster(load10: pd.Series, clusters: pd.Series) -> dict[int, pd.Series]:
    """每個 cluster → 該群 10min 負載序列（index 保留原時間戳）。"""
    cl = clusters.reindex(load10.index)
    out: dict[int, pd.Series] = {}
    for cid in range(8):
        mask = (cl == cid).fillna(False)
        out[cid] = load10.loc[mask].astype(float).rename(f"load_c{cid}")
    return out


def split_weather_by_cluster(weather: pd.DataFrame, clusters: pd.Series) -> dict[int, pd.DataFrame]:
    """每個 cluster → 該群天氣特徵子表。"""
    if weather is None or weather.empty:
        return {cid: pd.DataFrame() for cid in range(8)}
    cl = clusters.reindex(weather.index)
    out: dict[int, pd.DataFrame] = {}
    for cid in range(8):
        mask = (cl == cid).fillna(False)
        out[cid] = weather.loc[mask].copy()
    return out


def cluster_day_list(
    clusters: pd.Series,
    cluster_id: int,
    *,
    before: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """該 cluster 出現過的日曆日（降序：最近在前）。"""
    cl = clusters
    if before is not None:
        before = pd.Timestamp(before).normalize()
        cl = cl.loc[cl.index < before]
    mask = (cl == int(cluster_id)).fillna(False)
    if not mask.any():
        return []
    days = pd.DatetimeIndex(cl.index[mask].normalize()).unique()
    return sorted(days, reverse=True)


def same_cluster_days_before(
    origin: pd.Timestamp,
    cluster_id: int,
    clusters: pd.Series,
    k: int,
) -> list[pd.Timestamp]:
    """origin 之前最近 k 個同 cluster 出現日（不含 origin）。"""
    days = cluster_day_list(clusters, cluster_id, before=pd.Timestamp(origin).normalize())
    return list(days[: int(k)])


def same_cluster_days_before_from_holidays(
    origin: pd.Timestamp,
    cluster_id: int,
    holidays: pd.DataFrame,
    load_index: pd.DatetimeIndex,
    k: int,
    *,
    holiday_id: int = HOLIDAY_CLUSTER,
) -> list[pd.Timestamp]:
    """用 holidays + load_index 現算 clusters 再取同群日。"""
    clusters = assign_cluster(load_index, holidays, holiday_id=holiday_id)
    return same_cluster_days_before(origin, cluster_id, clusters, k)


def calendar_lookback_days(
    origin: pd.Timestamp,
    n: int,
    idx_days: set | None = None,
) -> list[pd.Timestamp]:
    """日曆 D-1 .. D-n（可選過濾有資料的日）。"""
    origin = pd.Timestamp(origin).normalize()
    out: list[pd.Timestamp] = []
    for i in range(1, int(n) + 1):
        day = origin - pd.Timedelta(days=i)
        if idx_days is not None and day.normalize() not in idx_days:
            continue
        out.append(day)
    return out


def cluster_id_for_day(day: pd.Timestamp, clusters: pd.Series) -> int:
    """取某日的 cluster_id（優先 noon 點）。"""
    day = pd.Timestamp(day).normalize()
    noon = day + pd.Timedelta(hours=12)
    if noon in clusters.index:
        return int(clusters.loc[noon])
    day_cl = clusters.loc[(clusters.index >= day) & (clusters.index < day + pd.Timedelta(days=1))]
    if len(day_cl):
        return int(day_cl.iloc[0])
    return int(day.dayofweek)


def buckets_summary(
    load_by_cluster: dict[int, pd.Series],
    weather_by_cluster: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """各 bucket 樣本數匯總（驗證分桶）。"""
    rows = []
    for cid in range(8):
        load_sub = load_by_cluster.get(cid, pd.Series(dtype=float))
        n_pts = int(len(load_sub))
        n_days = 0
        if n_pts:
            n_days = int(pd.DatetimeIndex(load_sub.index.normalize()).nunique())
        w_pts = 0
        if weather_by_cluster is not None:
            w = weather_by_cluster.get(cid)
            if w is not None and not w.empty:
                w_pts = int(len(w))
        rows.append(
            {
                "cluster_id": cid,
                "name": CLUSTER_NAMES.get(cid, str(cid)),
                "n_load_points": n_pts,
                "n_days": n_days,
                "n_weather_points": w_pts,
            }
        )
    return pd.DataFrame(rows)
