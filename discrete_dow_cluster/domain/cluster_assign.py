"""完全硬分群：星期幾 0–6 + 國定假日 7（互斥、每點只屬一群）。"""

from __future__ import annotations

import pandas as pd

CLUSTER_NAMES = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
    7: "Holiday",
}

HOLIDAY_CLUSTER = 7


def assign_cluster(
    index: pd.DatetimeIndex,
    holidays: pd.DataFrame,
    *,
    holiday_id: int = HOLIDAY_CLUSTER,
) -> pd.Series:
    """回傳 int Series：假日→holiday_id，否則 dayofweek；補班日不當假日。"""
    dates = pd.DatetimeIndex(pd.DatetimeIndex(index).normalize())
    hol: set = set()
    if holidays is not None and not holidays.empty:
        h = holidays.copy()
        h["Date"] = pd.to_datetime(h["Date"]).dt.normalize()
        if "kind" in h.columns:
            hol = set(h.loc[h["kind"].astype(str).str.lower() == "holiday", "Date"])
        else:
            hol = set(h["Date"])
    is_hol = pd.Series([d in hol for d in dates], index=index)
    dow = pd.Series(pd.DatetimeIndex(index).dayofweek, index=index, dtype=int)
    out = dow.where(~is_hol, other=int(holiday_id)).astype(int)
    out.name = "cluster_id"
    return out


def cluster_counts(clusters: pd.Series) -> pd.DataFrame:
    vc = clusters.value_counts().sort_index()
    rows = []
    for cid, n in vc.items():
        rows.append(
            {
                "cluster_id": int(cid),
                "name": CLUSTER_NAMES.get(int(cid), str(cid)),
                "n_points": int(n),
                "n_days_approx": round(int(n) / 144, 1),
            }
        )
    return pd.DataFrame(rows)
