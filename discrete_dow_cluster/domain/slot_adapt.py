"""近日 slot 偏差適配 + 雙向偏差校正（bias_clip）。

- slot：同 cluster 日回看（預設），或日曆 D-1..D-N
- bias_clip：g=加權 mean(實際−pre)；可選 peak tip 混合；週末可禁下修
  p=clip(g,-C,C)，最終=pre+p（函數名 underest_* 相容舊碼）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SLOTS_DAY = 144
# 與競賽尖峰窗一致：日尖峰 11:00–17:00、夜尖峰 17:10–21:00
PEAK_DAY_SLOTS = frozenset(range(11 * 6, 17 * 6))
PEAK_NIGHT_SLOTS = frozenset(range(17 * 6 + 1, 21 * 6 + 1))
DEFAULT_PEAK_SLOTS = PEAK_DAY_SLOTS | PEAK_NIGHT_SLOTS


def _slot_of(ts: pd.DatetimeIndex | pd.Timestamp) -> np.ndarray | int:
    if isinstance(ts, pd.Timestamp):
        return int(ts.hour * 6 + ts.minute // 10)
    return (ts.hour * 6 + ts.minute // 10).astype(int)


def h1_day_curve(blend: pd.DataFrame, day: pd.Timestamp) -> pd.Series:
    day = pd.Timestamp(day).normalize()
    part = blend[(blend["origin"] == day) & (blend["date_time"].dt.normalize() == day)]
    if part.empty:
        return pd.Series(dtype=float)
    return part.sort_values("date_time").set_index("date_time")["load_10"].astype(float)


def origin_curve(blend: pd.DataFrame, origin: pd.Timestamp, n_days: int) -> pd.Series:
    origin = pd.Timestamp(origin).normalize()
    end = origin + pd.Timedelta(days=n_days)
    part = blend[
        (blend["origin"] == origin) & (blend["date_time"] >= origin) & (blend["date_time"] < end)
    ]
    return part.sort_values("date_time").set_index("date_time")["load_10"].astype(float)


def estimate_slot_gap(
    actual: pd.Series,
    blend: pd.DataFrame,
    origin: pd.Timestamp,
    lookback_days: int,
    *,
    same_cluster_days: list[pd.Timestamp] | None = None,
) -> tuple[np.ndarray, int]:
    """回傳 shape=(144,) 的 mean(actual-pred)，以及實際用到的天數。

    same_cluster_days 若提供 → 只在這些同 cluster 日上算 gap（忽略日曆 lookback 混合）。
    否則 fallback：日曆 D-1 .. D-lookback_days。
    """
    origin = pd.Timestamp(origin).normalize()
    if same_cluster_days is not None:
        days = [pd.Timestamp(d).normalize() for d in same_cluster_days]
    else:
        days = [origin - pd.Timedelta(days=i) for i in range(1, int(lookback_days) + 1)]

    gaps: list[np.ndarray] = []
    used = 0
    for day in days:
        pred = h1_day_curve(blend, day)
        if pred.empty or len(pred) < SLOTS_DAY // 2:
            continue
        act = actual.reindex(pred.index)
        if act.notna().sum() < SLOTS_DAY // 2:
            continue
        slot = _slot_of(pred.index)
        g = np.full(SLOTS_DAY, np.nan, dtype=float)
        diff = (act - pred).to_numpy(dtype=float)
        for s in range(SLOTS_DAY):
            vals = diff[slot == s]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                g[s] = float(np.mean(vals))
        gaps.append(g)
        used += 1
    if not gaps:
        return np.zeros(SLOTS_DAY, dtype=float), 0
    return np.nanmean(np.vstack(gaps), axis=0), used


def apply_slot_adapt(base: pd.Series, gap: np.ndarray, beta: float) -> pd.Series:
    slots = _slot_of(base.index)
    add = np.asarray(gap, dtype=float)[slots]
    add = np.where(np.isfinite(add), add, 0.0)
    return pd.Series(base.to_numpy(dtype=float) + float(beta) * add, index=base.index, name="adapted")


def day_slice(series: pd.Series, day: pd.Timestamp) -> pd.Series:
    day = pd.Timestamp(day).normalize()
    return series.loc[(series.index >= day) & (series.index < day + pd.Timedelta(days=1))].astype(float)


def bias_ref_days(
    day: pd.Timestamp,
    lookback_days: int = 2,
    *,
    include_prev_same_dow: bool = True,
    idx_days: set | None = None,
    holiday_cluster_id: int | None = None,
    clusters: pd.Series | None = None,
    only_same_cluster: bool = False,
) -> list[pd.Timestamp]:
    """bias_clip 回看日：日曆 D-1..D-N ∪ 可選上週同 weekday；假日可限同群。"""
    day = pd.Timestamp(day).normalize()
    refs: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    target_cid = None
    if only_same_cluster and clusters is not None and len(clusters):
        if day in clusters.index:
            target_cid = int(clusters.loc[day])
        elif day.normalize() in clusters.index:
            target_cid = int(clusters.loc[day.normalize()])

    def _add(d: pd.Timestamp) -> None:
        d = pd.Timestamp(d).normalize()
        if d in seen or d >= day:
            return
        if idx_days is not None and d not in idx_days:
            return
        if target_cid is not None and clusters is not None:
            if d not in clusters.index and d.normalize() not in clusters.index:
                return
            cid = int(clusters.loc[d] if d in clusters.index else clusters.loc[d.normalize()])
            if cid != target_cid:
                return
        seen.add(d)
        refs.append(d)

    for i in range(1, int(lookback_days) + 1):
        _add(day - pd.Timedelta(days=i))
    if include_prev_same_dow:
        _add(day - pd.Timedelta(days=7))
    # 假日：若日曆鄰近日被濾掉，至少保留 D-7 同假日（上面已加）；再補最近同 cluster 日
    if (
        only_same_cluster
        and target_cid is not None
        and holiday_cluster_id is not None
        and target_cid == int(holiday_cluster_id)
        and clusters is not None
        and len(refs) < 1
    ):
        for past in sorted(
            (pd.Timestamp(x).normalize() for x in clusters.index if int(clusters.loc[x]) == target_cid),
            reverse=True,
        ):
            if past < day:
                _add(past)
            if len(refs) >= max(1, int(lookback_days)):
                break
    return refs


def ref_day_weight(
    day: pd.Timestamp,
    ref: pd.Timestamp,
    *,
    same_dow_weight: float = 2.0,
    calendar_weight: float = 1.0,
) -> float:
    """同 weekday（含 D-7）權重大；日曆鄰近日權重較小。"""
    day = pd.Timestamp(day).normalize()
    ref = pd.Timestamp(ref).normalize()
    if int(ref.dayofweek) == int(day.dayofweek):
        return float(same_dow_weight)
    return float(calendar_weight)


def collect_bias_pre_extra(
    origin: pd.Timestamp,
    target_dates: list[pd.Timestamp],
    lookback_days: int = 2,
    *,
    include_prev_same_dow: bool = True,
    idx_days: set | None = None,
    forecast_mode: str = "rolling",
) -> list[pd.Timestamp]:
    """目標窗外、為估 g 需先預測的日子（日曆前置 ∪ 各目標日上週同 weekday）。"""
    origin = pd.Timestamp(origin).normalize()
    targets = {pd.Timestamp(d).normalize() for d in target_dates}
    extra: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    mode = str(forecast_mode or "rolling").lower()

    def _add(d: pd.Timestamp) -> None:
        d = pd.Timestamp(d).normalize()
        if d in targets or d in seen:
            return
        if idx_days is not None and d not in idx_days:
            return
        # rolling：前置日須 < origin（交捲起點前已發生）；batch_oracle 允許窗外歷史即可
        if mode == "rolling" and d >= origin:
            return
        seen.add(d)
        extra.append(d)

    for i in range(int(lookback_days), 0, -1):
        _add(origin - pd.Timedelta(days=i))
    if include_prev_same_dow:
        for d in target_dates:
            _add(pd.Timestamp(d).normalize() - pd.Timedelta(days=7))
    # rolling 下窗內第 2+ 天的 D-1 可能是目標窗內前一天：不需進 pre_extra（會在 pre_dates 含目標日）
    return sorted(extra)


def estimate_underest_penalty(
    actual: pd.Series,
    pred_pre: pd.Series,
    day: pd.Timestamp,
    lookback_days: int = 2,
    *,
    cap_mw: float = 2400.0,
    idx_days: set | None = None,
    include_prev_same_dow: bool = True,
    same_dow_weight: float = 2.0,
    calendar_weight: float = 1.0,
    forecast_mode: str = "rolling",
    holiday_cluster_id: int = 7,
    clusters: pd.Series | None = None,
    gap_mode: str = "peak",
    peak_blend: float = 0.65,
    peak_slots: frozenset[int] | set[int] | None = None,
    weekend_uplift_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """雙向偏差校正（bias_clip）：refs 加權平均(actual−pre)，p=clip(g,-C,C)。

    權重：同 weekday（含 D-7）= same_dow_weight；其餘日曆鄰近 = calendar_weight。
    gap_mode=peak：尖峰 slot 混 tip_bias。
    weekend_uplift_only：週六／日僅允許 p≥0（避免把平日「高估下修」訊號誤套到週末）。
    """
    day = pd.Timestamp(day).normalize()
    cap = abs(float(cap_mw))
    only_same = False
    if clusters is not None and len(clusters):
        cid = None
        if day in clusters.index:
            cid = int(clusters.loc[day])
        elif day.normalize() in clusters.index:
            cid = int(clusters.loc[day.normalize()])
        only_same = cid is not None and cid == int(holiday_cluster_id)

    ref_days = bias_ref_days(
        day,
        lookback_days,
        include_prev_same_dow=include_prev_same_dow,
        idx_days=idx_days,
        holiday_cluster_id=holiday_cluster_id,
        clusters=clusters,
        only_same_cluster=only_same,
    )
    mode = str(forecast_mode or "rolling").lower()
    if mode == "rolling":
        ref_days = [r for r in ref_days if r < day]

    gaps: list[np.ndarray] = []
    weights: list[float] = []
    used = 0
    for ref in ref_days:
        pred = day_slice(pred_pre, ref)
        if pred.empty or len(pred) < SLOTS_DAY // 2:
            continue
        act = actual.reindex(pred.index)
        if act.notna().sum() < SLOTS_DAY // 2:
            continue
        slot = _slot_of(pred.index)
        g = np.full(SLOTS_DAY, np.nan, dtype=float)
        diff = (act.to_numpy(dtype=float) - pred.to_numpy(dtype=float))
        for s in range(SLOTS_DAY):
            vals = diff[slot == s]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                g[s] = float(np.mean(vals))
        gaps.append(g)
        weights.append(
            ref_day_weight(
                day,
                ref,
                same_dow_weight=same_dow_weight,
                calendar_weight=calendar_weight,
            )
        )
        used += 1

    if not gaps:
        z = np.zeros(SLOTS_DAY, dtype=float)
        return z, z, 0

    w = np.asarray(weights, dtype=float)
    stack = np.vstack(gaps)
    g_mean = np.zeros(SLOTS_DAY, dtype=float)
    for s in range(SLOTS_DAY):
        col = stack[:, s]
        m = np.isfinite(col)
        if not m.any():
            g_mean[s] = 0.0
            continue
        ww = w[m]
        g_mean[s] = float(np.sum(col[m] * ww) / np.sum(ww))

    if str(gap_mode or "mean").lower() == "peak":
        tip_slots = set(peak_slots) if peak_slots is not None else set(DEFAULT_PEAK_SLOTS)
        tip_vals: list[float] = []
        tip_w: list[float] = []
        for i, g in enumerate(gaps):
            tip = g[list(tip_slots)]
            tip = tip[np.isfinite(tip)]
            if len(tip):
                tip_vals.append(float(np.mean(tip)))
                tip_w.append(float(weights[i]))
        if tip_vals:
            tip_bias = float(np.average(tip_vals, weights=tip_w))
            blend = float(np.clip(peak_blend, 0.0, 1.0))
            for s in tip_slots:
                g_mean[s] = (1.0 - blend) * g_mean[s] + blend * tip_bias

    p = np.minimum(np.maximum(g_mean, -cap), cap)
    # 週末：平日高估 → 負 g；套到週六會把已校準曲線再壓下去 → 只留上修
    if weekend_uplift_only and int(day.dayofweek) >= 5:
        p = np.maximum(p, 0.0)
    return p.astype(float), g_mean.astype(float), used


def apply_underest_penalty_separate_days(
    actual: pd.Series,
    pred_pre: pd.Series,
    target_dates: list[pd.Timestamp],
    *,
    lookback_days: int = 2,
    cap_mw: float = 2400.0,
    idx_days: set | None = None,
    include_prev_same_dow: bool = True,
    same_dow_weight: float = 2.0,
    calendar_weight: float = 1.0,
    forecast_mode: str = "rolling",
    holiday_cluster_id: int = 7,
    clusters: pd.Series | None = None,
    gap_mode: str = "peak",
    peak_blend: float = 0.65,
    peak_slots: frozenset[int] | set[int] | None = None,
    weekend_uplift_only: bool = True,
) -> tuple[pd.Series, dict[pd.Timestamp, dict]]:
    """對評測窗每日套用雙向偏差校正（bias_clip）；pred_pre 需含 ref 日的 pre。"""
    pieces: list[pd.Series] = []
    meta: dict[pd.Timestamp, dict] = {}
    for day in target_dates:
        day = pd.Timestamp(day).normalize()
        pre = day_slice(pred_pre, day)
        if pre.empty:
            continue
        only_same = False
        if clusters is not None and len(clusters):
            cid = None
            if day in clusters.index:
                cid = int(clusters.loc[day])
            elif day.normalize() in clusters.index:
                cid = int(clusters.loc[day.normalize()])
            only_same = cid is not None and cid == int(holiday_cluster_id)
        refs = bias_ref_days(
            day,
            lookback_days,
            include_prev_same_dow=include_prev_same_dow,
            idx_days=idx_days,
            holiday_cluster_id=holiday_cluster_id,
            clusters=clusters,
            only_same_cluster=only_same,
        )
        if str(forecast_mode or "rolling").lower() == "rolling":
            refs = [r for r in refs if r < day]
        p, g, used = estimate_underest_penalty(
            actual,
            pred_pre,
            day,
            lookback_days,
            cap_mw=cap_mw,
            idx_days=idx_days,
            include_prev_same_dow=include_prev_same_dow,
            same_dow_weight=same_dow_weight,
            calendar_weight=calendar_weight,
            forecast_mode=forecast_mode,
            holiday_cluster_id=holiday_cluster_id,
            clusters=clusters,
            gap_mode=gap_mode,
            peak_blend=peak_blend,
            peak_slots=peak_slots,
            weekend_uplift_only=weekend_uplift_only,
        )
        meta[day] = {
            "g": g,
            "p": p,
            "used": used,
            "gap_mode": str(gap_mode),
            "weekend_uplift_only": bool(weekend_uplift_only and int(day.dayofweek) >= 5),
            "ref_days": [str(r.date()) for r in refs],
            "ref_weights": {
                str(r.date()): ref_day_weight(
                    day, r, same_dow_weight=same_dow_weight, calendar_weight=calendar_weight
                )
                for r in refs
            },
        }
        pieces.append(apply_slot_adapt(pre, p, 1.0).rename(pre.name or "load_10"))
    out = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return out, meta


def apply_level_scale_same_dow(
    actual: pd.Series,
    pred: pd.Series,
    target_dates: list[pd.Timestamp],
    *,
    k_same_dow: int = 2,
    clip_lo: float = 0.85,
    clip_hi: float = 1.15,
    idx_days: set | None = None,
) -> tuple[pd.Series, dict[pd.Timestamp, dict]]:
    """同 weekday 近期實際水準 / 當日 pre 水準 → 縮放 pre（再交給 bias_clip）。"""
    pieces: list[pd.Series] = []
    meta: dict[pd.Timestamp, dict] = {}
    for day in target_dates:
        day = pd.Timestamp(day).normalize()
        pre = day_slice(pred, day)
        if pre.empty:
            continue
        hist: list[pd.Timestamp] = []
        d = day - pd.Timedelta(days=1)
        while len(hist) < int(k_same_dow) and d >= (actual.index.min().normalize() if len(actual) else d):
            if int(d.dayofweek) == int(day.dayofweek):
                if idx_days is None or d in idx_days:
                    act_h = day_slice(actual, d)
                    if act_h.notna().sum() >= SLOTS_DAY // 2:
                        hist.append(d)
            d -= pd.Timedelta(days=1)
        pre_mean = float(np.nanmean(pre.to_numpy(dtype=float)))
        if not hist or not np.isfinite(pre_mean) or abs(pre_mean) < 1.0:
            scale = 1.0
        else:
            act_means = [float(np.nanmean(day_slice(actual, h).to_numpy(dtype=float))) for h in hist]
            act_mean = float(np.mean(act_means))
            scale = act_mean / pre_mean if abs(pre_mean) > 1.0 else 1.0
            scale = float(np.clip(scale, clip_lo, clip_hi))
        scaled = pd.Series(pre.to_numpy(dtype=float) * scale, index=pre.index, name=pre.name or "load_10")
        meta[day] = {
            "scale": scale,
            "hist_days": [str(h.date()) for h in hist],
            "pre_mean": pre_mean,
        }
        pieces.append(scaled)
    # 保留 pred 中非目標日（供 bias refs 用的 pre）
    out = pred.copy().astype(float)
    if pieces:
        adj = pd.concat(pieces)
        out.loc[adj.index] = adj.to_numpy(dtype=float)
    return out, meta


def metrics(a: np.ndarray, p: np.ndarray) -> dict[str, float]:
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    if len(a) == 0:
        return {"n": 0, "MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan"), "mean_err": float("nan")}
    e = p - a
    return {
        "n": int(len(a)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MAPE": float(np.mean(np.abs(e) / np.maximum(np.abs(a), 1.0))),
        "mean_err": float(np.mean(e)),
    }
