"""競賽綜合評分與日標籤抽取（四臂管線自用）。

Total = 0.6*S_peak_mw + 0.15*S_peak_time + 0.15*S_ramp_up + 0.1*S_ramp_down + S_under
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY_START = "11:00"
DAY_END = "17:00"
NIGHT_START = "17:10"
NIGHT_END = "21:00"


def _window_peak(day: pd.DataFrame, start: str, end: str) -> tuple[float, str]:
    part = day[(day["tod"] >= start) & (day["tod"] <= end)]
    if part.empty:
        return float("nan"), ""
    idx = part["load"].idxmax()
    row = part.loc[idx]
    return float(row["load"]), str(row["tod"])


def extract_day(day: pd.DataFrame) -> dict | None:
    """從單日 10 分鐘負載 DataFrame 抽出競賽 6 標籤。"""
    if len(day) < 100:
        return None
    peak_day, t_day = _window_peak(day, DAY_START, DAY_END)
    peak_night, t_night = _window_peak(day, NIGHT_START, NIGHT_END)
    diffs = day["load_diff"].dropna()
    if diffs.empty or pd.isna(peak_day) or pd.isna(peak_night):
        return None
    ramp_up = float(diffs.max())
    ramp_down = float((-diffs).max())
    return {
        "Date": day["date"].iloc[0].date().isoformat(),
        "Peak_MW_Day": peak_day,
        "Peak_Time_Day": t_day,
        "Peak_MW_Night": peak_night,
        "Peak_Time_Night": t_night,
        "Max_Ramp_Up_MW": ramp_up,
        "Max_Ramp_Down_MW": ramp_down,
        "n_points": int(len(day)),
        "weekday": int(day["date"].iloc[0].dayofweek),
    }


def _mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-6)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100.0)


def _time_to_steps(series: pd.Series) -> pd.Series:
    tod = pd.to_datetime(series.astype(str), format="%H:%M", errors="coerce")
    return tod.dt.hour * 6 + tod.dt.minute // 10


def _under_rel_mean(merged: pd.DataFrame, tcol: str, pcol: str) -> float:
    gap = merged[tcol] - merged[pcol]
    rel = np.where(merged[tcol] > 0, np.maximum(gap, 0) / merged[tcol], 0.0)
    return float(np.nanmean(rel))


def score_frame(true_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, float]:
    merged = true_df.merge(pred_df, on="Date", how="inner")
    if merged.empty:
        raise ValueError("true 與 pred 沒有重疊日期")

    s_peak_mw = 0.5 * (
        _mape(merged["Peak_MW_Day"], merged["Peak_MW_Day_pred"])
        + _mape(merged["Peak_MW_Night"], merged["Peak_MW_Night_pred"])
    )
    day_steps = np.abs(
        _time_to_steps(merged["Peak_Time_Day"]) - _time_to_steps(merged["Peak_Time_Day_pred"])
    )
    night_steps = np.abs(
        _time_to_steps(merged["Peak_Time_Night"]) - _time_to_steps(merged["Peak_Time_Night_pred"])
    )
    s_peak_time = 0.5 * (
        float(np.nanmean(np.power(day_steps, 1.2))) + float(np.nanmean(np.power(night_steps, 1.2)))
    )
    s_ramp_up = _mape(merged["Max_Ramp_Up_MW"], merged["Max_Ramp_Up_MW_pred"])
    s_ramp_down = _mape(merged["Max_Ramp_Down_MW"], merged["Max_Ramp_Down_MW_pred"])

    u_day = _under_rel_mean(merged, "Peak_MW_Day", "Peak_MW_Day_pred")
    u_night = _under_rel_mean(merged, "Peak_MW_Night", "Peak_MW_Night_pred")
    u_up = _under_rel_mean(merged, "Max_Ramp_Up_MW", "Max_Ramp_Up_MW_pred")
    penalty_peak = 0.2 * u_day + 0.2 * u_night
    penalty_ramp = 0.2 * u_up
    s_under = penalty_peak + penalty_ramp

    total = 0.6 * s_peak_mw + 0.15 * s_peak_time + 0.15 * s_ramp_up + 0.1 * s_ramp_down + s_under
    return {
        "n_days": float(len(merged)),
        "S_peak_mw": s_peak_mw,
        "S_peak_time": s_peak_time,
        "S_ramp_up": s_ramp_up,
        "S_ramp_down": s_ramp_down,
        "S_under_penalty": s_under,
        "Total_Score": total,
    }
