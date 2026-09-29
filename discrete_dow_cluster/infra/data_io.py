"""10 分鐘負載對齊與天氣載入（四臂管線自用，自洽，不依賴 sarimax_xgb）。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS_DAY = 144
BAR_MINUTES = 10
FREQ = "10min"

_CWB_MONTHLY_RE = re.compile(r"^(\d{5,6})-(\d{4})-(\d{2})\.csv$", re.IGNORECASE)


def existing_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    return None


def load_10min_load(path: Path) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["load"] = pd.to_numeric(df.get("load"), errors="coerce")
    df = df.dropna(subset=["date_time", "load"]).sort_values("date_time")
    s = df.set_index("date_time")["load"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(np.float64)


def align_to_10min(s: pd.Series, col: str = "load") -> pd.Series:
    """對齊到 10 分鐘網格；缺值用時間插值（短缺口）。"""
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return s.rename(col)
    out = s.asfreq(FREQ)
    out = out.interpolate(method="time", limit=6).ffill().bfill()
    return out.rename(col)


def _to_pct_humidity(s: pd.Series) -> pd.Series:
    med = float(s.median()) if s.notna().any() else 50.0
    if med <= 1.5:
        return s * 100.0
    return s


def load_odwo_10min(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=lambda c: str(c).lower() in {"obs_time", "temp", "humd", "wdsd"},
    )
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("obs_time")
    if tcol is None:
        return pd.DataFrame()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol])
    parts = {}
    if "temp" in cols:
        parts["T_mean"] = pd.to_numeric(df[cols["temp"]], errors="coerce")
    if "humd" in cols:
        parts["humidity"] = _to_pct_humidity(pd.to_numeric(df[cols["humd"]], errors="coerce"))
    if "wdsd" in cols:
        parts["wind_speed"] = pd.to_numeric(df[cols["wdsd"]], errors="coerce")
    if not parts:
        return pd.DataFrame()
    tmp = pd.DataFrame(parts)
    tmp["obs_time"] = df[tcol].to_numpy()
    g = tmp.groupby("obs_time", sort=True).mean(numeric_only=True)
    g.index = pd.to_datetime(g.index)
    frames = {c: align_to_10min(g[c], c) for c in g.columns}
    return pd.concat(frames, axis=1)


def _clean_sunshine(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").clip(lower=0)
    midnight = (s.index.hour == 0) & (s.index.minute == 0)
    s = s.copy()
    s.loc[midnight] = 0.0
    return s


def _list_cwb_monthly_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.csv") if _CWB_MONTHLY_RE.match(p.name))


def _to_float_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.strip()
        .replace({"": np.nan, "/": np.nan, "--": np.nan, "X": np.nan, "x": np.nan, "T": "0", "t": "0"})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_cwb_daily_stations_10min(folder: Path) -> pd.DataFrame:
    files = _list_cwb_monthly_files(folder)
    if not files:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for path in files:
        m = _CWB_MONTHLY_RE.match(path.name)
        if m is None:
            continue
        year, month = int(m.group(2)), int(m.group(3))
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
        except Exception:  # noqa: BLE001
            continue
        if "ObsTime" not in df.columns:
            continue
        day = pd.to_numeric(df["ObsTime"], errors="coerce")
        ok = day.notna() & (day >= 1) & (day <= 31)
        if not ok.any():
            continue
        day = day.loc[ok].astype(int)
        try:
            dates = pd.to_datetime(
                {"year": year, "month": month, "day": day.to_numpy()},
                errors="coerce",
            )
        except (ValueError, TypeError):
            continue
        rec: dict[str, np.ndarray | pd.Series] = {"date": dates}
        sub = df.loc[ok]
        if "Temperature" in sub.columns:
            rec["T_mean"] = _to_float_series(sub["Temperature"]).to_numpy()
        if "RH" in sub.columns:
            rec["humidity"] = _to_pct_humidity(_to_float_series(sub["RH"])).to_numpy()
        if "WS" in sub.columns:
            rec["wind_speed"] = _to_float_series(sub["WS"]).to_numpy()
        if "Sunshine" in sub.columns:
            rec["sunshine"] = _to_float_series(sub["Sunshine"]).clip(lower=0).to_numpy()
        part = pd.DataFrame(rec).dropna(subset=["date"])
        if part.empty:
            continue
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.normalize()
    daily = all_df.groupby("date", sort=True).mean(numeric_only=True)
    if daily.empty:
        return pd.DataFrame()

    pieces: list[pd.DataFrame] = []
    for day, row in daily.iterrows():
        idx = pd.date_range(pd.Timestamp(day), periods=SLOTS_DAY, freq=FREQ)
        block = pd.DataFrame({c: float(row[c]) for c in daily.columns}, index=idx)
        pieces.append(block)
    expanded = pd.concat(pieces).sort_index()
    expanded = expanded[~expanded.index.duplicated(keep="last")]
    if "sunshine" in expanded.columns:
        expanded["sunshine"] = _clean_sunshine(expanded["sunshine"])
    frames = {c: align_to_10min(expanded[c], c) for c in expanded.columns}
    return pd.concat(frames, axis=1)


def _load_station_weather_legacy_10min(folder: Path) -> pd.DataFrame:
    files = sorted(p for p in folder.glob("*.csv") if "odwo0011" not in p.name.lower())
    if not files:
        return pd.DataFrame()
    rows = []
    for path in files:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "obs_time" not in df.columns:
            continue
        df["obs_time"] = pd.to_datetime(df["obs_time"], errors="coerce", utc=True)
        df["obs_time"] = df["obs_time"].dt.tz_convert("Asia/Taipei").dt.tz_localize(None)
        rec = {"obs_time": df["obs_time"]}
        if "air_temp" in df.columns:
            rec["T_mean"] = pd.to_numeric(df["air_temp"], errors="coerce")
        if "rel_humidity" in df.columns:
            rec["humidity"] = _to_pct_humidity(pd.to_numeric(df["rel_humidity"], errors="coerce"))
        if "wind_speed" in df.columns:
            rec["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
        if "sunshine_duration" in df.columns:
            rec["sunshine"] = pd.to_numeric(df["sunshine_duration"], errors="coerce")
        rows.append(pd.DataFrame(rec))
    if not rows:
        return pd.DataFrame()
    all_df = pd.concat(rows, ignore_index=True).dropna(subset=["obs_time"])
    g = all_df.groupby("obs_time", sort=True).mean(numeric_only=True)
    if "sunshine" in g.columns:
        g["sunshine"] = _clean_sunshine(g["sunshine"])
    frames = {c: align_to_10min(g[c], c) for c in g.columns}
    return pd.concat(frames, axis=1)


def load_station_weather_10min(folder: Path | None) -> pd.DataFrame:
    if folder is None or not folder.exists():
        return pd.DataFrame()
    if _list_cwb_monthly_files(folder):
        return load_cwb_daily_stations_10min(folder)
    return _load_station_weather_legacy_10min(folder)


def load_solar_10min(path: Path | None) -> pd.Series:
    if path is None or not path.exists():
        return pd.Series(dtype=np.float64, name="sunshine")
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: str(c) in {"date_time", "solar"})
    if "solar" not in df.columns or "date_time" not in df.columns:
        return pd.Series(dtype=np.float64, name="sunshine")
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["solar"] = pd.to_numeric(df["solar"], errors="coerce")
    df = df.dropna(subset=["date_time"]).sort_values("date_time")
    s = df.set_index("date_time")["solar"]
    s = s[~s.index.duplicated(keep="last")]
    return align_to_10min(s, "sunshine")


def load_daily_temp_10min(path: Path | None, index: pd.DatetimeIndex) -> pd.Series:
    empty = pd.Series(np.nan, index=index, name="T_mean")
    if path is None or not path.exists() or index.empty:
        return empty
    w = pd.read_csv(path, parse_dates=["Date"])
    if "T_mean" not in w.columns:
        return empty
    w["Date"] = pd.to_datetime(w["Date"]).dt.normalize()
    daily = w.set_index("Date")["T_mean"]
    mapped = pd.Series(index.normalize().map(daily), index=index, name="T_mean")
    return pd.to_numeric(mapped, errors="coerce")


def load_holidays(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["Date", "name", "kind"])
    h = pd.read_csv(path, parse_dates=["Date"])
    h["Date"] = pd.to_datetime(h["Date"]).dt.normalize()
    h["kind"] = h["kind"].astype(str).str.lower()
    return h


def _combine_priority(*series: pd.Series) -> pd.Series:
    out = None
    for s in series:
        if s is None or s.empty:
            continue
        s = s.astype(np.float64)
        if out is None:
            out = s.copy()
        else:
            aligned = s.reindex(out.index)
            out = out.where(out.notna(), aligned)
            extra = s.loc[~s.index.isin(out.index)]
            if not extra.empty:
                out = pd.concat([out, extra]).sort_index()
                out = out[~out.index.duplicated(keep="first")]
    if out is None:
        return pd.Series(dtype=np.float64)
    return out


def merge_weather(
    index: pd.DatetimeIndex,
    odwo: pd.DataFrame,
    stations: pd.DataFrame,
    solar: pd.Series,
    daily_t: pd.Series,
) -> pd.DataFrame:
    t = _combine_priority(
        odwo["T_mean"] if "T_mean" in odwo.columns else None,
        stations["T_mean"] if "T_mean" in stations.columns else None,
        daily_t,
    )
    sun = _combine_priority(
        stations["sunshine"] if "sunshine" in stations.columns else None,
        solar,
    )
    rh = _combine_priority(
        odwo["humidity"] if "humidity" in odwo.columns else None,
        stations["humidity"] if "humidity" in stations.columns else None,
    )
    wind = _combine_priority(
        odwo["wind_speed"] if "wind_speed" in odwo.columns else None,
        stations["wind_speed"] if "wind_speed" in stations.columns else None,
    )
    out = pd.DataFrame(index=index)
    if not t.empty:
        out["T_mean"] = t.reindex(index)
    if not sun.empty:
        out["sunshine"] = sun.reindex(index)
    if not rh.empty:
        out["humidity"] = rh.reindex(index)
    if not wind.empty:
        out["wind_speed"] = wind.reindex(index)
    return out


def coverage_report(weather: pd.DataFrame) -> dict[str, str]:
    n = max(len(weather), 1)
    rep = {}
    for c in weather.columns:
        k = int(weather[c].notna().sum())
        if k == 0:
            rep[c] = "無資料"
            continue
        idx = weather.index[weather[c].notna()]
        rep[c] = f"{k}/{n}  {idx.min()} ~ {idx.max()}"
    return rep
