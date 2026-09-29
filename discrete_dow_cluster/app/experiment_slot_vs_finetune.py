"""Slot（D-1~D-7）vs FT-XGB 主實驗。

完全離散分群：負載/天氣按 Mon–Sun(+Holiday) 分桶，絕不跨 weekday 混用。
主臂：dow_proto_raw、dow_proto（日曆殘差）、slot_best、ft_xgb_best。
每天各自 origin、train<該日、只預測 24h。

正式三臂（非 raw）最後一步讀 settings.yaml → dow.underest_penalty：
  目標窗前多預測 lookback_days 天（pre_extra）→ g=mean(實際−pre)
  → p=clip(g,-C,C) → 最終=pre+p（雙向：低估上抬、高估下壓）。
定案 C 見 settings cap_by_arm（例 ft=2400 / slot=2800 / dow=1600）。

輸出 MAE/MAPE、curves、compare*.html、curves_pre_with_extra、underest_penalty.html。

  python -m app.experiment_slot_vs_finetune --origin 2026-09-10 --horizon-days 3 --skip-xgb-valid --open
  # 或：python run_experiment.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Div, HoverTool
from bokeh.plotting import figure, output_file, save

from domain.cluster_assign import CLUSTER_NAMES, assign_cluster, cluster_counts
from domain.daily_predict import (
    _day_label,
    apply_slot_separate_days,
    predict_dow_proto_resid_separate_days,
    predict_separate_days,
    roll_daily_proto_curves,
    select_slot_on_valid_daily,
)
from domain.dow_data_split import buckets_summary, split_series_by_cluster, split_weather_by_cluster
from domain.finetune import (
    finetune_xgb_separate_days,
    select_finetune_on_valid,
)
from domain.slot_adapt import (
    apply_level_scale_same_dow,
    apply_underest_penalty_separate_days,
    collect_bias_pre_extra,
    metrics,
)
from app.tune_underest_cap import score_curves
from domain.train_dow_xgb import load_common_data, load_settings, resolve_paths


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _day_mask(index: pd.DatetimeIndex, day: pd.Timestamp) -> np.ndarray:
    day = pd.Timestamp(day).normalize()
    return (index >= day) & (index < day + pd.Timedelta(days=1))


def _arm_row(
    name: str,
    method: str,
    pred: pd.Series,
    actual: pd.Series,
    dates: list[pd.Timestamp],
    *,
    lookback_or_k: float | int | None = None,
    beta: float | None = None,
    base_mae: float | None = None,
    base_mape: float | None = None,
) -> dict:
    y = actual.reindex(pred.index)
    m = metrics(y.to_numpy(), pred.to_numpy())
    out = {
        "arm": name,
        "method": method,
        "lookback_or_k": lookback_or_k,
        "beta": beta,
        **m,
    }
    for i, d in enumerate(dates):
        mask = _day_mask(pred.index, d)
        mm = metrics(y.loc[mask].to_numpy(), pred.loc[mask].to_numpy())
        out[f"MAE_d{i+1}"] = mm["MAE"]
        out[f"MAPE_d{i+1}"] = mm["MAPE"]
    if base_mae is not None and np.isfinite(m["MAE"]):
        out["MAE_improve_vs_proto"] = float(base_mae - m["MAE"])
    else:
        out["MAE_improve_vs_proto"] = float("nan")
    if base_mape is not None and np.isfinite(m["MAPE"]):
        out["MAPE_improve_vs_proto"] = float(base_mape - m["MAPE"])
    else:
        out["MAPE_improve_vs_proto"] = float("nan")
    return out


def _make_day_figure(
    day_idx: int,
    day: pd.Timestamp,
    y: pd.Series,
    arms: dict[str, pd.Series],
) -> figure:
    idx = y.index[_day_mask(y.index, day)]
    fig = figure(
        height=320,
        width=1100,
        x_axis_type="datetime",
        title=f"D{day_idx} {_day_label(day)}｜分開 24h",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    fig.toolbar.logo = None
    fig.yaxis.axis_label = "系統負載 (MW)"
    src_data: dict = {"t": idx, "actual": y.reindex(idx).to_numpy()}
    for name, series in arms.items():
        if series is not None and not series.empty:
            src_data[name] = series.reindex(idx).to_numpy()
    src = ColumnDataSource(src_data)
    fig.line("t", "actual", source=src, line_width=2.4, color="#111", legend_label="實際")
    colors = {
        "dow_proto_raw": ("#94a3b8", "dotted"),
        "dow_proto": ("#2563eb", "dashed"),
        "slot_best": ("#dc2626", "dotted"),
        "ft_xgb_best": ("#9333ea", "dotted"),
    }
    for name, (color, dash) in colors.items():
        if name in src_data:
            fig.line("t", name, source=src, line_width=1.8, color=color, legend_label=name, line_dash=dash)
    tips = [("時間", "@t{%F %H:%M}"), ("實際", "@actual{0,0.0}")]
    for name in colors:
        if name in src_data:
            tips.append((name, f"@{name}{{0,0.0}}"))
    fig.add_tools(HoverTool(tooltips=tips, formatters={"@t": "datetime"}, mode="vline"))
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


def _make_resid_figure(
    day_idx: int,
    day: pd.Timestamp,
    gap: np.ndarray,
) -> figure:
    """殘差基準線：slot 0..143，y=MW；水平虛線 y=0。"""
    slots = np.arange(len(gap), dtype=int)
    fig = figure(
        height=220,
        width=1100,
        title=f"D{day_idx} {_day_label(day)}｜殘差基準 (actual−proto) D-1/D-2 mean",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    fig.toolbar.logo = None
    fig.xaxis.axis_label = "slot (10min)"
    fig.yaxis.axis_label = "殘差 (MW)"
    src = ColumnDataSource({"slot": slots, "resid": np.asarray(gap, dtype=float)})
    fig.line("slot", "resid", source=src, line_width=2.0, color="#ea580c", legend_label="resid_baseline")
    fig.scatter("slot", "resid", source=src, size=3, color="#ea580c", alpha=0.6)
    fig.line(slots, np.zeros_like(slots, dtype=float), line_width=1.2, color="#6b7280", line_dash="dashed", legend_label="y=0")
    fig.add_tools(
        HoverTool(tooltips=[("slot", "@slot"), ("resid", "@resid{0,0.0}")], mode="vline")
    )
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


def _make_penalty_figure(
    day_idx: int,
    day: pd.Timestamp,
    g: np.ndarray,
    p: np.ndarray,
    cap_mw: float,
    *,
    arm: str = "dow_proto",
) -> figure:
    """雙向偏差校正：g=mean(actual−pre)，p=clip(g,-C,C)。"""
    slots = np.arange(len(p), dtype=int)
    cap = abs(float(cap_mw))
    fig = figure(
        height=220,
        width=1100,
        title=f"D{day_idx} {_day_label(day)}｜{arm} 雙向校正 p=clip(g,-C,C) C={cap:.0f}",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    fig.toolbar.logo = None
    fig.xaxis.axis_label = "slot (10min)"
    fig.yaxis.axis_label = "MW"
    src = ColumnDataSource(
        {
            "slot": slots,
            "g": np.asarray(g, dtype=float),
            "p": np.asarray(p, dtype=float),
        }
    )
    fig.line("slot", "g", source=src, line_width=1.5, color="#64748b", legend_label="g (raw)", line_dash="dotted")
    fig.line("slot", "p", source=src, line_width=2.0, color="#dc2626", legend_label="p (clip)")
    fig.line(slots, np.zeros_like(slots, dtype=float), line_width=1.0, color="#6b7280", line_dash="dashed", legend_label="y=0")
    fig.line(
        slots,
        np.full_like(slots, cap, dtype=float),
        line_width=1.0,
        color="#f59e0b",
        line_dash="dashed",
        legend_label=f"+C={cap:.0f}",
    )
    fig.line(
        slots,
        np.full_like(slots, -cap, dtype=float),
        line_width=1.0,
        color="#0ea5e9",
        line_dash="dashed",
        legend_label=f"-C={-cap:.0f}",
    )
    fig.add_tools(
        HoverTool(tooltips=[("slot", "@slot"), ("g", "@g{0,0.0}"), ("p", "@p{0,0.0}")], mode="vline")
    )
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    return fig


def main() -> None:
    _configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2026-08-06")
    ap.add_argument("--horizon-days", type=int, default=3)
    ap.add_argument("--skip-xgb-valid", action="store_true", help="跳過 valid 期 FT-XGB 選參（加速）")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    cfg = load_settings()
    paths = resolve_paths(cfg)
    dow_cfg = cfg.get("dow") or {}
    splits = cfg.get("splits") or {}
    feat_cfg = cfg.get("features") or {}
    xgb_cfg = cfg.get("xgb") or {}

    origin = pd.Timestamp(args.origin).normalize()
    horizon_days = int(args.horizon_days)
    gate_valid_start = pd.Timestamp(str(splits.get("gate_valid_start", "2026-07-01")))
    gate_test_start = pd.Timestamp(str(splits.get("gate_test_start", "2026-08-01")))
    target_dates = [origin + pd.Timedelta(days=i) for i in range(horizon_days)]

    lookbacks = list(dow_cfg.get("slot_lookbacks") or list(range(1, 8)))
    ft_ks = list(dow_cfg.get("finetune_lookbacks") or list(range(1, 8)))
    betas = list(dow_cfg.get("slot_betas") or [0.3, 0.5, 0.7, 1.0])
    same_dow_only = bool(dow_cfg.get("slot_same_dow_only", True))
    max_days = int(dow_cfg.get("proto_max_days", 16))
    proto_resid_lb = int(dow_cfg.get("proto_resid_lookback", 2))
    proto_resid_beta = float(dow_cfg.get("proto_resid_beta", 1.0))
    pen_cfg = dow_cfg.get("underest_penalty") or {}
    pen_enable = bool(pen_cfg.get("enable", True))
    pen_lb = int(pen_cfg.get("lookback_days", 2))
    pen_prev_dow = bool(pen_cfg.get("include_prev_same_dow", True))
    pen_same_w = float(pen_cfg.get("same_dow_weight", 2.0))
    pen_cal_w = float(pen_cfg.get("calendar_weight", 1.0))
    pen_gap_mode = str(pen_cfg.get("gap_mode", "peak")).lower()
    pen_peak_blend = float(pen_cfg.get("peak_blend", 0.65))
    pen_weekend_uplift = bool(pen_cfg.get("weekend_uplift_only", True))
    pen_cap = float(pen_cfg.get("cap_mw", 0.0))
    pen_by_arm_cfg = pen_cfg.get("cap_by_arm") or {}
    # 定案預設與 settings.yaml 一致；缺鍵時回退到定案值（勿再用過小的 800）
    pen_cap_by_arm = {
        "dow_proto": float(pen_by_arm_cfg.get("dow_proto", 1600.0)),
        "slot_best": float(pen_by_arm_cfg.get("slot_best", 2800.0)),
        "ft_xgb_best": float(pen_by_arm_cfg.get("ft_xgb_best", 2400.0)),
    }
    forecast_mode = str(dow_cfg.get("forecast_mode", "rolling")).lower()
    holiday_cid = int(dow_cfg.get("holiday_cluster_id", 7))
    lvl_cfg = dow_cfg.get("level_scale") or {}
    lvl_enable = bool(lvl_cfg.get("enable", True))
    lvl_k = int(lvl_cfg.get("k_same_dow", 2))
    lvl_clip = list(lvl_cfg.get("clip") or [0.85, 1.15])
    lvl_lo, lvl_hi = float(lvl_clip[0]), float(lvl_clip[1])
    min_samples = int(dow_cfg.get("min_cluster_samples", 200))
    xgb_train_days = int(dow_cfg.get("xgb_train_days", 365))
    weather_lag = int(feat_cfg.get("weather_lag_days", 1))

    out_dir = (
        Path(paths["output_dir"])
        / f"slot_vs_ft_{origin.strftime('%Y%m%d')}_{(origin + pd.Timedelta(days=horizon_days - 1)).strftime('%Y%m%d')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_common_data(cfg, paths)
    load10, holidays, actual, weather = (
        data["load10"],
        data["holidays"],
        data["actual"],
        data.get("weather", pd.DataFrame()),
    )
    clusters_all = assign_cluster(load10.index, holidays)
    day_clusters = clusters_all.groupby(clusters_all.index.normalize()).first()
    cc = cluster_counts(clusters_all)
    cc.to_csv(out_dir / "cluster_counts.csv", index=False, encoding="utf-8-sig")

    load_by = split_series_by_cluster(load10, clusters_all)
    weather_by = split_weather_by_cluster(weather, clusters_all)
    buckets_summary(load_by, weather_by).to_csv(
        out_dir / "dow_buckets_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(cc.to_string(index=False), flush=True)

    idx_days = set(pd.DatetimeIndex(load10.dropna().index).normalize())
    max_lb = max(lookbacks) if lookbacks else 7
    # gate origins：valid 期 + 同群回看需要的前置日
    gate_origins = [
        pd.Timestamp(d)
        for d in pd.date_range(
            gate_valid_start - pd.Timedelta(days=max_lb * 8),  # 同 weekday 約 7 天一現
            gate_test_start - pd.Timedelta(days=1),
            freq="D",
        )
        if pd.Timestamp(d).normalize() in idx_days
    ]

    # ---------- GATE：slot ----------
    print("\n===== GATE 分開日 proto 曲線（valid）=====", flush=True)
    gate_curves = roll_daily_proto_curves(
        load10=load10,
        holidays=holidays,
        origins=gate_origins,
        max_days_per_cluster=max_days,
        valid_start=gate_valid_start,
        test_start=gate_test_start,
    )
    gate_curves.to_csv(out_dir / "gate_dow_proto_curves_10min.csv", index=False, encoding="utf-8-sig")

    print("\n===== GATE 搜尋 slot（同 DOW，lookback 1~7）=====", flush=True)
    slot_best, slot_search = select_slot_on_valid_daily(
        actual,
        gate_curves,
        valid_start=gate_valid_start,
        test_start=gate_test_start,
        lookbacks=lookbacks,
        betas=betas,
        clusters=clusters_all,
        same_dow_only=same_dow_only,
    )
    slot_search.to_csv(out_dir / "slot_valid_search.csv", index=False, encoding="utf-8-sig")
    freeze_lb = int(slot_best["lookback_days"])
    freeze_beta = float(slot_best["beta"])
    print(f"凍結 slot lookback={freeze_lb} beta={freeze_beta}", flush=True)

    # ---------- GATE：FT-xgb ----------
    valid_days = [
        pd.Timestamp(d)
        for d in pd.date_range(gate_valid_start, gate_test_start - pd.Timedelta(days=1), freq="D")
        if pd.Timestamp(d).normalize() in idx_days
    ]
    if args.skip_xgb_valid:
        if 8 in ft_ks:
            freeze_k_xgb = 8
        elif ft_ks:
            freeze_k_xgb = int(max(ft_ks))
        else:
            freeze_k_xgb = 8
        ft_xgb_best = {
            "method": "ft_xgb",
            "k_same_dow": freeze_k_xgb,
            "n_days": 0,
            "valid_mean_MAE": float("nan"),
            "valid_mean_MAPE": float("nan"),
            "note": "skipped_valid_used_default_k8",
        }
        print(f"\n===== GATE FT-xgb 跳過，預設 K={freeze_k_xgb} =====", flush=True)
    else:
        print("\n===== GATE 搜尋 FT-xgb K =====", flush=True)
        ft_xgb_best, ft_xgb_search = select_finetune_on_valid(
            actual=actual,
            load10=load10,
            holidays=holidays,
            weather=weather,
            valid_days=valid_days,
            ks=ft_ks,
            method="xgb",
            max_days_per_cluster=max_days,
            xgb_train_days=xgb_train_days,
            min_samples=min_samples,
            xgb_params=xgb_cfg,
            weather_lag_days=weather_lag,
            quiet=True,
        )
        ft_xgb_search.to_csv(out_dir / "ft_xgb_valid_search.csv", index=False, encoding="utf-8-sig")
        freeze_k_xgb = int(ft_xgb_best["k_same_dow"])
        print(f"凍結 FT-xgb K={freeze_k_xgb}", flush=True)

    best_params = {
        "origin": str(origin.date()),
        "predict_mode": "separate_24h_per_day",
        "slot_same_dow_only": same_dow_only,
        "proto_max_days": max_days,
        "proto_resid": {
            "lookback_days": proto_resid_lb,
            "beta": proto_resid_beta,
            "mode": "calendar_D-1_D-N",
        },
        "forecast_mode": forecast_mode,
        "underest_penalty": {
            "enable": pen_enable,
            "lookback_days": pen_lb,
            "include_prev_same_dow": pen_prev_dow,
            "same_dow_weight": pen_same_w,
            "calendar_weight": pen_cal_w,
            "gap_mode": pen_gap_mode,
            "peak_blend": pen_peak_blend,
            "weekend_uplift_only": pen_weekend_uplift,
            "cap_mw": pen_cap,
            "cap_by_arm": pen_cap_by_arm,
            "arms": ["dow_proto", "slot_best", "ft_xgb_best"],
        },
        "level_scale": {
            "enable": lvl_enable,
            "k_same_dow": lvl_k,
            "clip": [lvl_lo, lvl_hi],
        },
        "slot": {"lookback_days": freeze_lb, "beta": freeze_beta, **{k: slot_best.get(k) for k in ("valid_mean_MAE", "valid_mean_MAPE", "n_days")}},
        "ft_xgb": dict(ft_xgb_best),
        "clusters": CLUSTER_NAMES,
    }
    (out_dir / "best_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # ---------- TEST ----------
    print(f"\n===== TEST 分開日 {target_dates[0].date()}~{target_dates[-1].date()} =====", flush=True)
    arms: dict[str, pd.Series] = {}
    penalty_meta: dict[str, dict[pd.Timestamp, dict]] = {}

    # 前置日：日曆 D-1..D-N ∪ 各目標日上週同 weekday（估 g 用）
    pre_extra = []
    if pen_enable:
        pre_extra = collect_bias_pre_extra(
            origin,
            target_dates,
            pen_lb,
            include_prev_same_dow=pen_prev_dow,
            idx_days=idx_days,
            forecast_mode=forecast_mode,
        )
    pre_dates = pre_extra + list(target_dates)
    if pen_enable:
        cap_desc = ", ".join(f"{a}={c:.0f}" for a, c in pen_cap_by_arm.items())
        print(
            f"  bias_clip(enable) mode={forecast_mode} gap={pen_gap_mode} "
            f"peak_blend={pen_peak_blend:.2f} weekend_uplift_only={pen_weekend_uplift} "
            f"lb={pen_lb} prev_same_dow={pen_prev_dow} "
            f"w_same={pen_same_w} w_cal={pen_cal_w} "
            f"default_C={pen_cap:.0f} by_arm[{cap_desc}] "
            f"pre_extra={[str(d.date()) for d in pre_extra]}",
            flush=True,
        )
    best_params["underest_penalty"]["note"] = (
        "bias_clip: peak tip + weekend uplift-only; weighted refs; p=clip(g,-C,C)"
    )
    best_params["underest_penalty"]["pre_extra_dates"] = [str(d.date()) for d in pre_extra]
    (out_dir / "best_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("\n-- dow_proto_raw --", flush=True)
    arms["dow_proto_raw"] = predict_separate_days(
        load10=load10,
        holidays=holidays,
        target_dates=target_dates,
        max_days_per_cluster=max_days,
    )

    print(
        f"\n-- dow_proto (resid calendar lb={proto_resid_lb} beta={proto_resid_beta}) --",
        flush=True,
    )
    pre_dow, resid_by_day = predict_dow_proto_resid_separate_days(
        actual=actual,
        load10=load10,
        holidays=holidays,
        target_dates=pre_dates if pen_enable else target_dates,
        max_days_per_cluster=max_days,
        idx_days=idx_days,
        valid_start=gate_valid_start,
        lookback_days=proto_resid_lb,
        beta=proto_resid_beta,
    )
    def _apply_bias(pre_ser, arm_name):
        return apply_underest_penalty_separate_days(
            actual,
            pre_ser,
            target_dates,
            lookback_days=pen_lb,
            cap_mw=pen_cap_by_arm[arm_name],
            idx_days=idx_days,
            include_prev_same_dow=pen_prev_dow,
            same_dow_weight=pen_same_w,
            calendar_weight=pen_cal_w,
            forecast_mode=forecast_mode,
            holiday_cluster_id=holiday_cid,
            clusters=day_clusters,
            gap_mode=pen_gap_mode,
            peak_blend=pen_peak_blend,
            weekend_uplift_only=pen_weekend_uplift,
        )

    if pen_enable:
        arms["dow_proto"], penalty_meta["dow_proto"] = _apply_bias(pre_dow, "dow_proto")
    else:
        arms["dow_proto"] = pre_dow.loc[
            (pre_dow.index >= target_dates[0])
            & (pre_dow.index < target_dates[-1] + pd.Timedelta(days=1))
        ]
    # improve_vs_proto 相對未補正 raw
    base_m = metrics(
        actual.reindex(arms["dow_proto_raw"].index).to_numpy(),
        arms["dow_proto_raw"].to_numpy(),
    )
    base_mae, base_mape = base_m["MAE"], base_m["MAPE"]

    resid_rows = []
    for d, gap in resid_by_day.items():
        if pd.Timestamp(d).normalize() not in {pd.Timestamp(x).normalize() for x in target_dates}:
            continue
        for s, v in enumerate(np.asarray(gap, dtype=float)):
            resid_rows.append({"date": str(pd.Timestamp(d).date()), "slot": s, "resid_mw": float(v)})
    pd.DataFrame(resid_rows).to_csv(
        out_dir / "resid_baseline_by_day.csv", index=False, encoding="utf-8-sig"
    )

    print("\n-- slot_best --", flush=True)
    pre_slot, used = apply_slot_separate_days(
        actual=actual,
        load10=load10,
        holidays=holidays,
        target_dates=pre_dates if pen_enable else target_dates,
        max_days_per_cluster=max_days,
        lookback_days=freeze_lb,
        beta=freeze_beta,
        idx_days=idx_days,
        valid_start=gate_valid_start,
        same_dow_only=same_dow_only,
    )
    print(f"  slot days_used={used}", flush=True)
    if pen_enable:
        arms["slot_best"], penalty_meta["slot_best"] = _apply_bias(pre_slot, "slot_best")
    else:
        arms["slot_best"] = pre_slot

    # 各 lookback 單獨記錄（beta 用凍結值；不對對照臂套懲罰以加速）
    for lb in lookbacks:
        name = f"slot_lb{lb}"
        print(f"\n-- {name} --", flush=True)
        arms[name], _ = apply_slot_separate_days(
            actual=actual,
            load10=load10,
            holidays=holidays,
            target_dates=target_dates,
            max_days_per_cluster=max_days,
            lookback_days=lb,
            beta=freeze_beta,
            idx_days=idx_days,
            valid_start=gate_valid_start,
            same_dow_only=same_dow_only,
            quiet=True,
        )

    print(f"\n-- ft_xgb_best K={freeze_k_xgb} --", flush=True)
    pre_xgb = finetune_xgb_separate_days(
        load10=load10,
        holidays=holidays,
        weather=weather,
        target_dates=pre_dates if pen_enable else target_dates,
        k_same_dow=freeze_k_xgb,
        max_days_per_cluster=max_days,
        xgb_train_days=xgb_train_days,
        min_samples=min_samples,
        xgb_params=xgb_cfg,
        weather_lag_days=weather_lag,
    )
    level_meta = {}
    if lvl_enable:
        pre_xgb, level_meta = apply_level_scale_same_dow(
            actual,
            pre_xgb,
            target_dates,
            k_same_dow=lvl_k,
            clip_lo=lvl_lo,
            clip_hi=lvl_hi,
            idx_days=idx_days,
        )
        print(
            "  level_scale: "
            + ", ".join(
                f"{pd.Timestamp(d).date()}×{level_meta[d]['scale']:.3f}"
                for d in target_dates
                if pd.Timestamp(d).normalize() in level_meta
            ),
            flush=True,
        )
        pd.DataFrame(
            [
                {
                    "date": str(pd.Timestamp(d).date()),
                    "scale": level_meta[pd.Timestamp(d).normalize()]["scale"],
                    "hist_days": ";".join(level_meta[pd.Timestamp(d).normalize()]["hist_days"]),
                }
                for d in target_dates
                if pd.Timestamp(d).normalize() in level_meta
            ]
        ).to_csv(out_dir / "level_scale_by_day.csv", index=False, encoding="utf-8-sig")
    if pen_enable:
        arms["ft_xgb_best"], penalty_meta["ft_xgb_best"] = _apply_bias(pre_xgb, "ft_xgb_best")
    else:
        arms["ft_xgb_best"] = pre_xgb
    for k in ft_ks:
        name = f"ft_xgb_k{k}"
        print(f"\n-- {name} --", flush=True)
        arms[name] = finetune_xgb_separate_days(
            load10=load10,
            holidays=holidays,
            weather=weather,
            target_dates=target_dates,
            k_same_dow=k,
            max_days_per_cluster=max_days,
            xgb_train_days=xgb_train_days,
            min_samples=min_samples,
            xgb_params=xgb_cfg,
            weather_lag_days=weather_lag,
            quiet=True,
        )

    # 雙向偏差校正明細（鍵名 underest_* 相容舊輸出）
    pre_extra_str = ";".join(str(d.date()) for d in pre_extra)
    pen_rows = []
    for arm_name, meta_by_day in penalty_meta.items():
        arm_cap = float(pen_cap_by_arm.get(arm_name, pen_cap))
        for d, m in meta_by_day.items():
            g = np.asarray(m["g"], dtype=float)
            p = np.asarray(m["p"], dtype=float)
            for s in range(len(p)):
                pen_rows.append(
                    {
                        "arm": arm_name,
                        "date": str(pd.Timestamp(d).date()),
                        "slot": s,
                        "g_mw": float(g[s]),
                        "p_mw": float(p[s]),
                        "cap_mw": arm_cap,
                        "days_used": int(m["used"]),
                        "pre_extra_dates": pre_extra_str,
                        "ref_days": ";".join(m.get("ref_days") or []),
                    }
                )
    pd.DataFrame(pen_rows).to_csv(
        out_dir / "underest_penalty_by_day.csv", index=False, encoding="utf-8-sig"
    )

    # 主臂 pre 全長（含目標日前 lookback 天），供驗證「有無多預測前置日」
    pre_frame: dict = {}
    pre_idx = None
    for pre_name, pre_ser in (
        ("dow_proto_pre", pre_dow if pen_enable else None),
        ("slot_best_pre", pre_slot if pen_enable else None),
        ("ft_xgb_best_pre", pre_xgb if pen_enable else None),
    ):
        if pre_ser is None or pre_ser.empty:
            continue
        if pre_idx is None:
            pre_idx = pre_ser.index
            pre_frame["date_time"] = pre_idx
            pre_frame["actual"] = actual.reindex(pre_idx).to_numpy()
            pre_frame["is_pre_extra"] = [
                int(pd.Timestamp(t).normalize() in {pd.Timestamp(x).normalize() for x in pre_extra})
                for t in pre_idx
            ]
        pre_frame[pre_name] = pre_ser.reindex(pre_idx).to_numpy()
    if pre_frame:
        pd.DataFrame(pre_frame).to_csv(
            out_dir / "curves_pre_with_extra_10min.csv", index=False, encoding="utf-8-sig"
        )

    # ---------- metrics ----------
    metric_rows = []
    for name, series in arms.items():
        if series is None or series.empty:
            continue
        if name == "dow_proto":
            method, lk, beta = "proto_resid", proto_resid_lb, proto_resid_beta
        elif name == "dow_proto_raw":
            method, lk, beta = "proto", max_days, None
        elif name == "slot_best":
            method, lk, beta = "slot", freeze_lb, freeze_beta
        elif name.startswith("slot_lb"):
            method, lk, beta = "slot", int(name.replace("slot_lb", "")), freeze_beta
        elif name == "ft_xgb_best":
            method, lk, beta = "ft_xgb", freeze_k_xgb, None
        elif name.startswith("ft_xgb_k"):
            method, lk, beta = "ft_xgb", int(name.replace("ft_xgb_k", "")), None
        else:
            method, lk, beta = "other", None, None
        metric_rows.append(
            _arm_row(
                name,
                method,
                series.reindex(arms["dow_proto"].index),
                actual,
                target_dates,
                lookback_or_k=lk,
                beta=beta,
                base_mae=base_mae,
                base_mape=base_mape,
            )
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("MAE")

    # 比賽 Total_Score（scoring.py；越低越好）
    score_rows = []
    y_score = actual.reindex(arms["dow_proto"].index)
    for name in ["dow_proto_raw", "dow_proto", "slot_best", "ft_xgb_best"]:
        if name not in arms or arms[name] is None or arms[name].empty:
            continue
        sc = score_curves(y_score, arms[name].reindex(y_score.index), list(target_dates))
        score_rows.append({"arm": name, **sc})
    scores_df = pd.DataFrame(score_rows)
    if not scores_df.empty:
        metrics_df = metrics_df.merge(scores_df, on="arm", how="left")
        scores_df.to_csv(out_dir / "competition_scores.csv", index=False, encoding="utf-8-sig")

    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    # by-day summary for key arms
    by_day_rows = []
    for name in ["dow_proto_raw", "dow_proto", "slot_best", "ft_xgb_best"]:
        if name not in arms:
            continue
        pred = arms[name]
        for i, d in enumerate(target_dates):
            mask = _day_mask(pred.index, d)
            mm = metrics(actual.reindex(pred.index).loc[mask].to_numpy(), pred.loc[mask].to_numpy())
            day_sc = score_curves(
                actual.reindex(pred.index).loc[mask],
                pred.loc[mask],
                [pd.Timestamp(d)],
            )
            by_day_rows.append(
                {
                    "arm": name,
                    "day_idx": i + 1,
                    "date": str(pd.Timestamp(d).date()),
                    "label": _day_label(d),
                    "MAE": mm["MAE"],
                    "MAPE": mm["MAPE"],
                    "RMSE": mm["RMSE"],
                    "Total_Score": day_sc.get("Total_Score", float("nan")),
                    "S_peak_mw": day_sc.get("S_peak_mw", float("nan")),
                    "S_peak_time": day_sc.get("S_peak_time", float("nan")),
                    "S_under_penalty": day_sc.get("S_under_penalty", float("nan")),
                }
            )
    metrics_by_day = pd.DataFrame(by_day_rows)
    metrics_by_day.to_csv(out_dir / "metrics_by_day.csv", index=False, encoding="utf-8-sig")

    print("\n=== 多臂 MAE / MAPE ===", flush=True)
    show_cols = [c for c in ["arm", "method", "lookback_or_k", "beta", "MAE", "MAPE", "MAE_d1", "MAE_d2", "MAE_d3"] if c in metrics_df.columns]
    print(metrics_df[show_cols].to_string(index=False), flush=True)
    if not scores_df.empty:
        print("\n=== 比賽 Total_Score（越低越好）===", flush=True)
        print(
            scores_df[["arm", "Total_Score", "S_peak_mw", "S_peak_time", "S_ramp_up", "S_ramp_down", "S_under_penalty"]]
            .sort_values("Total_Score")
            .to_string(index=False),
            flush=True,
        )

    # 關鍵臂曲線對齊
    key_arms = ["dow_proto_raw", "dow_proto", "slot_best", "ft_xgb_best"]
    frame = {"date_time": arms["dow_proto"].index, "actual": actual.reindex(arms["dow_proto"].index).to_numpy()}
    for name in key_arms:
        if name in arms:
            frame[name] = arms[name].reindex(arms["dow_proto"].index).to_numpy()
    pd.DataFrame(frame).to_csv(out_dir / "curves_aligned_10min.csv", index=False, encoding="utf-8-sig")

    # HTML
    def _fmt(row, col):
        v = row.get(col, float("nan"))
        return f"{v:.1f}" if np.isfinite(v) else "nan"

    table = (
        "<table style='border-collapse:collapse;font-family:Consolas,monospace;font-size:13px'>"
        "<tr style='background:#f3f4f6'>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>arm</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>MAE</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>MAPE</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>RMSE</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>Total_Score</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>S_peak_time</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>S_under</th></tr>"
    )
    for _, r in metrics_df[metrics_df["arm"].isin(key_arms)].sort_values(
        "Total_Score" if "Total_Score" in metrics_df.columns else "MAE"
    ).iterrows():
        ts = r.get("Total_Score", float("nan"))
        spt = r.get("S_peak_time", float("nan"))
        su = r.get("S_under_penalty", float("nan"))
        table += (
            f"<tr><td style='padding:4px 8px;border:1px solid #ddd'>{r['arm']}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{_fmt(r, 'MAE')}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{r['MAPE']*100:.2f}%</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{_fmt(r, 'RMSE')}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{ts:.4f}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{spt:.3f}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{su:.4f}</td></tr>"
        )
    table += "</table>"
    suggest_html = ""
    if not scores_df.empty and "Total_Score" in scores_df.columns:
        best_arm = scores_df.sort_values("Total_Score").iloc[0]
        suggest_html = (
            f"<p style='font-family:Microsoft JhengHei,Segoe UI,sans-serif;color:#065f46'>"
            f"<b>建議交卷臂</b>（本窗 Total_Score 最低）= <b>{best_arm['arm']}</b> "
            f"（Total={best_arm['Total_Score']:.4f}）。僅提示，未自動 blend。</p>"
        )

    y = actual.reindex(arms["dow_proto"].index)
    plot_arms = {k: arms[k] for k in key_arms if k in arms}
    pre_extra_disp = [str(d.date()) for d in pre_extra] if pre_extra else []
    bias_line = (
        f"mode={forecast_mode}；雙向偏差 on [{', '.join(f'{a}={int(c)}' for a, c in pen_cap_by_arm.items())}]；"
        f"refs=日曆D-1..D-{pen_lb}"
        + ("+上週同weekday" if pen_prev_dow else "")
        + f"（w_same={pen_same_w}, w_cal={pen_cal_w}）；"
        + (f"level_scale on k={lvl_k}；" if lvl_enable else "")
        + f"pre_extra={pre_extra_disp}。"
        if pen_enable
        else "雙向偏差校正 off。"
    )
    blocks = [
        Div(
            text=(
                "<h2 style='margin:0;font-family:Microsoft JhengHei,Segoe UI,sans-serif'>"
                f"Slot vs Fine-tune｜origin={origin.date()}</h2>"
                "<p style='font-family:Microsoft JhengHei,Segoe UI,sans-serif;color:#444'>"
                f"dow_proto = 原型 + 日曆殘差 lb={proto_resid_lb} β={proto_resid_beta}；"
                f"slot=({freeze_lb},{freeze_beta})；"
                f"FT-xgb K={freeze_k_xgb}；"
                f"{bias_line}"
                "每日分開 24h。</p>"
                + suggest_html
                + table
            ),
            width=1100,
        )
    ]
    src_data = {"t": arms["dow_proto"].index, "actual": y.to_numpy()}
    for name in key_arms:
        if name in arms:
            src_data[name] = arms[name].reindex(arms["dow_proto"].index).to_numpy()
    src = ColumnDataSource(src_data)
    fig = figure(
        height=400,
        width=1100,
        x_axis_type="datetime",
        title="實際 vs proto_raw / proto+resid / slot / ft_xgb",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    fig.toolbar.logo = None
    fig.line("t", "actual", source=src, line_width=2.4, color="#111", legend_label="實際")
    style = {
        "dow_proto_raw": ("#94a3b8", "dotted"),
        "dow_proto": ("#2563eb", "dashed"),
        "slot_best": ("#dc2626", "dotted"),
        "ft_xgb_best": ("#9333ea", "dotted"),
    }
    for name, (color, dash) in style.items():
        if name in src_data:
            fig.line("t", name, source=src, line_width=2.0, color=color, legend_label=name, line_dash=dash)
    fig.add_tools(
        HoverTool(
            tooltips=[("時間", "@t{%F %H:%M}"), ("實際", "@actual{0,0.0}")]
            + [(n, f"@{n}{{0,0.0}}") for n in style if n in src_data],
            formatters={"@t": "datetime"},
            mode="vline",
        )
    )
    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    blocks.append(fig)

    html_path = out_dir / "compare.html"
    output_file(str(html_path), title=f"slot vs ft {origin.date()}")
    save(column(*blocks))

    day_table = (
        "<table style='border-collapse:collapse;font-family:Consolas,monospace;font-size:13px'>"
        "<tr style='background:#f3f4f6'>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>日</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>raw MAE</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>proto+resid MAE</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>slot MAE</th>"
        "<th style='padding:4px 8px;border:1px solid #ddd'>ft_xgb MAE</th></tr>"
    )
    for i, d in enumerate(target_dates):
        mask = _day_mask(arms["dow_proto"].index, d)
        row_mae = {}
        for name in key_arms:
            if name in arms:
                row_mae[name] = metrics(y.loc[mask].to_numpy(), arms[name].reindex(y.index).loc[mask].to_numpy())["MAE"]
        day_table += (
            f"<tr><td style='padding:4px 8px;border:1px solid #ddd'>D{i+1} {_day_label(d)}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{row_mae.get('dow_proto_raw', float('nan')):.1f}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{row_mae.get('dow_proto', float('nan')):.1f}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{row_mae.get('slot_best', float('nan')):.1f}</td>"
            f"<td style='padding:4px 8px;border:1px solid #ddd'>{row_mae.get('ft_xgb_best', float('nan')):.1f}</td></tr>"
        )
    day_table += "</table>"
    day_blocks: list = [
        Div(
            text=(
                "<h2 style='margin:0;font-family:Microsoft JhengHei,Segoe UI,sans-serif'>"
                f"分開日 24h｜{origin.date()}~{target_dates[-1].date()}</h2>"
                "<p style='color:#444;font-family:Microsoft JhengHei,Segoe UI,sans-serif;margin:6px 0'>"
                f"dow_proto 殘差日曆 lb={proto_resid_lb} β={proto_resid_beta}；"
                f"slot same_dow={same_dow_only}；"
                f"{bias_line}</p>"
                + day_table
            ),
            width=1100,
        )
    ]
    resid_blocks: list = [
        Div(
            text=(
                "<h2 style='margin:0;font-family:Microsoft JhengHei,Segoe UI,sans-serif'>"
                f"殘差基準線｜滾動日曆 D-1..D-{proto_resid_lb}</h2>"
                "<p style='color:#444;font-family:Microsoft JhengHei,Segoe UI,sans-serif;margin:6px 0'>"
                "每天使用的 slot 殘差 mean(actual−proto)；加回後得到 dow_proto（懲罰前）。</p>"
            ),
            width=1100,
        )
    ]
    pen_blocks: list = [
        Div(
            text=(
                "<h2 style='margin:0;font-family:Microsoft JhengHei,Segoe UI,sans-serif'>"
                f"雙向偏差校正（bias_clip）｜refs=日曆 D-1..D-{pen_lb}"
                + (" + 上週同weekday" if pen_prev_dow else "")
                + f"，最終=pre+clip(g,-C,C)，"
                f"C=[{', '.join(f'{a}={int(c)}' for a,c in pen_cap_by_arm.items())}]</h2>"
                "<p style='color:#444;font-family:Microsoft JhengHei,Segoe UI,sans-serif;margin:6px 0'>"
                f"pre_extra={pre_extra_disp}：日曆前置 + 上週同 weekday 亦先預測，用於估 g。"
                "低估最多+C、高估最多-C；套用臂：dow_proto / slot_best / ft_xgb_best（raw 不套）。"
                "明細見 curves_pre_with_extra_10min.csv；比賽分見 competition_scores.csv。</p>"
            ),
            width=1100,
        )
    ]
    for i, d in enumerate(target_dates):
        day_blocks.append(_make_day_figure(i + 1, d, y, plot_arms))
        d_norm = pd.Timestamp(d).normalize()
        if d_norm in resid_by_day:
            gap = resid_by_day[d_norm]
            day_blocks.append(_make_resid_figure(i + 1, d, gap))
            resid_blocks.append(_make_resid_figure(i + 1, d, gap))
        # 懲罰圖：各臂用各自 C
        if pen_enable and "dow_proto" in penalty_meta and d_norm in penalty_meta["dow_proto"]:
            for arm_name in ("dow_proto", "slot_best", "ft_xgb_best"):
                if arm_name in penalty_meta and d_norm in penalty_meta[arm_name]:
                    mm = penalty_meta[arm_name][d_norm]
                    arm_c = float(pen_cap_by_arm.get(arm_name, pen_cap))
                    if arm_name == "dow_proto":
                        day_blocks.append(
                            _make_penalty_figure(i + 1, d, mm["g"], mm["p"], arm_c, arm=arm_name)
                        )
                    pen_blocks.append(
                        _make_penalty_figure(i + 1, d, mm["g"], mm["p"], arm_c, arm=arm_name)
                    )

    html_day_path = out_dir / "compare_by_day.html"
    output_file(str(html_day_path), title=f"slot vs ft by day {origin.date()}")
    save(column(*day_blocks))

    html_resid_path = out_dir / "resid_baseline.html"
    output_file(str(html_resid_path), title=f"resid baseline {origin.date()}")
    save(column(*resid_blocks))

    html_pen_path = out_dir / "underest_penalty.html"
    if pen_enable and len(pen_blocks) > 1:
        output_file(str(html_pen_path), title=f"bias_clip {origin.date()}")
        save(column(*pen_blocks))

    print(f"\n輸出: {out_dir}", flush=True)
    print(f"圖(72h): {html_path}", flush=True)
    print(f"圖(3×24h): {html_day_path}", flush=True)
    print(f"圖(殘差): {html_resid_path}", flush=True)
    if pen_enable and (out_dir / "curves_pre_with_extra_10min.csv").exists():
        print(f"pre曲線(含前置日): {out_dir / 'curves_pre_with_extra_10min.csv'}", flush=True)
        print(f"  pre_extra={pre_extra_disp}", flush=True)
    if pen_enable and html_pen_path.exists():
        print(f"圖(雙向偏差): {html_pen_path}", flush=True)
    if args.open:
        webbrowser.open(html_day_path.resolve().as_uri())


if __name__ == "__main__":
    main()
