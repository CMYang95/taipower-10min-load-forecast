"""掃雙向偏差校正上限 C（bias_clip；settings 鍵名 underest_penalty），用競賽 Total Score 選最佳 C。

主實驗已讀 settings.yaml 定案 C；本腳本用於換窗／重調時掃描。
臂：dow_proto_raw（不套，每窗 1 分）；dow_proto / slot_best / ft_xgb_best 掃 C。

  python -m app.tune_underest_cap --caps 0,800,1600,2400,2800,3200 --skip-xgb-valid --xgb-k 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 套件根（settings.yaml）
PKG_ROOT = Path(__file__).resolve().parent.parent


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


from domain.cluster_assign import assign_cluster  # noqa: E402
from domain.daily_predict import (  # noqa: E402
    apply_slot_separate_days,
    predict_dow_proto_resid_separate_days,
    predict_separate_days,
    roll_daily_proto_curves,
    select_slot_on_valid_daily,
)
from domain.finetune import finetune_xgb_separate_days  # noqa: E402
from domain.scoring import extract_day, score_frame  # noqa: E402
from domain.slot_adapt import (  # noqa: E402
    apply_level_scale_same_dow,
    apply_underest_penalty_separate_days,
    collect_bias_pre_extra,
    day_slice,
)
from domain.train_dow_xgb import load_common_data, load_settings, resolve_paths  # noqa: E402


TUNABLE_ARMS = ("dow_proto", "slot_best", "ft_xgb_best")


def _series_to_day_frame(s: pd.Series, day: pd.Timestamp) -> pd.DataFrame:
    part = day_slice(s, day)
    if part.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"date_time": part.index, "load": part.to_numpy(dtype=float)})
    df["date"] = pd.Timestamp(day).normalize()
    df["tod"] = df["date_time"].dt.strftime("%H:%M")
    df["load_diff"] = df["load"].diff()
    return df


def labels_from_series(s: pd.Series, dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for d in dates:
        day_df = _series_to_day_frame(s, d)
        if day_df.empty:
            continue
        rec = extract_day(day_df)
        if rec is not None:
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def score_curves(
    actual: pd.Series,
    pred: pd.Series,
    dates: list[pd.Timestamp],
) -> dict[str, float]:
    true_df = labels_from_series(actual, dates)
    pred_df = labels_from_series(pred, dates)
    if true_df.empty or pred_df.empty:
        return {
            "n_days": 0.0,
            "S_peak_mw": float("nan"),
            "S_peak_time": float("nan"),
            "S_ramp_up": float("nan"),
            "S_ramp_down": float("nan"),
            "S_under_penalty": float("nan"),
            "Total_Score": float("nan"),
        }
    pred_df = pred_df.rename(
        columns={
            "Peak_MW_Day": "Peak_MW_Day_pred",
            "Peak_MW_Night": "Peak_MW_Night_pred",
            "Peak_Time_Day": "Peak_Time_Day_pred",
            "Peak_Time_Night": "Peak_Time_Night_pred",
            "Max_Ramp_Up_MW": "Max_Ramp_Up_MW_pred",
            "Max_Ramp_Down_MW": "Max_Ramp_Down_MW_pred",
        }
    )
    keep = ["Date"] + [c for c in pred_df.columns if c.endswith("_pred")]
    return score_frame(true_df, pred_df[keep])


def _pre_dates(
    origin: pd.Timestamp,
    target_dates: list[pd.Timestamp],
    pen_lb: int,
    idx_days: set,
    *,
    include_prev_same_dow: bool = True,
    forecast_mode: str = "rolling",
) -> list[pd.Timestamp]:
    extra = collect_bias_pre_extra(
        origin,
        target_dates,
        pen_lb,
        include_prev_same_dow=include_prev_same_dow,
        idx_days=idx_days,
        forecast_mode=forecast_mode,
    )
    return extra + list(target_dates)


def run_window(
    *,
    origin: pd.Timestamp,
    horizon_days: int,
    caps: list[float],
    load10: pd.Series,
    holidays: pd.DataFrame,
    actual: pd.Series,
    weather: pd.DataFrame,
    idx_days: set,
    gate_valid_start: pd.Timestamp,
    gate_test_start: pd.Timestamp,
    freeze_lb: int,
    freeze_beta: float,
    freeze_k_xgb: int,
    max_days: int,
    proto_resid_lb: int,
    proto_resid_beta: float,
    pen_lb: int,
    same_dow_only: bool,
    xgb_train_days: int,
    min_samples: int,
    xgb_cfg: dict,
    weather_lag: int,
    pen_prev_dow: bool = True,
    pen_same_w: float = 2.0,
    pen_cal_w: float = 1.0,
    pen_gap_mode: str = "peak",
    pen_peak_blend: float = 0.65,
    pen_weekend_uplift: bool = True,
    forecast_mode: str = "rolling",
    holiday_cid: int = 7,
    day_clusters: pd.Series | None = None,
    lvl_enable: bool = True,
    lvl_k: int = 2,
    lvl_lo: float = 0.85,
    lvl_hi: float = 1.15,
) -> list[dict]:
    target_dates = [origin + pd.Timedelta(days=i) for i in range(horizon_days)]
    window = f"{origin.strftime('%Y%m%d')}_{(origin + pd.Timedelta(days=horizon_days - 1)).strftime('%Y%m%d')}"
    print(f"\n===== WINDOW {window} =====", flush=True)

    pre_dates = _pre_dates(
        origin,
        target_dates,
        pen_lb,
        idx_days,
        include_prev_same_dow=pen_prev_dow,
        forecast_mode=forecast_mode,
    )
    print(f"  pre_dates={[str(d.date()) for d in pre_dates]}", flush=True)

    print("  -- pre dow_proto_raw --", flush=True)
    pre_raw = predict_separate_days(
        load10=load10,
        holidays=holidays,
        target_dates=target_dates,
        max_days_per_cluster=max_days,
        quiet=True,
    )

    print("  -- pre dow_proto (resid) --", flush=True)
    pre_dow, _ = predict_dow_proto_resid_separate_days(
        actual=actual,
        load10=load10,
        holidays=holidays,
        target_dates=pre_dates,
        max_days_per_cluster=max_days,
        idx_days=idx_days,
        valid_start=gate_valid_start,
        lookback_days=proto_resid_lb,
        beta=proto_resid_beta,
        quiet=True,
    )

    print("  -- pre slot_best --", flush=True)
    pre_slot, _ = apply_slot_separate_days(
        actual=actual,
        load10=load10,
        holidays=holidays,
        target_dates=pre_dates,
        max_days_per_cluster=max_days,
        lookback_days=freeze_lb,
        beta=freeze_beta,
        idx_days=idx_days,
        valid_start=gate_valid_start,
        same_dow_only=same_dow_only,
        quiet=True,
    )

    print(f"  -- pre ft_xgb_best K={freeze_k_xgb} --", flush=True)
    pre_xgb = finetune_xgb_separate_days(
        load10=load10,
        holidays=holidays,
        weather=weather,
        target_dates=pre_dates,
        k_same_dow=freeze_k_xgb,
        max_days_per_cluster=max_days,
        xgb_train_days=xgb_train_days,
        min_samples=min_samples,
        xgb_params=xgb_cfg,
        weather_lag_days=weather_lag,
        quiet=True,
    )
    if lvl_enable:
        pre_xgb, _ = apply_level_scale_same_dow(
            actual,
            pre_xgb,
            target_dates,
            k_same_dow=lvl_k,
            clip_lo=lvl_lo,
            clip_hi=lvl_hi,
            idx_days=idx_days,
        )

    rows: list[dict] = []

    # raw：不套抬升
    sc_raw = score_curves(actual, pre_raw, target_dates)
    rows.append(
        {
            "window": window,
            "origin": str(origin.date()),
            "arm": "dow_proto_raw",
            "cap_mw": float("nan"),
            "tunable": False,
            **sc_raw,
        }
    )
    print(
        f"  dow_proto_raw  Total={sc_raw['Total_Score']:.6f}  "
        f"under={sc_raw['S_under_penalty']:.6f}",
        flush=True,
    )

    pre_by_arm = {
        "dow_proto": pre_dow,
        "slot_best": pre_slot,
        "ft_xgb_best": pre_xgb,
    }
    for arm, pre in pre_by_arm.items():
        for c in caps:
            final, _ = apply_underest_penalty_separate_days(
                actual,
                pre,
                target_dates,
                lookback_days=pen_lb,
                cap_mw=float(c),
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
            sc = score_curves(actual, final, target_dates)
            rows.append(
                {
                    "window": window,
                    "origin": str(origin.date()),
                    "arm": arm,
                    "cap_mw": float(c),
                    "tunable": True,
                    **sc,
                }
            )
            print(
                f"  {arm} C={c:.0f}  Total={sc['Total_Score']:.6f}  "
                f"under={sc['S_under_penalty']:.6f}",
                flush=True,
            )
    return rows


def main() -> None:
    _configure_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", default="2026-08-06,2026-09-10")
    ap.add_argument("--horizon-days", type=int, default=3)
    ap.add_argument("--caps", default="0,800,1600,2400,2800,3200")
    ap.add_argument("--write-settings", action="store_true", default=True, help="把跨窗最佳 C 寫回 settings.yaml")
    ap.add_argument("--no-write-settings", action="store_false", dest="write_settings")
    ap.add_argument("--skip-xgb-valid", action="store_true", default=True)
    ap.add_argument("--no-skip-xgb-valid", action="store_false", dest="skip_xgb_valid")
    ap.add_argument("--xgb-k", type=int, default=8, help="skip valid 時的預設 K")
    args = ap.parse_args()

    origins = [pd.Timestamp(x.strip()).normalize() for x in str(args.origins).split(",") if x.strip()]
    caps = [float(x.strip()) for x in str(args.caps).split(",") if x.strip()]
    horizon_days = int(args.horizon_days)

    cfg = load_settings()
    paths = resolve_paths(cfg)
    dow_cfg = cfg.get("dow") or {}
    splits = cfg.get("splits") or {}
    feat_cfg = cfg.get("features") or {}
    xgb_cfg = cfg.get("xgb") or {}

    gate_valid_start = pd.Timestamp(str(splits.get("gate_valid_start", "2026-07-01")))
    gate_test_start = pd.Timestamp(str(splits.get("gate_test_start", "2026-08-01")))
    lookbacks = list(dow_cfg.get("slot_lookbacks") or list(range(1, 8)))
    betas = list(dow_cfg.get("slot_betas") or [0.3, 0.5, 0.7, 1.0])
    same_dow_only = bool(dow_cfg.get("slot_same_dow_only", True))
    max_days = int(dow_cfg.get("proto_max_days", 16))
    proto_resid_lb = int(dow_cfg.get("proto_resid_lookback", 2))
    proto_resid_beta = float(dow_cfg.get("proto_resid_beta", 1.0))
    pen_cfg = dow_cfg.get("underest_penalty") or {}
    pen_lb = int(pen_cfg.get("lookback_days", 2))
    pen_prev_dow = bool(pen_cfg.get("include_prev_same_dow", True))
    pen_same_w = float(pen_cfg.get("same_dow_weight", 2.0))
    pen_cal_w = float(pen_cfg.get("calendar_weight", 1.0))
    pen_gap_mode = str(pen_cfg.get("gap_mode", "peak")).lower()
    pen_peak_blend = float(pen_cfg.get("peak_blend", 0.65))
    pen_weekend_uplift = bool(pen_cfg.get("weekend_uplift_only", True))
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
    ft_ks = list(dow_cfg.get("finetune_lookbacks") or list(range(1, 8)))

    tag = "_".join(o.strftime("%Y%m%d") for o in origins)
    out_dir = Path(paths["output_dir"]) / f"tune_cap_{tag}"
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
    idx_days = set(pd.DatetimeIndex(load10.dropna().index).normalize())

    # Gate：slot 選參（兩窗共用）
    max_lb = max(lookbacks) if lookbacks else 7
    gate_origins = [
        pd.Timestamp(d)
        for d in pd.date_range(
            gate_valid_start - pd.Timedelta(days=max_lb * 8),
            gate_test_start - pd.Timedelta(days=1),
            freq="D",
        )
        if pd.Timestamp(d).normalize() in idx_days
    ]
    print("===== GATE slot =====", flush=True)
    gate_curves = roll_daily_proto_curves(
        load10=load10,
        holidays=holidays,
        origins=gate_origins,
        max_days_per_cluster=max_days,
        valid_start=gate_valid_start,
        test_start=gate_test_start,
        quiet=False,
    )
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

    if args.skip_xgb_valid:
        freeze_k_xgb = int(args.xgb_k) if args.xgb_k else (8 if 8 in ft_ks else (int(max(ft_ks)) if ft_ks else 8))
        print(f"FT-xgb skip valid，K={freeze_k_xgb}", flush=True)
    else:
        from domain.finetune import select_finetune_on_valid

        valid_days = [
            pd.Timestamp(d)
            for d in pd.date_range(gate_valid_start, gate_test_start - pd.Timedelta(days=1), freq="D")
            if pd.Timestamp(d).normalize() in idx_days
        ]
        ft_xgb_best, _ = select_finetune_on_valid(
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
        freeze_k_xgb = int(ft_xgb_best["k_same_dow"])
        print(f"凍結 FT-xgb K={freeze_k_xgb}", flush=True)

    meta = {
        "origins": [str(o.date()) for o in origins],
        "horizon_days": horizon_days,
        "caps": caps,
        "slot": {"lookback_days": freeze_lb, "beta": freeze_beta},
        "ft_xgb_k": freeze_k_xgb,
        "pen_lookback_days": pen_lb,
        "proto_resid": {"lookback_days": proto_resid_lb, "beta": proto_resid_beta},
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_rows: list[dict] = []
    for origin in origins:
        all_rows.extend(
            run_window(
                origin=origin,
                horizon_days=horizon_days,
                caps=caps,
                load10=load10,
                holidays=holidays,
                actual=actual,
                weather=weather,
                idx_days=idx_days,
                gate_valid_start=gate_valid_start,
                gate_test_start=gate_test_start,
                freeze_lb=freeze_lb,
                freeze_beta=freeze_beta,
                freeze_k_xgb=freeze_k_xgb,
                max_days=max_days,
                proto_resid_lb=proto_resid_lb,
                proto_resid_beta=proto_resid_beta,
                pen_lb=pen_lb,
                same_dow_only=same_dow_only,
                xgb_train_days=xgb_train_days,
                min_samples=min_samples,
                xgb_cfg=xgb_cfg,
                weather_lag=weather_lag,
                pen_prev_dow=pen_prev_dow,
                pen_same_w=pen_same_w,
                pen_cal_w=pen_cal_w,
                pen_gap_mode=pen_gap_mode,
                pen_peak_blend=pen_peak_blend,
                pen_weekend_uplift=pen_weekend_uplift,
                forecast_mode=forecast_mode,
                holiday_cid=holiday_cid,
                day_clusters=day_clusters,
                lvl_enable=lvl_enable,
                lvl_k=lvl_k,
                lvl_lo=lvl_lo,
                lvl_hi=lvl_hi,
            )
        )

    scores = pd.DataFrame(all_rows)
    scores.to_csv(out_dir / "scores_all.csv", index=False, encoding="utf-8-sig")

    # 每窗每臂最佳 C
    best_rows = []
    for (window, arm), g in scores.groupby(["window", "arm"], sort=False):
        if arm == "dow_proto_raw":
            r = g.iloc[0]
            best_rows.append(
                {
                    "window": window,
                    "arm": arm,
                    "best_cap_mw": float("nan"),
                    "Total_Score": r["Total_Score"],
                    "S_peak_mw": r["S_peak_mw"],
                    "S_peak_time": r["S_peak_time"],
                    "S_ramp_up": r["S_ramp_up"],
                    "S_ramp_down": r["S_ramp_down"],
                    "S_under_penalty": r["S_under_penalty"],
                }
            )
            continue
        g2 = g.dropna(subset=["Total_Score"]).sort_values("Total_Score")
        if g2.empty:
            continue
        r = g2.iloc[0]
        best_rows.append(
            {
                "window": window,
                "arm": arm,
                "best_cap_mw": r["cap_mw"],
                "Total_Score": r["Total_Score"],
                "S_peak_mw": r["S_peak_mw"],
                "S_peak_time": r["S_peak_time"],
                "S_ramp_up": r["S_ramp_up"],
                "S_ramp_down": r["S_ramp_down"],
                "S_under_penalty": r["S_under_penalty"],
            }
        )
    best_per_arm = pd.DataFrame(best_rows)
    best_per_arm.to_csv(out_dir / "best_per_arm.csv", index=False, encoding="utf-8-sig")

    # 跨窗：可調臂選使兩窗 Total 平均最低的單一 C
    overall = []
    for arm in TUNABLE_ARMS:
        sub = scores[scores["arm"] == arm].copy()
        if sub.empty:
            continue
        pivot = (
            sub.groupby("cap_mw", as_index=False)["Total_Score"]
            .mean()
            .sort_values("Total_Score")
        )
        if pivot.empty:
            continue
        best_c = float(pivot.iloc[0]["cap_mw"])
        mean_total = float(pivot.iloc[0]["Total_Score"])
        by_win = sub[sub["cap_mw"] == best_c][["window", "Total_Score"]].copy()
        overall.append(
            {
                "arm": arm,
                "best_cap_mw": best_c,
                "mean_Total_Score": mean_total,
                "n_windows": int(by_win["window"].nunique()),
                "detail": "; ".join(
                    f"{r.window}:{r.Total_Score:.6f}" for r in by_win.itertuples(index=False)
                ),
            }
        )
    # raw 兩窗平均（無對照）
    raw = scores[scores["arm"] == "dow_proto_raw"]
    if not raw.empty:
        overall.append(
            {
                "arm": "dow_proto_raw",
                "best_cap_mw": float("nan"),
                "mean_Total_Score": float(raw["Total_Score"].mean()),
                "n_windows": int(raw["window"].nunique()),
                "detail": "; ".join(
                    f"{r.window}:{r.Total_Score:.6f}" for r in raw.itertuples(index=False)
                ),
            }
        )
    best_overall = pd.DataFrame(overall)
    best_overall.to_csv(out_dir / "best_overall.csv", index=False, encoding="utf-8-sig")

    print("\n=== best_per_arm ===", flush=True)
    print(best_per_arm.to_string(index=False), flush=True)
    print("\n=== best_overall（跨窗平均 Total 最低的單一 C）===", flush=True)
    print(best_overall.to_string(index=False), flush=True)
    print(f"\n輸出: {out_dir}", flush=True)

    if args.write_settings:
        settings_path = PKG_ROOT / "settings.yaml"
        text = settings_path.read_text(encoding="utf-8")
        for arm in TUNABLE_ARMS:
            row = best_overall[best_overall["arm"] == arm]
            if row.empty or not np.isfinite(row.iloc[0]["best_cap_mw"]):
                continue
            c_val = int(round(float(row.iloc[0]["best_cap_mw"])))
            # 替換 cap_by_arm 下對應行
            import re

            text = re.sub(
                rf"({arm}:\s*)\d+",
                rf"\g<1>{c_val}",
                text,
                count=1,
            )
        settings_path.write_text(text, encoding="utf-8")
        print(f"已寫回 {settings_path} → dow.underest_penalty.cap_by_arm", flush=True)
        print(best_overall[best_overall["arm"].isin(TUNABLE_ARMS)][["arm", "best_cap_mw", "mean_Total_Score"]].to_string(index=False), flush=True)
    else:
        print("定案後請把最佳 C 寫回 settings.yaml → dow.underest_penalty.cap_by_arm", flush=True)


if __name__ == "__main__":
    main()
