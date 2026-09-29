"""爬取台電「今日電力資訊」當日每 10 分鐘系統負載，追加寫入 CSV。

資料來源（與官網圖表相同後端，不爬 Cloudflare HTML）：
  https://www.taipower.com.tw/d006/loadGraph/loadGraph/data/loadpara.txt
  https://www.taipower.com.tw/d006/loadGraph/loadGraph/data/loadfueltype.csv

官網頁：
  https://www.taipower.com.tw/2289/2363/2367/2368/10262/normalPost

單位：圖表 CSV 為「萬瓩」→ 轉成 MW（×10）。
僅能取得「當天」曲線；歷史缺口不會被補上。

用法：
  python update_taipower_load.py
  python update_taipower_load.py --csv "台電每10分鐘負載資料.csv"
  python update_taipower_load.py --refresh-today   # 重寫當日列（含小數兩位）
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE = "https://www.taipower.com.tw/d006/loadGraph/loadGraph/data"
DEFAULT_CSV = Path(__file__).resolve().parent / "台電每10分鐘負載資料.csv"


def _get(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; taipower-load-update/1.0)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_loadpara(text: str) -> dict:
    """解析 loadpara.txt：目前負載、日期等。"""
    out: dict = {}
    m = re.search(r"var loadInfo\s*=\s*\[(.*?)\];", text, re.S)
    if m:
        vals = re.findall(r'"([^"]*)"', m.group(1))
        if len(vals) >= 4:
            out["curr_load_mw"] = float(vals[0].replace(",", ""))
            out["fore_peak_mw"] = float(vals[1].replace(",", ""))
            out["supply_mw"] = float(vals[2].replace(",", ""))
            out["update_label"] = vals[3]
    m2 = re.search(r"(\d{3})\.(\d{2})\.(\d{2})", out.get("update_label", "") or text)
    if m2:
        y = int(m2.group(1)) + 1911
        out["date"] = pd.Timestamp(year=y, month=int(m2.group(2)), day=int(m2.group(3))).normalize()
    m3 = re.search(r"<!--\s*(\d+)-(\d+)-(\d+)\s+(\d+):(\d+)", text)
    if m3:
        y = int(m3.group(1)) + 1911
        if "date" not in out:
            out["date"] = pd.Timestamp(year=y, month=int(m3.group(2)), day=int(m3.group(3))).normalize()
        out["update_hm"] = f"{int(m3.group(4)):02d}:{int(m3.group(5)):02d}"
    return out


def _parse_tod(raw: str) -> str | None:
    s = str(raw).strip()
    if not s or s == ",":
        return None
    if re.fullmatch(r"\d{1,2}", s):
        return f"{int(s):02d}:00"
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        hh, mm = s.split(":")
        return f"{int(hh):02d}:{int(mm):02d}"
    return None


def parse_fueltype_csv(raw: bytes, day: pd.Timestamp) -> pd.DataFrame:
    text = raw.decode("utf-8", errors="replace").strip()
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.strip() == ",":
            continue
        parts = [p.strip() for p in line.split(",")]
        tod = _parse_tod(parts[0])
        if tod is None:
            continue
        nums = []
        for p in parts[1:]:
            if p == "" or p is None:
                continue
            try:
                nums.append(float(p))
            except ValueError:
                break
        if not nums:
            continue
        total_wan = float(np.nansum(nums))
        rows.append({"tod": tod, "load": round(total_wan * 10.0, 2)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date_time"] = pd.to_datetime(day.strftime("%Y-%m-%d") + " " + df["tod"])
    return df.drop_duplicates("date_time").sort_values("date_time").reset_index(drop=True)


def format_date_time(ts: pd.Timestamp) -> str:
    """對齊既有 CSV：YYYY/M/D HH:MM（月日不補零）。"""
    t = pd.Timestamp(ts)
    return f"{t.year}/{t.month}/{t.day} {t.hour:02d}:{t.minute:02d}"


def format_load(val: float) -> str:
    """固定保留小數點後兩位（與歷史 CSV 一致，例如 28867.22）。"""
    return f"{round(float(val), 2):.2f}"


def last_csv_timestamp(csv_path: Path) -> pd.Timestamp | None:
    if not csv_path.exists():
        return None
    # 大檔只讀尾端數行即可
    with csv_path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", errors="replace")
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    if not lines:
        return None
    # 跳過可能殘缺的第一行
    for line in reversed(lines):
        if line.startswith("date_time"):
            continue
        part = line.split(",", 1)[0].strip()
        try:
            return pd.to_datetime(part)
        except (ValueError, TypeError):
            continue
    return None


def scrape_today() -> tuple[pd.DataFrame, dict]:
    para_raw = _get(f"{BASE}/loadpara.txt")
    fuel_raw = _get(f"{BASE}/loadfueltype.csv")
    meta = parse_loadpara(para_raw.decode("utf-8", errors="replace"))
    day = meta.get("date") or pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
    fuel = parse_fueltype_csv(fuel_raw, day)
    meta = {**meta, "date": day}
    return fuel, meta


def truncate_csv_before(csv_path: Path, cutoff: pd.Timestamp) -> None:
    """刪除 cutoff（含）之後的列，保留檔頭與更早資料。"""
    if not csv_path.exists():
        return
    keep: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for i, line in enumerate(f):
            raw = line.rstrip("\r\n")
            if i == 0:
                keep.append(raw)
                continue
            if not raw.strip():
                continue
            part = raw.split(",", 1)[0].strip()
            try:
                ts = pd.to_datetime(part)
            except (ValueError, TypeError):
                keep.append(raw)
                continue
            if ts < cutoff:
                keep.append(raw)
    text = "\n".join(keep) + "\n"
    try:
        csv_path.write_text(text, encoding="utf-8", newline="\n")
    except PermissionError:
        new_path = csv_path.with_name(csv_path.stem + ".new" + csv_path.suffix)
        new_path.write_text(text, encoding="utf-8", newline="\n")
        try:
            new_path.replace(csv_path)
        except PermissionError as e:
            raise SystemExit(
                f"無法覆寫 {csv_path}（檔案被鎖定）。已寫入：{new_path}"
            ) from e


def append_new_rows(csv_path: Path, live: pd.DataFrame, last_ts: pd.Timestamp | None) -> pd.DataFrame:
    if live.empty:
        return live.iloc[0:0]
    new = live.copy()
    if last_ts is not None:
        new = new[new["date_time"] > last_ts]
    return new.reset_index(drop=True)


def _rows_to_csv_lines(new_rows: pd.DataFrame) -> str:
    lines = []
    for _, row in new_rows.iterrows():
        dt = format_date_time(row["date_time"])
        load = format_load(row["load"])
        lines.append(f"{dt},{load},")
    return "\n".join(lines) + "\n"


def write_append(csv_path: Path, new_rows: pd.DataFrame) -> Path:
    """追加寫入；若目標被鎖定則寫完整檔到 *.new.csv 並嘗試取代。回傳實際寫入路徑。"""
    if new_rows.empty:
        return csv_path
    payload = _rows_to_csv_lines(new_rows)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.write_text("date_time,load,load_diff\n" + payload, encoding="utf-8", newline="\n")
        return csv_path

    try:
        need_nl = False
        if csv_path.stat().st_size > 0:
            with csv_path.open("rb") as f:
                f.seek(-1, 2)
                need_nl = f.read(1) not in (b"\n", b"\r")
        with csv_path.open("a", encoding="utf-8", newline="\n") as f:
            if need_nl:
                f.write("\n")
            f.write(payload)
        return csv_path
    except PermissionError:
        pass

    # 檔案被鎖定（常見於 Cursor/Excel 開啟中）：寫完整更新檔再 replace
    old = csv_path.read_text(encoding="utf-8")
    if old and not old.endswith("\n"):
        old += "\n"
    new_path = csv_path.with_name(csv_path.stem + ".new" + csv_path.suffix)
    new_path.write_text(old + payload, encoding="utf-8", newline="\n")
    try:
        new_path.replace(csv_path)
        return csv_path
    except PermissionError:
        print(
            f"警告：無法覆寫 {csv_path}（檔案被其他程式鎖定，例如 Excel）。\n"
            f"已寫入完整更新檔：{new_path}\n"
            f"請關閉該檔後執行：Move-Item -Force \"{new_path}\" \"{csv_path}\"",
            flush=True,
        )
        return new_path


def main() -> None:
    ap = argparse.ArgumentParser(description="爬取台電當日 10 分鐘負載並追加 CSV")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="目標 CSV 路徑")
    ap.add_argument(
        "--refresh-today",
        action="store_true",
        help="重寫當日已存在列（用於修正小數格式等）",
    )
    args = ap.parse_args()
    csv_path = Path(args.csv)

    print("爬取台電 loadGraph 資料…", flush=True)
    live, meta = scrape_today()
    day = pd.Timestamp(meta["date"]).normalize()
    if live.empty:
        raise SystemExit("爬到的用電曲線為空")

    if args.refresh_today:
        print(f"重寫當日資料：刪除 {day.date()} 起既有列後再寫入…", flush=True)
        truncate_csv_before(csv_path, day)
        last_ts = last_csv_timestamp(csv_path)
        new_rows = live.copy()
    else:
        last_ts = last_csv_timestamp(csv_path)
        new_rows = append_new_rows(csv_path, live, last_ts)

    written = write_append(csv_path, new_rows)

    print(
        f"日期 {day.date()}  曲線點數={len(live)}  "
        f"官網目前負載={meta.get('curr_load_mw', '—')} MW  "
        f"更新={meta.get('update_hm', meta.get('update_label', '—'))}",
        flush=True,
    )
    print(f"CSV 原最後時間: {last_ts}", flush=True)
    if new_rows.empty:
        print("無需追加（沒有比 CSV 更新的時段）", flush=True)
    else:
        print(
            f"已寫入 {len(new_rows)} 筆："
            f"{format_date_time(new_rows['date_time'].iloc[0])} → "
            f"{format_date_time(new_rows['date_time'].iloc[-1])}",
            flush=True,
        )
        curve_end = float(live["load"].iloc[-1])
        curr = meta.get("curr_load_mw")
        if curr is not None:
            print(f"曲線末端={curve_end:.2f} MW  vs 官網={float(curr):.2f} MW", flush=True)
    print(f"寫入: {written}", flush=True)


if __name__ == "__main__":
    main()
