# Fully Discrete Clustering（星期幾硬分群 + 中位數原型）

本目錄是**完整、可獨立執行**的預測流程（輕量三層：`app` / `domain` / `infra`）：一次產出 **四種預測方法** 的結果，方便互相比較，不依賴其他子專案。

每筆點只屬一群（互斥）：

| cluster_id | 含義 |
|------------|------|
| 0–6 | Mon–Sun |
| 7 | 國定假日 |

**核心原則**：負載與天氣依 weekday 分桶；訓練 / 原型 / slot / fine-tune **絕不跨星期混用**。

完整架構：[README_模型架構.md](README_模型架構.md)。專案總覽與資料下載：[../README.md](../README.md)。

## 目錄結構

```text
discrete_dow_cluster/
  app/          # 應用層：實驗入口、掃 C、出圖
  domain/       # 領域層：分群、預測、校正、評分
  infra/        # 資料層：負載／天氣／假日 I/O
  data/         # holidays.csv
  settings.yaml
  run_experiment.py   # 薄入口
  run_tune_cap.py
  outputs/
```

## 主程式在做什麼

入口：`app/experiment_slot_vs_finetune.py`（讀 [`settings.yaml`](settings.yaml)）。

```text
分開日 24h 預測
  → 四種方法：dow_proto_raw / dow_proto / slot_best / ft_xgb_best
  → ft：同 weekday 近期水準縮放（level_scale）抑季節漂移
  → 除 raw 外的三種再套 bias_clip（forecast_mode=rolling）
       · refs = 日曆 D-1/D-2 ∪ 上週同 weekday；同 weekday 權重較大
       · g = 加權 mean(實際−pre)；p = clip(g,-C,C)；最終 = pre + p
       · 假日目標：refs 限同假日群
       · 輸出比賽 Total_Score + 建議交卷方法（Total 最低者）
```

**定案 C** 見 `settings.yaml` → `cap_by_arm`（可由 `app/tune_underest_cap.py` 跨窗 Total_Score 寫回）。

## 資料如何取得（負載／天氣 CSV 不進 Git）

路徑見 [`settings.yaml`](settings.yaml)。請下載後放在**專案根目錄**相對位置。

### 電力負載（必要）

- 檔案：`../台電每10分鐘負載資料.csv`
- 官網：[台電今日電力資訊](https://www.taipower.com.tw/2289/2363/2367/2368/10262/normalPost)
- 增量更新（專案根目錄）：

```bash
python update_taipower_load.py
```

### 天氣測站（建議，FT-XGB 用）

- 目錄：`../weatherdata2/`（檔名如 `{站號}-{YYYY}-{MM}.csv`）
- 下載：[CODiS 氣候觀測 — 測站資料](https://codis.cwa.gov.tw/StationData)  
  （中央氣象署；使用請註明出處）

### 日頻天氣特徵（可選）

- `../weather_arima/weather_daily_features.csv`（缺檔仍可跑）

### 假日表

- 已納入版本庫：`data/holidays.csv`

## 標準流程

請在本目錄下執行（`cd discrete_dow_cluster`）：

```text
# ① 主實驗（請依需求改 --origin 日期）
python run_experiment.py --origin 2026-09-10 --horizon-days 3 --skip-xgb-valid --open
# 等同：python -m app.experiment_slot_vs_finetune ...

# ② 跨窗掃 C（預設 8/6 + 9/10）並寫回 settings
python run_tune_cap.py --skip-xgb-valid --xgb-k 8
# 等同：python -m app.tune_underest_cap ...
```

| 檔案 | 內容 |
|------|------|
| `compare_by_day.html` | 逐日曲線 + Total_Score + 建議交卷方法 |
| `competition_scores.csv` | 分數比 |
| `level_scale_by_day.csv` | ft 水準縮放 |
| `curves_pre_with_extra_10min.csv` | pre（含 refs） |
| `underest_penalty.html` | g / p / refs |

## 模組對照

| 路徑 | 角色 |
|------|------|
| `infra/data_io.py` | 負載／天氣／假日載入 |
| `domain/scoring.py` | Total_Score + `extract_day` |
| `domain/train_dow_xgb.py` 等 | 四種預測方法的核心 |
| `app/experiment_slot_vs_finetune.py` | 主實驗 |
| `data/holidays.csv` | 國定假日表 |

## 依賴

```text
pip install -r requirements.txt
```
