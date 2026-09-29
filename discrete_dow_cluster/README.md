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
  app/          # 應用層：實驗入口、調校正幅度、出圖
  domain/       # 領域層：分群、預測、校正、評分
  infra/        # 資料層：負載／天氣／假日 I/O
  data/         # holidays.csv
  settings.yaml
  run_experiment.py   # 薄入口
  run_tune_cap.py     # 試不同「校正幅度上限」
  outputs/
```

## 主程式在做什麼

入口：`app/experiment_slot_vs_finetune.py`（讀 [`settings.yaml`](settings.yaml)）。

```text
分開日 24h 預測
  → 四種方法：dow_proto_raw / dow_proto / slot_best / ft_xgb_best
  → ft：用同 weekday 近期水準縮放（level_scale），減輕季節漂移
  → 除 raw 外的三種，再做「近期高低估」雙向校正
       · 先看前幾天：實際比預測偏高還是偏低
       · 把這份偏差加回明天的預測，但幅度有上限（單位 MW）
       · 假日預測時，參考日限同假日群
  → 輸出逐日曲線與誤差比較圖／表
```

### 「校正幅度上限」是什麼？（程式裡常寫成 C）

最後一步會問：最近幾天整體是不是一直低估或高估？若是，就把預測整條往上抬或往下壓一點。

但抬／壓不能無限制，否則改過頭。於是設一個**上限**（單位是 MW）：

- 上限 = 0 → 等於不做這步校正  
- 上限太小 → 修不夠  
- 上限太大 → 可能修過頭  

目前各方法的上限寫在 `settings.yaml` → `cap_by_arm`。  
若要重找合適的數字，可跑 `run_tune_cap.py`：它會**試一串不同上限**（例如 0、800、1600…），在幾個日期窗上比較誤差，再把較好的值寫回設定檔。

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

# ② 試不同校正幅度上限（預設含 8/6、9/10 兩段窗），可寫回 settings
python run_tune_cap.py --skip-xgb-valid --xgb-k 8
# 等同：python -m app.tune_underest_cap ...
```

| 檔案 | 內容 |
|------|------|
| `compare_by_day.html` | 逐日曲線與方法比較 |
| `competition_scores.csv` | 各方法誤差分數表 |
| `level_scale_by_day.csv` | ft 水準縮放 |
| `curves_pre_with_extra_10min.csv` | 校正前曲線（含參考日） |
| `underest_penalty.html` | 近期偏差與校正量說明圖 |

## 模組對照

| 路徑 | 角色 |
|------|------|
| `infra/data_io.py` | 負載／天氣／假日載入 |
| `domain/scoring.py` | 誤差分數 + `extract_day` |
| `domain/train_dow_xgb.py` 等 | 四種預測方法的核心 |
| `app/experiment_slot_vs_finetune.py` | 主實驗 |
| `app/tune_underest_cap.py` | 試不同校正幅度上限 |
| `data/holidays.csv` | 國定假日表 |

## 依賴

```text
pip install -r requirements.txt
```
