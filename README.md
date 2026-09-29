# 2026 AI Competition — 台電 10 分鐘負載預測

日前系統負載預測管線（10 分鐘解析度），核心在 [`discrete_dow_cluster/`](discrete_dow_cluster/)：依星期／假日硬分群，產出四臂預測（`dow_proto_raw` / `dow_proto` / `slot_best` / `ft_xgb_best`）。

## 倉庫內容

| 路徑 | 說明 |
|------|------|
| `discrete_dow_cluster/` | 自洽四臂預測管線（`app` / `domain` / `infra`） |
| `update_taipower_load.py` | 爬取台電當日 10 分鐘負載並追加 CSV |
| `discrete_dow_cluster/data/holidays.csv` | 國定假日表（已納入版本庫） |

**本倉庫不包含** 電力負載 CSV、氣象測站 CSV 等大檔；請依下方說明自行下載後放到專案對應路徑。

## 需要自行準備的資料

路徑設定見 [`discrete_dow_cluster/settings.yaml`](discrete_dow_cluster/settings.yaml)。

### 1. 電力負載（必要）

- **檔名**：`台電每10分鐘負載資料.csv`（放在專案根目錄）
- **欄位**：`date_time,load,load_diff`（時間為每 10 分鐘一筆）
- **取得方式**：
  1. 台電「今日電力資訊」：  
     https://www.taipower.com.tw/2289/2363/2367/2368/10262/normalPost  
     （實際資料後端為 `loadGraph` 的 `loadpara.txt` / `loadfueltype.csv`）
  2. 或用本倉庫腳本增量更新：

```bash
python update_taipower_load.py
# 可選：重寫當日列
python update_taipower_load.py --refresh-today
```

> 注意：官網 API 通常只提供**當日**曲線；歷史區間請自行累積或向資料提供單位取得。

### 2. 天氣測站資料（建議，FT-XGB 用）

- **目錄**：`weatherdata2/`
- **格式**：中央氣象署測站月檔，檔名如 `{站號}-{YYYY}-{MM}.csv`（例如 `466920-2026-04.csv`）
- **下載**：[CODiS 氣候觀測資料查詢服務 — 測站資料](https://codis.cwa.gov.tw/StationData)  
  於網站篩選測站與時間後下載；使用時請依氣象署規定註明出處。
- **可選**：同目錄亦可放 `odwo0011-每日逐時氣象觀測資料.csv`（若有）

### 3. 日頻天氣特徵（可選）

- **路徑**：`weather_arima/weather_daily_features.csv`
- **用途**：日均溫等特徵回退（`settings.yaml` → `weather_daily_csv`）
- 可由測站日資料自行彙整產生；缺檔時程式仍可跑，相關欄位可能為空。

### 4. 假日表

- 已提供：`discrete_dow_cluster/data/holidays.csv`  
  欄位含 `Date`、`kind`（`holiday` / `makeup` 等）。

## 快速開始

```bash
cd discrete_dow_cluster
pip install -r requirements.txt

# 確認根目錄已有 台電每10分鐘負載資料.csv，且（建議）有 weatherdata2/
python run_experiment.py --origin 2026-09-10 --horizon-days 3 --skip-xgb-valid
```

細節與模型說明：[discrete_dow_cluster/README.md](discrete_dow_cluster/README.md)、[discrete_dow_cluster/README_模型架構.md](discrete_dow_cluster/README_模型架構.md)。

## 授權與資料出處

- 氣象資料：中央氣象署 CODiS（使用時請註明出處）  
  https://codis.cwa.gov.tw/StationData
- 電力負載：台灣電力公司公開資訊／本腳本抓取之公開曲線
