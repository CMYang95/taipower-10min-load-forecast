# 預測模型架構說明（白話版）

這份文件說明 `discrete_dow_cluster` **目前實際在跑的模型**：怎麼分群、怎麼一天一天預測、**四種預測方法**差在哪，以及最後的**雙向偏差校正**怎麼加。

| 項目 | 位置 |
|------|------|
| 主實驗腳本 | `app/experiment_slot_vs_finetune.py` |
| 設定檔 | `settings.yaml`（`dow.*`、`splits.*`、天氣 lag） |
| 誤差評分 | `domain/scoring.py` |
| 四種方法 | `dow_proto_raw` → `dow_proto` → `slot_best` → `ft_xgb_best` |

---

## 一句話總覽

```text
依「星期幾／假日」分群
  → 每天獨立預測 24 小時（144 個 10 分鐘點）
  → 先貼「同群歷史中位數曲線」當底
  → 各方法再加不同校正（殘差 / slot / XGB）
  → 除 raw 外的三種，再做「近期高低估」雙向校正
       （ft 會先做 level_scale；校正幅度有上限；raw 不加這步）
  → 輸出曲線與誤差比較
```

**沒有**把週五 23:50 的預測值連續接到週六 00:00；跨日傳遞的是「前幾天的實際誤差資訊」，不是上一預測點的數值。

---

## 1. 共同規則（四種方法都遵守）

### 1.1 硬分群：一天只屬一群

每筆 10 分鐘負載點只進一群（互斥）：

| cluster | 意思 | 怎麼定 |
|---------|------|--------|
| 0–6 | 週一～週日 | 依當天星期幾 |
| 7 | 國定假日 | `holidays.csv` 裡 `kind=holiday` |

- 國定假日整天進 cluster 7，不再算星期幾。
- **補班日**不當假日，仍依星期幾分群。
- 建原型、slot、XGB 訓練時，**原則上不跨群混用**資料。
- 例外：`dow_proto` 的日曆殘差、以及雙向偏差校正，會看**日曆近兩天**（可跨星期，例如預測週六會用到週五）。

### 1.2 分開日預測：一天做一次

評測窗裡，每一天 **D** 都重跑一遍：

1. 訓練截止點 = 當天 0:00（只用比 D 更早的歷史）
2. 只預測當天 24 小時
3. 多天結果再串起來（例如 8/6、8/7、8/8 各預測一次，不是一次連預 72 小時）

因此：預測 8/7 時，可以用到 8/6 整天的**實際**負載——這是**日前滾動**，不是「早上一次看三天」。

週五接週六時：兩天用不同群的原型，午夜**沒有強制平滑**；圖上可能出現階躍，屬預期行為。

### 1.3 Valid / Test

| 用途 | 預設區間 |
|------|----------|
| Valid set | 2026-07-01～07-31 |
| Test set | 2026-08-01 起 |

`slot_best`、`ft_xgb_best` 的參數建議在 valid 選好後固定，不要再依 test 重挑。

### 1.4 共同積木：中位數日曲線

對某個星期幾／假日群：

1. 在截止日之前，找出該群出現過的歷史日（由近到遠）
2. 取最近 N 天（一般 N=16；FT-XGB 預設 **K=8**）
3. 每個時段（一天 144 格）對這 N 天取**中位數** → 得到一條「典型日曲線」
4. 預測當天：看當天屬於哪一群，把該群曲線整條貼上去

這就是所有方法的「底板」。

---

## 2. 四種方法長什麼樣

```text
dow_proto_raw     只用同群 16 日中位數（對照用，不加校正）
       ↓
dow_proto         同上 + 日曆近兩天「實際 − 原型」再抬／壓一點
       ↓          + 雙向偏差校正（目前上限=0 → 這步關閉）
slot_best         同上底板 + 同星期幾歷史誤差 × β
       ↓          + 雙向偏差校正（目前上限=3200 MW）
ft_xgb_best       同群近 K=8 日中位數 + XGB 殘差（天氣等）
                  + level_scale + 雙向偏差校正（目前上限=3200 MW）
```

| 方法 | 底板 | 額外校正 | 雙向偏差校正 | 超參 |
|----|------|----------|-----------|------|
| `dow_proto_raw` | 同群 16 日中位數 | 無 | **否** | 固定 |
| `dow_proto` | 同群 16 日中位數 | 日曆 D−1、D−2 殘差 | **是**（上限=0） | 固定 lookback=2、β=1 |
| `slot_best` | 同群 16 日中位數 | 同群最近 L 日殘差 × β | **是**（上限=3200） | valid 選 L、β（目前 L=6、β=1.0） |
| `ft_xgb_best` | 同群最近 **K=8** 日中位數 | 同群 XGB 殘差 + level_scale | **是**（上限=3200） | skip 預設 K=8；valid 可試 1–8 |

四種方法是**互斥對照**，不是串成一條路：`slot_best` 不做 XGB；`ft_xgb_best` 不做 slot。

---

## 3. 方法流程

### 3.1 `dow_proto_raw`：純原型

每天：取當天那一群、最近 16 天的中位數曲線，直接當預測。  
沒有天氣、沒有殘差、沒有雙向偏差校正。  
用途：看「光貼歷史形狀」會偏多少（實務上常整條偏低）。

### 3.2 `dow_proto`：原型 + 近兩天水位修正

1. 先做出跟 raw 一樣的中位數曲線。
2. 回頭看**日曆**昨天、前天（可跨星期）：
   - 用「當時若只貼原型」會怎麼錯
   - 每個時段算：實際 − 原型
3. 兩天對齊取平均，得到「近期每個時段大概低估／高估多少」。
4. 整條加回去（係數 β 預設 1，等於全加）。
5. 再套雙向偏差校正（目前上限=0，等於關掉）。

**例子**（預測 8/6～8/8）：

| 預測日 | 看誰的誤差 |
|--------|------------|
| 8/6 | 8/4、8/5 |
| 8/7 | 8/5、8/6 |
| 8/8 | 8/6、8/7 |

和 `slot_best` 的差別：這裡修的是**最近兩天整體水位**；slot 修的是**同一個星期幾**的歷史習慣偏差。

### 3.3 `slot_best`：同星期幾的習慣誤差

1. Valid 上試「回看幾個同群日 L」×「加多少 β」，選 MAE 較好的組合並固定。
2. Test 每天：
   - 先貼 16 日中位數底板
   - 找預測日之前、最近 L 個**同一個星期幾／假日群**的日子
   - 那些日子上：實際 −（當時的）原型 → 每個時段平均
   - 預測 = 底板 + β × 這條誤差曲線
3. 再套雙向偏差校正（目前上限=3200 MW）。

預設只回看同群（`slot_same_dow_only: true`），所以預測週六主要看歷史週六，不看週五。

### 3.4 `ft_xgb_best`：短窗原型 + XGB 補殘差

白話：用最近 K 個同群日畫基準曲線，再訓練 XGB 學「基準還差實際多少」，把差補回去。**不做 slot。**

1. **基準**：同群最近 K 日中位數（`--skip-xgb-valid` 預設 **K=8**；否則 valid 可試 `finetune_lookbacks` 1–8）。
2. **標籤**：實際 − 基準（學殘差，不是直接學負載）。
3. **特徵大概有**：一天裡第幾個時段、基準值、**前一天**同時段天氣、上一同群同時段負載等（不用預測當天的天氣預報）。
4. **pre** = 基準 + XGB 估的殘差（樣本太少則退回只用基準）。
5. **level_scale**：同 weekday 近期實際水準 / 當日 pre 水準，縮放後限制在 `[0.85, 1.15]`。
6. **雙向偏差校正**：先估近期整體高低估 `g`，再把校正量限制在「±上限」內後加回。

---

## 4. 雙向偏差校正（最後一步）

主實驗固定會讀 `settings.yaml` → `dow.underest_penalty` 做這一步（不是選配）。  
這一步是在**改預測**，不是獨立的評分項。設定檔鍵名沿用舊名 `underest_penalty`，程式裡也常寫成 `bias_clip`。

### 4.1 為什麼要加

近期若系統性**低估**就往上抬，**高估**就往下壓；但抬／壓要有天花板，否則改過頭。

這個天花板就叫**校正幅度上限**（單位 MW；設定檔寫在 `cap_by_arm`，程式變數常簡寫成 C）：

- **有套用**：`dow_proto`、`slot_best`、`ft_xgb_best`
- **不套用**：`dow_proto_raw`；對照用的 `slot_lb*` / `ft_xgb_k*` 亦不套（加速）
- **上限 = 0**：該方法關閉這步校正（目前只有 `dow_proto`）
- **上限太小會修不夠**：例如實際高估兩千多 MW，但上限只有 800，就最多只能往下壓 800

### 4.2 前置參考日（例：目標 9/10–9/12）

`lookback_days: 2` 且 `include_prev_same_dow: true` 時，會**額外先預測**參考日所需的前置日（前置日本身不是最終要交的曲線，只用來估近期偏差）：

```text
pre_extra 例 = [9/3, 9/4, 9/5, 9/8, 9/9]   # 含日曆鄰近 + 上週同 weekday
pre_dates     = pre_extra ∪ [9/10, 9/11, 9/12]
```

用這些日的 `實際 − 校正前預測` 估近期偏差，再回正 9/10–9/12。  
輸出：`curves_pre_with_extra_10min.csv`、`best_params.json` → `pre_extra_dates`。

### 4.3 定案設定（`settings.yaml`）

```yaml
dow:
  forecast_mode: rolling          # 參考日僅 <D 且已發生
  finetune_lookbacks: [1, 2, 3, 4, 5, 6, 7, 8]
  underest_penalty:               # 鍵名相容舊腳本；語意 = 雙向偏差校正
    enable: true
    lookback_days: 2
    include_prev_same_dow: true
    same_dow_weight: 2.0          # 同 weekday / D-7 權重大
    calendar_weight: 1.0
    gap_mode: peak                # 尖峰時段混一點 tip_bias
    peak_blend: 0.65
    weekend_uplift_only: true     # 六日只允許往上修、不下壓
    cap_by_arm:                   # 各方法的校正幅度上限（MW）
      dow_proto: 0
      slot_best: 3200
      ft_xgb_best: 3200
  level_scale:                    # ft 水準縮放（抑季節漂移）
    enable: true
    k_same_dow: 2
    clip: [0.85, 1.15]
```

### 4.4 怎麼算

```text
參考日 refs = {D-1, D-2} ∪ {D-7}   # 假日目標則限同假日群
g = 加權平均(實際 − 校正前預測)     # 同 weekday 權重較大
校正量 p = 把 g 限制在 ±上限 之內
週末且 weekend_uplift_only：p 不允許為負（只抬不壓）
ft：先做 level_scale，再 最終 = 縮放後預測 + p
```

輸出：`underest_penalty_by_day.csv`、`level_scale_by_day.csv`、誤差分數 CSV、HTML。

### 4.5 怎麼重找合適的校正幅度上限

換一段日期、或覺得抬／壓太兇／太弱時，可跑：

```text
python run_tune_cap.py --origins 2026-08-06,2026-09-10 --horizon-days 3 \
  --caps 0,800,1600,2400,2800,3200 --skip-xgb-valid --xgb-k 8
```

白話：對每個候選上限（0、800、…、3200）各跑一遍，在多個日期窗上比誤差，挑平均較好的寫回 `cap_by_arm`。  
若只要看結果、不要改設定檔，加 `--no-write-settings`。

---

## 5. 單日流程圖

```mermaid
flowchart TD
  start[預測日 D]
  cut[只用 D 之前的歷史]
  cl[決定星期幾／假日群]
  start --> cut --> cl

  subgraph raw [dow_proto_raw]
    m16[同群 16 日中位數]
  end

  subgraph resid [dow_proto]
    m16b[同群 16 日中位數]
    cal[日曆近兩天誤差]
    add1[加上殘差]
    pen1[雙向校正 上限=0]
    m16b --> add1
    cal --> add1 --> pen1
  end

  subgraph slot [slot_best]
    m16c[同群 16 日中位數]
    same[同群近 L 日誤差]
    add2[加上 β × 誤差]
    pen2[雙向校正 上限=3200]
    m16c --> add2
    same --> add2 --> pen2
  end

  subgraph ftx [ft_xgb_best]
    mk2[同群近 K=8 日中位數]
    xgb[XGB 殘差]
    ls[level_scale]
    pen3[雙向校正 上限=3200]
    mk2 --> xgb --> ls --> pen3
  end

  cl --> raw
  cl --> resid
  cl --> slot
  cl --> ftx
```

`raw` 停在中位數；另外三種方法在各自校正後再加雙向偏差校正。ft 多一步 level_scale。

---

## 6. 程式對照與常用指令

| 主題 | 檔案 |
|------|------|
| 分群 | `domain/cluster_assign.py` |
| 同群日檢索 | `domain/dow_data_split.py` |
| 中位數原型、讀資料 | `domain/train_dow_xgb.py`、`infra/data_io.py` |
| 分開日、slot、日曆殘差 | `domain/daily_predict.py` |
| Slot 殘差／雙向偏差校正 | `domain/slot_adapt.py` |
| FT-XGB | `domain/finetune.py`、`domain/dow_xgb.py` |
| 主實驗與出圖 | `app/experiment_slot_vs_finetune.py` |
| 試不同校正幅度上限 | `app/tune_underest_cap.py` |
| 誤差評分 | `domain/scoring.py` |
| 超參 | `settings.yaml` → `dow.proto_*`、`underest_penalty`、`slot_*`、`finetune_lookbacks` |

```text
cd discrete_dow_cluster

# ① 主實驗（四種方法 + 雙向偏差校正；上限用 settings；FT K=8）
python run_experiment.py --origin 2026-09-10 --horizon-days 3 --skip-xgb-valid --open

# ② 換窗／重調：試不同校正幅度上限
python run_tune_cap.py --origins 2026-08-06,2026-09-10 --horizon-days 3 \
  --caps 0,800,1600,2400,2800,3200 --skip-xgb-valid --xgb-k 8
```

主實驗輸出：`outputs/slot_vs_ft_YYYYMMDD_YYYYMMDD/`  
調上限輸出：`outputs/tune_cap_*/`（含 `best_overall.csv` 等）

---

## 7. 建議閱讀順序

1. **§1** 分群 + 分開日 + 中位數底板（搞清楚「一天一天、一群一群」）
2. **§2–§3** 四種方法各多加了什麼（slot ≠ ft）
3. **§4** 雙向偏差校正與「校正幅度上限」怎麼選
4. 看結果時：正式結論用 `*_best`；表上的 `slot_lb*`、`ft_xgb_k*` 只是對照
