"""薄入口：轉呼叫 app.tune_underest_cap。

用法（在 discrete_dow_cluster 目錄下）：
  python run_tune_cap.py --skip-xgb-valid --xgb-k 8
  python -m app.tune_underest_cap --skip-xgb-valid --xgb-k 8
"""
from __future__ import annotations

from app.tune_underest_cap import main

if __name__ == "__main__":
    main()
