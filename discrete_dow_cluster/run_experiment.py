"""薄入口：轉呼叫 app.experiment_slot_vs_finetune。

用法（在 discrete_dow_cluster 目錄下）：
  python run_experiment.py --origin 2026-09-10 --horizon-days 3 --skip-xgb-valid
  python -m app.experiment_slot_vs_finetune --origin 2026-09-10 --horizon-days 3 --skip-xgb-valid
"""
from __future__ import annotations

from app.experiment_slot_vs_finetune import main

if __name__ == "__main__":
    main()
