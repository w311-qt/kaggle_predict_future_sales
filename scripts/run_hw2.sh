#!/usr/bin/env bash
set -e
python -m src.run_hw2 --config configs/hw2.yaml   # 11 runs x 5 folds x 3 seeds, about 40 minutes
mlflow ui --backend-store-uri sqlite:///mlflow.db  # optional, http://127.0.0.1:5000
