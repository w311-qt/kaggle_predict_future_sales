import json
import subprocess
import sys

import numpy as np
import pandas as pd

CONFIG = "configs/baseline.yaml"

for module in ("src.train", "src.evaluate"):
    subprocess.run([sys.executable, "-m", module, "--config", CONFIG, "--smoke"], check=True)

with open("reports/smoke/metrics.json") as f:
    metrics = json.load(f)
for row in metrics["per_fold"]:
    assert np.isfinite(row["rmse"]) and 0 < row["rmse"] < 5, row

preds = pd.read_csv("reports/smoke/preds/fold_33.csv.gz")
assert not preds[["y", "pred_zeros", "pred_lag1", "pred_lgbm"]].isna().any().any()
assert preds.pred_lgbm.between(0, 20).all()
rmse = {(r["fold"], r["model"]): r["rmse"] for r in metrics["per_fold"]}
assert rmse[33, "lgbm"] < rmse[33, "zeros"], "lgbm does not beat zeros"

subprocess.run([sys.executable, "-m", "src.run_hw2", "--config", "configs/hw2.yaml", "--smoke", "--runs", "E0,E3,A1"],
               check=True)
table = pd.read_csv("reports/hw2/smoke/experiments.csv")
assert len(table) == 3, table
assert np.isfinite(table.rmse_mean).all() and table.rmse_mean.between(0, 5).all(), table
e3 = pd.read_csv("reports/hw2/smoke/preds/E3.csv.gz")
assert not e3.isna().any().any() and e3.pred.between(0, 20).all()
print("SMOKE OK")
