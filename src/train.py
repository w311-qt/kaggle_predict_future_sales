import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .data import TEST_MONTH, build_grid
from .features import add_features, feature_names
from .utils import load_config, reports_path

CATEGORICAL = ["shop_id", "item_category_id"]
PRED_COLS = ["shop_id", "item_id", "item_category_id", "date_block_num", "slice", "in_log", "y", "y_raw"]


def fit_lgbm(train, features, cfg):
    target = "y" if cfg["clip_train_target"] else "y_raw"
    model = LGBMRegressor(**cfg["lgbm"], random_state=cfg["seed"], importance_type="gain")
    model.fit(train[features], train[target], categorical_feature=CATEGORICAL)
    return model


def predict(model, data, features, cfg):
    return np.clip(model.predict(data[features]), *cfg["clip"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)

    df = add_features(build_grid(cfg), cfg)
    features = feature_names(cfg["lags"])
    print(f"features: {len(features)}, memory {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    first = cfg["first_train_month"]
    stats = []
    for t in cfg["val_months"] + cfg["diag_months"]:
        train = df[(df.date_block_num >= first) & (df.date_block_num < t)]
        val = df[df.date_block_num == t]

        start = time.time()
        model = fit_lgbm(train, features, cfg)
        seconds = time.time() - start

        out = val[PRED_COLS].copy()
        out["pred_zeros"] = 0.0
        out["pred_lag1"] = val.cnt_lag_1.clip(*cfg["clip"])
        out["pred_lgbm"] = predict(model, val, features, cfg)
        out.to_csv(reports_path(cfg, "preds", f"fold_{t}.csv.gz"), index=False)
        stats.append({"fold": t, "train_rows": len(train), "val_rows": len(val), "train_seconds": round(seconds, 1)})
        print(stats[-1])

        if t == cfg["holdout_month"]:
            imp = pd.Series(model.feature_importances_, index=features, name="gain")
            imp.sort_values(ascending=False).rename_axis("feature").to_csv(reports_path(cfg, "feature_importance.csv"))
    pd.DataFrame(stats).to_csv(reports_path(cfg, "fold_stats.csv"), index=False)

    if cfg["smoke"]:
        return
    train = df[(df.date_block_num >= first) & (df.date_block_num < TEST_MONTH)]
    test = df[df.date_block_num == TEST_MONTH]
    model = fit_lgbm(train, features, cfg)
    Path("models").mkdir(exist_ok=True)
    model.booster_.save_model("models/lgbm_final.txt")
    sub = pd.DataFrame({"ID": test.ID.to_numpy(), "item_cnt_month": predict(model, test, features, cfg)})
    sub.sort_values("ID").to_csv(reports_path(cfg, "submission.csv"), index=False)
    print(f"submission: {len(sub)} rows, mean {sub.item_cnt_month.mean():.3f}")


if __name__ == "__main__":
    main()
