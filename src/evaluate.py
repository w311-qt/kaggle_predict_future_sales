import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

from .utils import MODELS, item_bootstrap_rmse, load_config, reports_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLICES = ["A", "B", "C"]


def rmse(err):
    return float(np.sqrt(np.mean(err ** 2)))


def md_table(df):
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def fold_metrics(folds):
    rows = []
    for t, d in folds.items():
        for m in MODELS:
            err = d["pred_" + m] - d.y
            rows.append({"fold": t, "model": m, "rmse": rmse(err), "mae": float(err.abs().mean())})
    return pd.DataFrame(rows)


def slice_metrics(d):
    rows = []
    for m in MODELS:
        se = (d["pred_" + m] - d.y) ** 2
        for s in SLICES:
            mask = d.slice == s
            rows.append({"model": m, "slice": s, "rows_share": mask.mean(),
                         "rmse": float(np.sqrt(se[mask].mean())), "sq_err_share": se[mask].sum() / se.sum()})
    return pd.DataFrame(rows)


def log_metrics(d):
    rows = []
    for m in MODELS:
        err = d["pred_" + m] - d.y
        rows.append({"model": m, "rmse_full_grid": rmse(err), "rmse_logged_only": rmse(err[d.in_log])})
    return pd.DataFrame(rows)


def bootstrap(d, cfg):
    """Cluster bootstrap over items using per-item sums of squared errors."""
    errors = {m: (d["pred_" + m] - d.y).to_numpy() for m in MODELS}
    boot = item_bootstrap_rmse(d.item_id.to_numpy(), errors, cfg["n_boot"], cfg["seed"])
    delta = boot["lgbm"] - boot["lag1"]

    out = {m: [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))] for m in MODELS}
    out["delta_lgbm_minus_lag1"] = {
        "mean": float(delta.mean()),
        "ci": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "share_negative": float((delta < 0).mean())}
    return out


def error_cases(d):
    rows = []
    for s in SLICES:
        sub = d[d.slice == s]
        if sub.empty:
            continue
        r = sub.loc[(sub.pred_lgbm - sub.y).abs().idxmax()]
        rows.append({"slice": s, "shop_id": int(r.shop_id), "item_id": int(r.item_id),
                     "category": int(r.item_category_id), "y": r.y, "pred_lgbm": r.pred_lgbm, "pred_lag1": r.pred_lag1})
    return pd.DataFrame(rows)


def plot_bars(table, index, title, path):
    table = table.pivot(index=index, columns="model", values="rmse")[MODELS]
    ax = table.plot.bar(figsize=(7, 4), rot=0)
    ax.set_xlabel(index)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)

    months = cfg["val_months"] + cfg["diag_months"]
    folds = {t: pd.read_csv(reports_path(cfg, "preds", f"fold_{t}.csv.gz"), dtype={"slice": "category"}) for t in months}
    holdout = folds[cfg["holdout_month"]]

    per_fold = fold_metrics(folds)
    val = per_fold[per_fold.fold.isin(cfg["val_months"])]
    summary = (val.groupby("model").agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                                        mae_mean=("mae", "mean"), mae_std=("mae", "std"))
                  .reindex(MODELS).reset_index())
    by_slice = slice_metrics(holdout)
    by_log = log_metrics(holdout)
    boot = bootstrap(holdout, cfg)
    cases = error_cases(holdout)

    metrics = {"per_fold": per_fold.to_dict("records"), "summary": summary.to_dict("records"),
               "holdout_slices": by_slice.to_dict("records"), "holdout_log": by_log.to_dict("records"),
               "bootstrap": boot, "error_cases": cases.to_dict("records")}
    with open(reports_path(cfg, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    fold_table = per_fold.pivot(index="fold", columns="model", values="rmse")[MODELS].reset_index()
    boot_rows = pd.DataFrame([{"model": m, "ci_low": boot[m][0], "ci_high": boot[m][1]} for m in MODELS])
    d = boot["delta_lgbm_minus_lag1"]
    text = [f"# Baseline results (holdout month {cfg['holdout_month']})", "",
            f"## RMSE per fold (diagnostic folds {cfg['diag_months']} are not in the mean)", md_table(fold_table), "",
            f"## Mean and std over validation months {cfg['val_months']}", md_table(summary), "",
            "## Holdout by slice (A: pair seen, B: item seen elsewhere, C: new item)", md_table(by_slice), "",
            "## Holdout: logged pairs only vs full test-like grid", md_table(by_log), "",
            f"## Cluster bootstrap over items ({cfg['n_boot']} resamples), 95% CI of RMSE", md_table(boot_rows), "",
            f"Delta RMSE (lgbm - lag1): mean {d['mean']:.4f}, 95% CI [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}], "
            f"share of resamples with delta < 0: {d['share_negative']:.3f}", "",
            "## Largest lgbm errors on holdout per slice", md_table(cases), ""]
    reports_path(cfg, "results.md").write_text("\n".join(text), encoding="utf-8")

    plot_bars(per_fold, "fold", "RMSE by fold", reports_path(cfg, "figures", "rmse_by_fold.png"))
    plot_bars(by_slice, "slice", "Holdout RMSE by slice", reports_path(cfg, "figures", "rmse_by_slice.png"))
    imp = pd.read_csv(reports_path(cfg, "feature_importance.csv")).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(imp.feature, imp.gain)
    plt.xlabel("gain")
    plt.title("LightGBM feature importance (holdout fold)")
    plt.tight_layout()
    plt.savefig(reports_path(cfg, "figures", "feature_importance.png"), dpi=120)
    plt.close()
    print(f"wrote results to {cfg['reports_dir']}")


if __name__ == "__main__":
    main()
