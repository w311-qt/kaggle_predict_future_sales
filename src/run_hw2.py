import argparse
import json
import time

import matplotlib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .data import build_grid
from .evaluate import SLICES, md_table, rmse
from .features import FEATURE_GROUPS, add_features, add_hw2_features, feature_names
from .train import CATEGORICAL
from .utils import item_bootstrap_rmse, load_config, reports_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRED_KEYS = ["date_block_num", "shop_id", "item_id", "item_category_id", "slice", "y"]
COMPARE_COLS = ["delta_vs_baseline", "ci_low", "ci_high", "folds_improved", "delta_fold22_vs_baseline", "ci22_low",
                "ci22_high", "delta_vs_parent", "ci_parent_low", "ci_parent_high"]


def log(start, msg):
    print(f"[{time.time() - start:6.0f}s] {msg}", flush=True)


def run_features(cfg, run):
    return feature_names(cfg["lags"]) + [c for g in run["groups"] for c in FEATURE_GROUPS[g]]


def fit_run(df, run, cfg, seeds, start):
    """Rolling backtest of one experiment; returns raw results and seed-averaged predictions."""
    features = run_features(cfg, run)
    first, holdout = cfg["first_train_month"], cfg["holdout_month"]
    months = df.date_block_num.to_numpy()
    rows = {m: np.searchsorted(months, m) for m in range(first, 36)}
    per_seed, parts, seconds, curve, importance = [], [], 0.0, None, None

    for t in cfg["val_months"] + cfg["diag_months"]:
        fold_seeds = seeds if t in cfg["val_months"] else seeds[:1]
        train = df.iloc[rows[first]:rows[t]]
        val = df.iloc[rows[t]:rows[t + 1]]
        x_train, x_val, y_val = train[features], val[features], val.y.to_numpy()
        preds = []
        for s in fold_seeds:
            model = LGBMRegressor(**cfg["lgbm"], random_state=s, importance_type="gain")
            extra = {}
            if t == holdout and s == seeds[0]:
                extra = {"eval_X": x_val, "eval_y": val.y, "eval_metric": "rmse"}
            t0 = time.time()
            model.fit(x_train, train.y, categorical_feature=CATEGORICAL, **extra)
            seconds += time.time() - t0
            p = np.clip(model.predict(x_val), *cfg["clip"])
            preds.append(p)
            per_seed.append({"fold": t, "seed": s, "rmse": rmse(p - y_val), "mae": float(np.abs(p - y_val).mean())})
            if extra:
                curve = model.evals_result_["valid_0"]["rmse"]
                importance = dict(zip(features, model.booster_.feature_importance("gain")))
        out = val[PRED_KEYS].copy()
        out["pred"] = np.mean(preds, axis=0).astype("float32").round(4)
        parts.append(out)
        log(start, f"{run['id']} fold {t}: rmse {np.mean([r['rmse'] for r in per_seed if r['fold'] == t]):.4f}")

    res = {"features": features, "per_seed": per_seed, "seconds": seconds, "curve": curve, "importance": importance}
    return res, pd.concat(parts, ignore_index=True)


def summarize(res, pred, cfg):
    """Scalar metrics of one run."""
    ps = pd.DataFrame(res["per_seed"])
    val = ps[ps.fold.isin(cfg["val_months"])]
    fold_rmse = val.groupby("fold").rmse.mean()
    pooled = pred[pred.date_block_num.isin(cfg["val_months"])]
    se = (pooled.pred - pooled.y) ** 2
    diag = ps[ps.fold.isin(cfg["diag_months"])]
    out = {"rmse_fold": {int(t): float(v) for t, v in fold_rmse.items()},
           "rmse_mean": float(fold_rmse.mean()),
           "rmse_std": float(fold_rmse.std()) if len(fold_rmse) > 1 else float("nan"),
           "seed_std": float(val.groupby("fold").rmse.std().mean()),
           "mae_mean": float(val.groupby("fold").mae.mean().mean()),
           "rmse_fold22": float(diag.rmse.mean()) if len(diag) else float("nan"),
           "n_features": len(res["features"]), "train_seconds": round(res["seconds"], 1)}
    for s in SLICES:
        out[f"rmse_{s}"] = float(np.sqrt(se[pooled.slice == s].mean()))
    return out


def load_preds(cfg, run_id):
    return pd.read_csv(reports_path(cfg, "preds", f"{run_id}.csv.gz"), dtype={"slice": "category"})


def compare(cfg, pred_a, pred_b, fold_a, fold_b, months=None):
    """RMSE(a) - RMSE(b) on pooled rows of the given months, with a paired item-cluster bootstrap 95% CI."""
    months = cfg["val_months"] if months is None else months
    a, b = (p[p.date_block_num.isin(months)] for p in (pred_a, pred_b))
    for k in ["date_block_num", "shop_id", "item_id"]:
        assert (a[k].to_numpy() == b[k].to_numpy()).all(), f"predictions are not aligned on {k}"
    ea, eb = [(p.pred - p.y).to_numpy() for p in (a, b)]
    boot = item_bootstrap_rmse(a.item_id.to_numpy(), {"a": ea, "b": eb}, cfg["n_boot"], cfg["seed"])
    delta = boot["a"] - boot["b"]
    return {"delta": rmse(ea) - rmse(eb), "ci_low": float(np.percentile(delta, 2.5)),
            "ci_high": float(np.percentile(delta, 97.5)),
            "folds_improved": int(sum(fold_a[t] < fold_b[t] for t in fold_a))}


def build_table(cfg, results, start):
    ref = next((r for r in cfg["runs"] if r["kind"] == cfg["hw2"]["reference"] and r["id"] in results), None)
    preds, rows = {}, []

    def get(run_id):
        if run_id not in preds:
            preds[run_id] = load_preds(cfg, run_id)
        return preds[run_id]

    for run in cfg["runs"]:
        if run["id"] not in results:
            continue
        m = results[run["id"]]["metrics"]
        seeds = sorted({r["seed"] for r in results[run["id"]]["per_seed"]})
        row = {"id": run["id"], "kind": run["kind"], "groups": "+".join(run["groups"]) or "-",
               "hypothesis": run["hypothesis"], "seeds": ",".join(map(str, seeds)), "boot_seed": cfg["seed"],
               "n_features": m["n_features"]}
        row.update({f"rmse_fold_{t}": m["rmse_fold"][str(t)] for t in cfg["val_months"]})
        row.update({k: m[k] for k in ["rmse_mean", "rmse_std", "seed_std", "mae_mean", "rmse_A", "rmse_B", "rmse_C",
                                      "rmse_fold22"]})
        if ref and run["id"] != ref["id"]:
            fold_ref = results[ref["id"]]["metrics"]["rmse_fold"]
            c = compare(cfg, get(run["id"]), get(ref["id"]), m["rmse_fold"], fold_ref)
            row.update({"delta_vs_baseline": c["delta"], "ci_low": c["ci_low"], "ci_high": c["ci_high"],
                        "folds_improved": c["folds_improved"]})
            if cfg["diag_months"]:
                d = compare(cfg, get(run["id"]), get(ref["id"]), m["rmse_fold"], fold_ref, cfg["diag_months"])
                row.update({"delta_fold22_vs_baseline": d["delta"], "ci22_low": d["ci_low"], "ci22_high": d["ci_high"]})
        parent = run.get("parent")
        if parent in results:
            c = compare(cfg, get(run["id"]), get(parent), m["rmse_fold"], results[parent]["metrics"]["rmse_fold"])
            row.update({"delta_vs_parent": c["delta"], "ci_parent_low": c["ci_low"], "ci_parent_high": c["ci_high"]})
        rows.append(row)
        log(start, f"compared {run['id']}")
    return pd.DataFrame(rows), preds


def save_fig(cfg, name):
    plt.tight_layout()
    plt.savefig(reports_path(cfg, "figures", name), dpi=120)
    plt.close()


def delta_bars(cfg, t, ids, cols, labels, title, xlabel, name):
    col, lo, hi = cols
    d = t.loc[[i for i in ids if i in t.index and col in t and pd.notna(t.loc[i, col])]]
    if d.empty:
        return
    err = np.array([d[col] - d[lo], d[hi] - d[col]])
    plt.figure(figsize=(7, 1.5 + 0.5 * len(d)))
    plt.barh([labels[i] for i in d.index], d[col], xerr=err, capsize=4)
    plt.axvline(0, color="black", lw=1)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    save_fig(cfg, name)


def make_figures(cfg, table, results, preds):
    t = table.set_index("id")
    folds = [f"rmse_fold_{m}" for m in cfg["val_months"]]
    base = table.id[table.kind == cfg["hw2"]["reference"]]

    plt.figure(figsize=(8, 4))
    x = np.arange(len(t))
    plt.errorbar(x, t.rmse_mean, yerr=t.rmse_std.fillna(0), fmt="o", capsize=4, label="mean +- std over folds")
    for i, (_, r) in enumerate(t.iterrows()):
        plt.scatter(np.full(len(folds), i), r[folds].to_numpy(float), s=10, color="gray", alpha=0.6)
    if len(base):
        plt.axhline(t.rmse_mean[base.iloc[0]], ls="--", color="red", label="baseline")
    plt.xticks(x, t.index)
    plt.ylabel("RMSE")
    plt.title("RMSE by experiment (gray: single folds)")
    plt.legend()
    save_fig(cfg, "rmse_by_experiment.png")

    names = {i: i for i in t.index}
    delta_bars(cfg, t, ["E1", "E2", "E3", "E4"], ("delta_vs_baseline", "ci_low", "ci_high"), names,
               "Change vs baseline (95% CI)", "RMSE(run) - RMSE(baseline)", "delta_vs_baseline.png")
    abl = {i: f"{i} (-{results[i]['run']['ablates']})" for i in t.index if "ablates" in results[i]["run"]}
    delta_bars(cfg, t, list(abl), ("delta_vs_parent", "ci_parent_low", "ci_parent_high"), abl,
               "Ablation: removed group vs full (95% CI)", "RMSE(without group) - RMSE(full); > 0: group helps",
               "ablation_delta.png")

    ids = [i for i in ["E0", "E1", "E2", "E3", "E4"] if i in t.index]
    if ids:
        t.loc[ids, ["rmse_A", "rmse_B", "rmse_C"]].rename(columns=lambda c: c[-1]).T.plot.bar(figsize=(7, 4), rot=0)
        plt.xlabel("slice")
        plt.ylabel("RMSE")
        plt.title("RMSE by slice (pooled validation folds)")
        save_fig(cfg, "slice_rmse.png")

    curves = {i: results[i]["curve"] for i in ids if results[i].get("curve")}
    if curves:
        plt.figure(figsize=(7, 4))
        for i, c in curves.items():
            plt.plot(np.arange(1, len(c) + 1), c, label=i)
        plt.xlabel("iteration")
        plt.ylabel("holdout validation RMSE")
        plt.title(f"Learning curve, fold {cfg['holdout_month']}, seed {cfg['hw2']['seeds'][0]}")
        plt.legend()
        save_fig(cfg, "learning_curve.png")

    pair = [i for i in ("E0", "E3") if i in preds]
    if pair:
        fig, axes = plt.subplots(1, len(pair), figsize=(5 * len(pair), 4.5), squeeze=False)
        for ax, i in zip(axes[0], pair):
            p = preds[i]
            p = p[(p.date_block_num == cfg["holdout_month"]) & (p.slice == "C")]
            g = p.groupby("item_id")[["y", "pred"]].mean()
            ax.scatter(g.y, g.pred, s=8, alpha=0.5)
            ax.plot([0, g.y.max()], [0, g.y.max()], "r--", lw=1)
            ax.set_xlabel("mean y over shops")
            ax.set_ylabel("mean pred over shops")
            ax.set_title(f"{i}: new items, fold {cfg['holdout_month']}")
        save_fig(cfg, "new_items_scatter.png")

    if results.get("E3", {}).get("importance"):
        imp = pd.Series(results["E3"]["importance"]).sort_values()
        plt.figure(figsize=(7, 6))
        plt.barh(imp.index, imp.to_numpy())
        plt.xlabel("gain")
        plt.title("Feature importance, E3 (holdout fold, seed 0)")
        save_fig(cfg, "feature_importance_full.png")


def log_mlflow(cfg, run, res, metrics, seeds):
    with mlflow.start_run(run_name=run["id"]) as active:
        mlflow.log_params({"groups": ",".join(run["groups"]), "kind": run["kind"], "hypothesis": run["hypothesis"],
                           "seeds": seeds, "n_features": metrics["n_features"], "val_months": cfg["val_months"],
                           **{f"lgbm_{k}": v for k, v in cfg["lgbm"].items()}})
        mlflow.set_tags({"kind": run["kind"], **({"ablates": run["ablates"]} if "ablates" in run else {})})
        mlflow.log_text("\n".join(res["features"]), "features.txt")
        scalars = {k: metrics[k] for k in ["rmse_mean", "rmse_std", "seed_std", "mae_mean", "rmse_fold22",
                                           "train_seconds", "rmse_A", "rmse_B", "rmse_C"]}
        mlflow.log_metrics({**scalars, **{f"rmse_fold_{t}": v for t, v in metrics["rmse_fold"].items()}})
        for i, v in enumerate(res["curve"] or [], 1):
            mlflow.log_metric("holdout_valid_rmse", v, step=i)
        imp = pd.Series(res["importance"] or {}, name="gain").sort_values(ascending=False)
        mlflow.log_text(imp.rename_axis("feature").reset_index().to_csv(index=False), "feature_importance.csv")
        return active.info.run_id


def log_comparisons(results, table):
    """Add the bootstrap comparisons to the MLflow runs once all results are known."""
    for _, r in table.iterrows():
        cols = {k: float(r[k]) for k in COMPARE_COLS if k in r and pd.notna(r[k])}
        run_id = results[r["id"]].get("mlflow_run_id")
        if cols and run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runs", default="")
    parser.add_argument("--report-only", action="store_true", help="rebuild tables and figures from saved runs")
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)
    seeds = cfg["hw2"]["seeds"][:2] if args.smoke else cfg["hw2"]["seeds"]
    if args.smoke:
        cfg["n_boot"] = 50
        cfg["mlflow"]["experiment"] = "pfs-hw2-smoke"
    selected = args.runs.split(",") if args.runs else [r["id"] for r in cfg["runs"]]
    start = time.time()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    df = None if args.report_only else add_hw2_features(add_features(build_grid(cfg), cfg), cfg)
    if df is not None:
        log(start, f"frame {df.shape}, memory {df.memory_usage().sum() / 1e9:.2f} GB")
    for run in cfg["runs"]:
        if args.report_only or run["id"] not in selected:
            continue
        res, pred = fit_run(df, run, cfg, seeds, start)
        metrics = summarize(res, pred, cfg)
        pred.to_csv(reports_path(cfg, "preds", f"{run['id']}.csv.gz"), index=False)
        res["metrics"], res["run"] = metrics, run
        res["mlflow_run_id"] = log_mlflow(cfg, run, res, metrics, seeds)
        with open(reports_path(cfg, "runs", f"{run['id']}.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        log(start, f"{run['id']} done: rmse_mean {metrics['rmse_mean']:.4f}, {metrics['train_seconds']}s of training")
    del df
    if args.report_only:
        selected = [r["id"] for r in cfg["runs"]]

    # tables and figures use every saved run (in smoke mode only the selected ones, to ignore stale files)
    results = {}
    for run in cfg["runs"]:
        path = reports_path(cfg, "runs", f"{run['id']}.json")
        if path.exists() and (not args.smoke or run["id"] in selected):
            with open(path) as f:
                results[run["id"]] = json.load(f)
    table, preds = build_table(cfg, results, start)
    table.to_csv(reports_path(cfg, "experiments.csv"), index=False)
    reports_path(cfg, "experiments.md").write_text(
        f"# HW2 experiments (validation months {cfg['val_months']}, seeds {seeds})\n\n" + md_table(table) + "\n",
        encoding="utf-8")
    with open(reports_path(cfg, "metrics.json"), "w") as f:
        json.dump({i: {"metrics": r["metrics"], "per_seed": r["per_seed"]} for i, r in results.items()}, f, indent=1,
                  default=float)
    make_figures(cfg, table, results, preds)
    if "E0" in preds and "E3" in preds:  # small holdout file for the repo, full preds stay in preds/
        h = [p[p.date_block_num == cfg["holdout_month"]].reset_index(drop=True) for p in (preds["E0"], preds["E3"])]
        out = h[0][PRED_KEYS].assign(pred_E0=h[0].pred, pred_E3=h[1].pred)
        out.to_csv(reports_path(cfg, "holdout_preds_E0_E3.csv.gz"), index=False)
    log_comparisons(results, table)
    # mlflow.db stays local (absolute paths, user name), so the run table is exported for the repo
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])
    local = ["artifact_uri", "tags.mlflow.source.name", "tags.mlflow.user"]
    runs.drop(columns=local, errors="ignore").to_csv(reports_path(cfg, "mlflow_runs.csv"), index=False)
    log(start, f"wrote results to {cfg['reports_dir']}")


if __name__ == "__main__":
    main()
