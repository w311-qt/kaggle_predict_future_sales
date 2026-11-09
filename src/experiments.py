import argparse
import json
import time

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .data import build_grid
from .evaluate import SLICES, md_table, rmse
from .features import add_features, feature_names
from .train import CATEGORICAL
from .utils import item_bootstrap_rmse, load_config, reports_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fit_predict(train, val, features, cfg, target, seed, extra=None):
    model = LGBMRegressor(**{**cfg["lgbm"], **(extra or {})}, random_state=seed)
    model.fit(train[features], train[target], categorical_feature=CATEGORICAL)
    return np.clip(model.predict(val[features]), *cfg["clip"])


def paired_bootstrap(item_id, err_a, err_b, cfg):
    """Cluster bootstrap over items of RMSE(a) - RMSE(b) on the same rows."""
    boot = item_bootstrap_rmse(item_id, {"a": err_a, "b": err_b}, cfg["n_boot"], cfg["seed"])
    delta = boot["a"] - boot["b"]
    return {"mean": float(delta.mean()), "ci": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
            "share_positive": float((delta > 0).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.smoke)
    exp = cfg["experiments"]
    seeds = exp["seeds"][:2] if args.smoke else exp["seeds"]
    variants = {"default": {}, "subsample": exp["subsample_variant"]}
    start = time.time()

    df = add_features(build_grid(cfg), cfg)
    features = feature_names(cfg["lags"])
    first, holdout_t = cfg["first_train_month"], cfg["holdout_month"]

    rq4_rows, seed_rows, lag1_rows = [], [], []
    for t in cfg["val_months"]:
        train = df[(df.date_block_num >= first) & (df.date_block_num < t)]
        val = df[df.date_block_num == t]
        y = val.y.to_numpy()
        lag1_rows.append({"fold": t, "rmse": rmse(val.cnt_lag_1.clip(*cfg["clip"]).to_numpy() - y)})

        preds = {name: fit_predict(train, val, features, cfg, target, exp["clip_seed"])
                 for name, target in [("clipped", "y"), ("raw", "y_raw")]}
        for name, p in preds.items():
            rq4_rows.append({"fold": t, "train_target": name, "rmse": rmse(p - y)})
        if t == holdout_t:
            hold = val[["item_id", "slice"]].reset_index(drop=True)
            hold["y"] = y
            for name, p in preds.items():
                hold[name] = p

        for vname, extra in variants.items():
            for s in seeds:
                p = fit_predict(train, val, features, cfg, "y", s, extra)
                seed_rows.append({"variant": vname, "seed": s, "fold": t, "rmse": rmse(p - y)})
        print(f"fold {t} done, {time.time() - start:.0f}s", flush=True)

    # RQ4
    rq4 = pd.DataFrame(rq4_rows)
    rq4_fold = rq4.pivot(index="fold", columns="train_target", values="rmse")[["clipped", "raw"]]
    rq4_summary = pd.DataFrame({"train_target": ["clipped", "raw"],
                                "rmse_mean": rq4_fold.mean().to_numpy(), "rmse_std": rq4_fold.std().to_numpy()})
    slice_rows = []
    for s in SLICES + ["all"]:
        h = hold if s == "all" else hold[hold.slice == s]
        if len(h):
            slice_rows.append({"slice": s, **{n: rmse(h[n] - h.y) for n in ("clipped", "raw")}})
    boot = paired_bootstrap(hold.item_id.to_numpy(), (hold.raw - hold.y).to_numpy(),
                            (hold.clipped - hold.y).to_numpy(), cfg)

    # RQ5
    seed_df = pd.DataFrame(seed_rows)
    lag1 = pd.DataFrame(lag1_rows).set_index("fold").rmse
    per_fold_rows, summary_rows, plot_data = [], [], {}
    for vname in variants:
        wide = seed_df[seed_df.variant == vname].pivot(index="fold", columns="seed", values="rmse")
        seed_std = wide.std(axis=1)
        per_fold_rows += [{"variant": vname, "fold": t, "seed_mean": wide.loc[t].mean(), "seed_std": seed_std[t],
                           "seed_min": wide.loc[t].min(), "seed_max": wide.loc[t].max()} for t in wide.index]
        fold_mean = wide.mean(axis=1)
        summary_rows.append({"variant": vname, "mean_seed_std": seed_std.mean(), "max_seed_std": seed_std.max(),
                             "fold_std": fold_mean.std(), "lgbm_minus_lag1_gap": (lag1 - fold_mean).mean(),
                             "max_spread_across_seeds": float((wide.max(axis=1) - wide.min(axis=1)).max())})
        plot_data[vname] = wide
    per_fold = pd.DataFrame(per_fold_rows)
    summary = pd.DataFrame(summary_rows)
    d = summary.set_index("variant")
    default_identical = d.loc["default", "max_spread_across_seeds"] == 0

    out = {"rq4": {"per_fold": rq4.to_dict("records"), "summary": rq4_summary.to_dict("records"),
                   "holdout_slices": slice_rows, "bootstrap_delta_raw_minus_clipped": boot},
           "rq5": {"seeds": seeds, "per_seed": seed_rows, "per_fold": per_fold.to_dict("records"),
                   "summary": summary.to_dict("records"), "lag1_rmse": lag1_rows,
                   "default_identical_across_seeds": bool(default_identical)},
           "runtime_seconds": round(time.time() - start)}
    with open(reports_path(cfg, "experiments.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    ci = boot["ci"]
    text = [f"# Experiments (holdout month {holdout_t}, validation months {cfg['val_months']})", "",
            "## RQ4: clip the training target or only the predictions",
            f"LightGBM with baseline params, seed {exp['clip_seed']}. Predictions are clipped to {cfg['clip']} "
            "in both cases; RMSE is against the clipped target.", "",
            "RMSE per fold", md_table(rq4_fold.reset_index()), "",
            "Mean and std over folds", md_table(rq4_summary), "",
            "Holdout RMSE by slice", md_table(pd.DataFrame(slice_rows)), "",
            f"Paired cluster bootstrap over items ({cfg['n_boot']} resamples), delta RMSE (raw - clipped) on holdout: "
            f"mean {boot['mean']:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}], "
            f"share of resamples with delta > 0: {boot['share_positive']:.3f}", "",
            f"Summary: mean RMSE over folds is {rq4_summary.rmse_mean[0]:.4f} (clipped train target) vs "
            f"{rq4_summary.rmse_mean[1]:.4f} (raw train target). On holdout raw - clipped is {boot['mean']:.4f}; "
            f"the 95% CI {'includes' if ci[0] <= 0 <= ci[1] else 'excludes'} zero.", "",
            "## RQ5: seed sensitivity",
            f"Seeds: {seeds}. Variants: default = baseline params; "
            f"subsample = baseline params + {exp['subsample_variant']}.", "",
            f"Default LightGBM params: RMSE {'is identical' if default_identical else 'differs'} across seeds "
            f"(max spread over folds {d.loc['default', 'max_spread_across_seeds']:.2e}).", "",
            "Per fold, across seeds", md_table(per_fold), "",
            "Seed std vs fold std vs gap to lag1 (fold_std: std over folds of seed-averaged RMSE; "
            "gap: mean over folds of lag1 RMSE - lgbm RMSE)", md_table(summary), "",
            f"Summary: mean seed std per fold is {d.loc['default', 'mean_seed_std']:.4f} (default) and "
            f"{d.loc['subsample', 'mean_seed_std']:.4f} (subsample); std over folds is "
            f"{d.loc['default', 'fold_std']:.4f} / {d.loc['subsample', 'fold_std']:.4f}; mean gap to lag1 is "
            f"{d.loc['default', 'lgbm_minus_lag1_gap']:.4f} / {d.loc['subsample', 'lgbm_minus_lag1_gap']:.4f}.", "",
            f"Runtime: {out['runtime_seconds']} s", ""]
    reports_path(cfg, "experiments.md").write_text("\n".join(text), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (vname, wide) in enumerate(plot_data.items()):
        x = np.arange(len(wide)) + (i - 0.5) * 0.2
        ax.scatter(np.repeat(x, wide.shape[1]), wide.to_numpy().ravel(), s=18, label=vname)
    ax.set_xticks(np.arange(len(wide)))
    ax.set_xticklabels(wide.index)
    ax.set_xlabel("fold")
    ax.set_ylabel("RMSE")
    ax.set_title("LightGBM RMSE across seeds")
    ax.legend()
    plt.tight_layout()
    plt.savefig(reports_path(cfg, "figures", "seed_spread.png"), dpi=120)
    plt.close()
    print(f"wrote experiments to {cfg['reports_dir']}, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
