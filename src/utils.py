from pathlib import Path

import numpy as np
import yaml

MODELS = ["zeros", "lag1", "lgbm"]


def load_config(path, smoke=False):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["smoke"] = smoke
    if smoke:
        cfg["reports_dir"] = str(Path(cfg["reports_dir"]) / "smoke")
        cfg["val_months"] = [cfg["holdout_month"]]
        cfg["diag_months"] = []
        cfg["lgbm"]["n_estimators"] = 20
        cfg["item_frac"] = 0.05
    return cfg


def item_bootstrap_rmse(item_id, errors, n_boot, seed):
    """Cluster bootstrap over items: per-model RMSE arrays on identical item resamples."""
    _, inv = np.unique(item_id, return_inverse=True)
    n_items = inv.max() + 1
    counts = np.bincount(inv, minlength=n_items)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_items, size=(n_boot, n_items))
    n_rows = counts[idx].sum(axis=1)
    return {m: np.sqrt(np.bincount(inv, weights=np.asarray(e) ** 2, minlength=n_items)[idx].sum(axis=1) / n_rows)
            for m, e in errors.items()}


def reports_path(cfg, *parts):
    path = Path(cfg["reports_dir"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
