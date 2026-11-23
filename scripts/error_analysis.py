"""Numbers for the "where the baseline fails" analysis (print only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data import load_sales
from src.features import name_stem
from src.utils import load_config

FOLDS = range(29, 34)
PS4 = "Игры - PS4"


def load_preds():
    d = pd.concat([pd.read_csv(f"reports/preds/fold_{t}.csv.gz").assign(fold=t) for t in FOLDS], ignore_index=True)
    d["se"] = (d.pred_lgbm - d.y) ** 2
    return d


def pct(x):
    return f"{100 * x:5.1f}%"


def slices(d):
    print("1. slice shares (pooled 29..33): rows / squared error of pred_lgbm")
    for s, g in d.groupby("slice"):
        print(f"   {s}: rows {pct(len(g) / len(d))}, squared error {pct(g.se.sum() / d.se.sum())}")
    a = d[d.slice == "A"]
    print(f"2. slice A: share of A squared error from under-prediction (pred < y): "
          f"{pct(a.se[a.pred_lgbm < a.y].sum() / a.se.sum())}")
    spike = (d.pred_lag1 <= 1) & (d.y >= 5)
    sa = spike & (d.slice == "A")
    print("3. spikes (pred_lag1 <= 1 and y >= 5), restricted to slice A")
    print(f"   share of all rows {pct(sa.sum() / len(d))}, of A rows {pct(sa.sum() / len(a))}")
    print(f"   share of all squared error {pct(d.se[sa].sum() / d.se.sum())}, of A squared error "
          f"{pct(d.se[sa].sum() / a.se.sum())}")


def new_items(d, items, cats):
    c = d[d.slice == "C"]
    g = c.groupby(["fold", "item_id"]).agg(cat=("item_category_id", "first"), y=("y", "mean"), pred=("pred_lgbm", "mean"),
                                           spread=("pred_lgbm", lambda p: p.max() - p.min()), se=("se", "sum"))
    cat_mean = g.groupby(["fold", "cat"]).pred.transform("mean")
    print(f"4. slice C: {len(g)} new items (fold, item)")
    print(f"   identical prediction across all shops (max-min < 1e-6): {pct((g.spread < 1e-6).mean())}")
    print(f"   item mean equals (fold, category) mean over new items (1e-3): {pct(((g.pred - cat_mean).abs() < 1e-3).mean())}")
    ps4 = cats.index[cats == PS4][0]
    last = g.loc[33].reset_index()
    print(f"   fold 33, new items in '{PS4}': mean pred {last.pred[last.cat == ps4].mean():.4f} "
          f"({(last.cat == ps4).sum()} items)")
    print("   fold 33 top-5 new items by squared error (sum over shops): name | mean y | mean pred")
    for _, r in last.nlargest(5, "se").iterrows():
        print(f"     {items.item_name[int(r.item_id)]} | {r.y:.3f} | {r.pred:.3f}")
    print("5. new items pooled 29..33, item-level mean y over shops (overall | in PS4)")
    low = g.reset_index().item_id.map(items.item_name.str.lower())
    in_ps4 = (g.cat == ps4).to_numpy()
    for word in ["русская версия", "английская версия", "цифровая версия"]:
        has = low.str.contains(word).to_numpy()
        print(f"   '{word}': {g.y[has].mean():.3f} (n={has.sum()}) | {g.y[has & in_ps4].mean():.3f} (n={(has & in_ps4).sum()})")


def new_item_variants(sales, stem, t_range):
    """Per month, per item: number of distinct items of the month-t log sharing its stem, and the item's first month."""
    first = sales.groupby("item_id").date_block_num.min()
    out = []
    for t in t_range:
        its = sales.item_id[sales.date_block_num == t].unique()
        n = stem[its].map(stem[its].value_counts())
        out.append(pd.DataFrame({"t": t, "item_id": its, "n_variants": n.to_numpy(), "is_new": first[its].to_numpy() == t}))
    return pd.concat(out, ignore_index=True)


def train_releases(sales, stem):
    print("6. simultaneous releases, training months 12..28: new item y = mean over active shops of clipped pair sales")
    pair = sales.groupby(["date_block_num", "shop_id", "item_id"]).item_cnt_day.sum().clip(0, 20).reset_index()
    n_shops = sales.groupby("date_block_num").shop_id.nunique()
    y = pair.groupby(["date_block_num", "item_id"]).item_cnt_day.sum().rename("y").reset_index()
    v = new_item_variants(sales, stem, range(12, 29)).query("is_new")
    v = v.merge(y, left_on=["t", "item_id"], right_on=["date_block_num", "item_id"])
    v["y"] /= v.t.map(n_shops)
    v["bucket"] = v.n_variants.clip(upper=4)
    s = v.groupby("bucket").y.agg(["count", "mean", "median"])
    print(s.rename(index={4: "4+"}).round(4).to_string())
    print(f"   ratio mean(3+) / mean(1): {v.y[v.n_variants >= 3].mean() / v.y[v.n_variants == 1].mean():.2f}")


def sibling_shares(sales, stem, cfg):
    print("7. share of new items with n_variants >= 2, per month")
    v = new_item_variants(sales, stem, FOLDS).query("is_new")
    for t, g in v.groupby("t"):
        print(f"   month {t}: {pct((g.n_variants >= 2).mean())} of {len(g)} new items")
    test = pd.read_csv(Path(cfg["data_dir"]) / "test.csv").item_id.unique()
    n = stem[test].map(stem[test].value_counts())
    new = ~np.isin(test, sales.item_id.unique())
    print(f"   month 34 (test): {pct((n[new] >= 2).mean())} of {new.sum()} new test items")


def main():
    cfg = load_config("configs/baseline.yaml")
    items = pd.read_csv(Path(cfg["data_dir"]) / "items.csv").set_index("item_id")
    cats = pd.read_csv(Path(cfg["data_dir"]) / "item_categories.csv").set_index("item_category_id").item_category_name
    stem = name_stem(items.item_name)
    d = load_preds()
    slices(d)
    new_items(d, items, cats)
    sales = load_sales(cfg)
    train_releases(sales, stem)
    sibling_shares(sales, stem, cfg)


if __name__ == "__main__":
    main()
