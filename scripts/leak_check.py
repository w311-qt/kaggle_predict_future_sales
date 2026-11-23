"""Leakage checks for the HW2 features: brute-force recomputation and a future-perturbation test."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.utils import load_config
from src.data import build_grid, load_sales, item_month_price
from src.features import add_features, add_hw2_features, FEATURE_GROUPS

cfg = load_config("configs/hw2.yaml", True)
cfg["item_frac"] = 0.1
base = build_grid(cfg)
df = add_hw2_features(add_features(base.copy(), cfg), cfg)
new = sum(FEATURE_GROUPS.values(), [])
rng = np.random.default_rng(1)

sales = load_sales(cfg)
imp = item_month_price(sales).set_index(["date_block_num", "item_id"]).price
y = df.set_index(["date_block_num", "shop_id", "item_id"]).y
rows = df[df.date_block_num >= 14].sample(400, random_state=1)
for _, r in rows.iterrows():
    t, s, i, c = int(r.date_block_num), int(r.shop_id), int(r.item_id), int(r.item_category_id)
    # pair_last_sale_ago
    h = df[(df.shop_id == s) & (df.item_id == i) & (df.date_block_num < t) & df.in_log]
    exp = t - h.date_block_num.max() if len(h) else -1
    assert r.pair_last_sale_ago == exp, ("ago", t, s, i, r.pair_last_sale_ago, exp)
    # pair_mean_6
    m6 = np.mean([y.get((t - k, s, i), 0.0) for k in range(1, 7)])
    assert abs(r.pair_mean_6 - m6) < 1e-4
    # shop_cat_mean_lag_1
    g = df[(df.date_block_num == t - 1) & (df.shop_id == s) & (df.item_category_id == c)].y
    assert abs(r.shop_cat_mean_lag_1 - (g.mean() if len(g) else 0)) < 1e-4
    # price_ratio_lag_1
    hist = imp[(imp.index.get_level_values(0) <= t - 1) & (imp.index.get_level_values(1) == i)]
    exp = np.nan if (t - 1, i) not in imp.index else imp[(t - 1, i)] / hist.max()
    assert (np.isnan(exp) and np.isnan(r.price_ratio_lag_1)) or abs(r.price_ratio_lag_1 - exp) < 1e-4, ("price", exp, r.price_ratio_lag_1)
print("brute-force checks OK on 400 rows (ago, mean_6, shop_cat, price_ratio)")

# future perturbation: change y of month 33 and prices of month 33, features of months <= 33 must not change
# (month 33 features may only depend on months <= 32, month 34 features legitimately depend on 33)
b = base.copy()
m33 = b.date_block_num == 33
b.loc[m33, "y"] = rng.uniform(0, 20, m33.sum()).astype("float32")
b.loc[m33, "y_raw"] = b.loc[m33, "y"]
df2 = add_features(b, cfg)
import src.features as F
orig = F.load_sales
def perturbed(cfg):
    s = orig(cfg).copy(); mm = s.date_block_num == 33
    s.loc[mm, "item_price"] *= np.float32(rng.uniform(0.2, 3))
    return s
F.load_sales = perturbed
df2 = add_hw2_features(df2, cfg)
F.load_sales = orig
old = df[df.date_block_num <= 33][new].to_numpy(float)
cur = df2[df2.date_block_num <= 33][new].to_numpy(float)
same = np.isclose(old, cur, equal_nan=True)
assert same.all(), dict(zip(new, (~same).sum(axis=0)))
chg = ~np.isclose(df[df.date_block_num == 34][new].to_numpy(float), df2[df2.date_block_num == 34][new].to_numpy(float), equal_nan=True)
print("perturbation OK: features of months <= 33 unchanged; month 34 features changed in columns:",
      dict(zip(new, chg.sum(axis=0))))
