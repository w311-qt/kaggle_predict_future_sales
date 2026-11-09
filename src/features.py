import numpy as np
import pandas as pd


def feature_names(lags):
    return ([f"cnt_lag_{k}" for k in lags]
            + [f"item_mean_lag_{k}" for k in (1, 2, 3)]
            + ["shop_mean_lag_1", "cat_mean_lag_1", "item_age", "pair_age",
               "month_of_year", "shop_id", "item_category_id"])


def lookup(keys, values, query, default=0):
    """Vectorised dict lookup; keys must be sorted."""
    pos = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    return np.where(keys[pos] == query, values[pos], default)


def group_mean(df, col):
    """Mean of y per (month, col) over grid rows; keys are sorted by construction."""
    g = df[df.y.notna()].groupby(["date_block_num", col]).y.mean()
    keys = g.index.get_level_values(0).to_numpy(np.int64) * 100000 + g.index.get_level_values(1).to_numpy(np.int64)
    return keys, g.to_numpy()


def first_month(df, key):
    """First month in which the key had a log record."""
    g = df[df.in_log].groupby(key).date_block_num.min()
    return g.index.to_numpy(np.int64), g.to_numpy(np.int64)


def add_features(df, cfg):
    t = df.date_block_num.to_numpy(np.int64)
    shop = df.shop_id.to_numpy(np.int64)
    item = df.item_id.to_numpy(np.int64)
    cat = df.item_category_id.to_numpy(np.int64)

    pair_keys = (t * 100 + shop) * 100000 + item
    y = df.y.to_numpy()
    for k in cfg["lags"]:
        df[f"cnt_lag_{k}"] = lookup(pair_keys, y, ((t - k) * 100 + shop) * 100000 + item).astype("float32")

    keys, vals = group_mean(df, "item_id")
    for k in (1, 2, 3):
        df[f"item_mean_lag_{k}"] = lookup(keys, vals, (t - k) * 100000 + item).astype("float32")
    keys, vals = group_mean(df, "shop_id")
    df["shop_mean_lag_1"] = lookup(keys, vals, (t - 1) * 100000 + shop).astype("float32")
    keys, vals = group_mean(df, "item_category_id")
    df["cat_mean_lag_1"] = lookup(keys, vals, (t - 1) * 100000 + cat).astype("float32")

    # first month overall equals first month before t whenever it is < t, so this is leak-free
    keys, vals = first_month(df, "item_id")
    first_item = lookup(keys, vals, item, default=999)
    df["pair_key"] = shop * 100000 + item
    keys, vals = first_month(df, "pair_key")
    first_pair = lookup(keys, vals, df.pair_key.to_numpy(), default=999)
    df = df.drop(columns="pair_key")
    df["item_age"] = np.where(first_item < t, t - first_item, -1).astype("int8")
    df["pair_age"] = np.where(first_pair < t, t - first_pair, -1).astype("int8")
    df["month_of_year"] = (t % 12).astype("int8")

    slice_ = np.select([df.pair_age >= 0, df.item_age >= 0], ["A", "B"], "C")
    df["slice"] = pd.Categorical(slice_, categories=["A", "B", "C"])
    return df
