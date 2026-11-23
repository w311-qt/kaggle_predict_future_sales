from pathlib import Path

import numpy as np
import pandas as pd

from .data import TEST_MONTH, item_month_price, load_sales


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


FEATURE_GROUPS = {
    "siblings": ["n_siblings"],
    "name_flags": ["name_ru", "name_en", "name_digital", "name_special"],
    "new_cat": ["new_cat_mean_lag_1"],
    "recency": ["pair_last_sale_ago", "pair_mean_6"],
    "shop_cat": ["shop_cat_mean_lag_1"],
    "price": ["price_ratio_lag_1"],
}
SPECIAL_WORDS = ["коллекционное", "специальное", "подарочное", "limited", "collector"]


def name_stem(names):
    """Lowercase name cut at the first bracket, punctuation removed."""
    stem = names.str.lower().str.split(r"[\[(]", n=1, regex=True).str[0]
    return stem.str.replace(r"\W+", " ", regex=True).str.strip()


def add_hw2_features(df, cfg):
    t = df.date_block_num.to_numpy(np.int64)
    shop = df.shop_id.to_numpy(np.int64)
    item = df.item_id.to_numpy(np.int64)
    cat = df.item_category_id.to_numpy(np.int64)
    pair_keys = (t * 100 + shop) * 100000 + item
    y = df.y.to_numpy()

    # name features, looked up by item_id
    items = pd.read_csv(Path(cfg["data_dir"]) / "items.csv", usecols=["item_id", "item_name"])
    lower = items.item_name.str.lower()
    size = items.item_id.max() + 1
    stem_code = np.zeros(size, np.int64)
    stem_code[items.item_id] = pd.factorize(name_stem(items.item_name))[0]
    flags = {"name_ru": lower.str.contains("русская версия|русские субтитры"),
             "name_en": lower.str.contains("английская версия"),
             "name_digital": lower.str.contains("цифровая версия"),
             "name_special": lower.str.contains("|".join(SPECIAL_WORDS))}
    for col, flag in flags.items():
        arr = np.zeros(size, np.int8)
        arr[items.item_id] = flag
        df[col] = arr[item]

    # siblings: distinct items with the same stem in the month-t grid (the item set of t is known upfront)
    month_item = np.unique(t * 100000 + item)
    month_stem = (month_item // 100000) * 100000 + stem_code[month_item % 100000]
    keys, counts = np.unique(month_stem, return_counts=True)
    df["n_siblings"] = lookup(keys, counts, t * 100000 + stem_code[item]).astype("int16")

    # mean y of items that were new in t-1, per category
    keys, vals = group_mean(df[df.item_age == -1], "item_category_id")
    df["new_cat_mean_lag_1"] = lookup(keys, vals, (t - 1) * 100000 + cat).astype("float32")

    # months since the pair last had a log record; the grid is sorted by month
    codes = np.unique(shop * 100000 + item, return_inverse=True)[1]
    last = np.full(codes.max() + 1, -1, np.int64)
    ago = np.empty(len(df), np.int8)
    in_log = df.in_log.to_numpy()
    bounds = np.searchsorted(t, np.arange(TEST_MONTH + 2))
    for m in range(TEST_MONTH + 1):
        sl = slice(bounds[m], bounds[m + 1])
        prev = last[codes[sl]]
        ago[sl] = np.where(prev >= 0, m - prev, -1)
        last[codes[sl][in_log[sl]]] = m
    df["pair_last_sale_ago"] = ago
    lags = sum(lookup(pair_keys, y, ((t - k) * 100 + shop) * 100000 + item) for k in range(1, 7))
    df["pair_mean_6"] = (lags / 6).astype("float32")

    g = df[df.y.notna()].groupby(["date_block_num", "shop_id", "item_category_id"]).y.mean()
    idx = g.index.to_frame(index=False).to_numpy(np.int64)
    keys = (idx[:, 0] * 100 + idx[:, 1]) * 100 + idx[:, 2]
    df["shop_cat_mean_lag_1"] = lookup(keys, g.to_numpy(), ((t - 1) * 100 + shop) * 100 + cat).astype("float32")

    # price in t-1 relative to the item's highest monthly price so far (discount signal)
    price = item_month_price(load_sales(cfg)).sort_values(["item_id", "date_block_num"])
    price["ratio"] = price.price / price.groupby("item_id").price.cummax()
    price = price.assign(key=price.date_block_num.astype(np.int64) * 100000 + price.item_id).sort_values("key")
    ratio = lookup(price.key.to_numpy(), price.ratio.to_numpy(), (t - 1) * 100000 + item, default=np.nan)
    df["price_ratio_lag_1"] = ratio.astype("float32")
    return df
