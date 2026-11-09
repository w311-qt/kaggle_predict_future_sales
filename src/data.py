from pathlib import Path

import numpy as np
import pandas as pd

TEST_MONTH = 34


def load_sales(cfg):
    path = Path(cfg["data_dir"]) / "sales_train.csv"
    dtypes = {"date_block_num": "int8", "shop_id": "int8", "item_id": "int16",
              "item_price": "float32", "item_cnt_day": "float32"}
    sales = pd.read_csv(path, dtype=dtypes)
    n = len(sales)
    sales = sales.drop_duplicates()
    sales = sales[(sales.item_price > 0) & (sales.item_price <= cfg["price_max"])]
    sales = sales[sales.item_cnt_day <= cfg["cnt_day_max"]]
    sales["shop_id"] = sales.shop_id.replace(cfg["shop_merge"]).astype("int8")
    sales = sales[~sales.shop_id.isin(cfg["excluded_shops"])]
    print(f"sales: {n} -> {len(sales)} rows after cleaning")
    return sales[["date_block_num", "shop_id", "item_id", "item_cnt_day"]]


def build_grid(cfg):
    """One row per (month, shop, item): shops and items active in that month, crossed."""
    data_dir = Path(cfg["data_dir"])
    items = pd.read_csv(data_dir / "items.csv", usecols=["item_id", "item_category_id"])
    test = pd.read_csv(data_dir / "test.csv")
    sales = load_sales(cfg)

    if cfg["item_frac"] < 1:
        rng = np.random.default_rng(cfg["seed"])
        keep = rng.choice(items.item_id.to_numpy(), int(len(items) * cfg["item_frac"]), replace=False)
        sales = sales[sales.item_id.isin(keep)]
        test = test[test.item_id.isin(keep)]

    monthly = (sales.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)
                    .item_cnt_day.sum().rename(columns={"item_cnt_day": "y_raw"}))
    monthly["in_log"] = True

    parts = []
    for t in range(TEST_MONTH):
        m = monthly[monthly.date_block_num == t]
        shops, its = m.shop_id.unique(), m.item_id.unique()
        grid = pd.DataFrame({"shop_id": np.repeat(shops, len(its)), "item_id": np.tile(its, len(shops))})
        grid["date_block_num"] = np.int8(t)
        grid = grid.merge(m, on=["date_block_num", "shop_id", "item_id"], how="left")
        parts.append(grid)
    grid = pd.concat(parts, ignore_index=True)
    grid["in_log"] = grid.in_log.fillna(False).astype(bool)
    grid["y_raw"] = grid.y_raw.fillna(0).astype("float32")
    grid["y"] = grid.y_raw.clip(*cfg["clip"])
    grid["ID"] = np.int32(-1)

    test["shop_id"] = test.shop_id.replace(cfg["shop_merge"])
    test_rows = pd.DataFrame({
        "shop_id": test.shop_id.astype("int8"), "item_id": test.item_id.astype("int16"),
        "date_block_num": np.int8(TEST_MONTH), "in_log": False,
        "y_raw": np.float32("nan"), "y": np.float32("nan"), "ID": test.ID.astype("int32")})
    df = pd.concat([grid, test_rows], ignore_index=True)

    df = df.merge(items.astype({"item_id": "int16", "item_category_id": "int8"}), on="item_id", how="left")
    df = df.sort_values(["date_block_num", "shop_id", "item_id"], ignore_index=True)

    last = df[df.date_block_num == TEST_MONTH - 1]
    print(f"grid: {len(df)} rows total; month 33: {len(last)} rows = "
          f"{last.shop_id.nunique()} shops x {last.item_id.nunique()} items; "
          f"mean y {last.y.mean():.3f}; test rows {len(test_rows)}")
    return df
