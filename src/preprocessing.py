import pandas as pd


def data_preprocessing() -> tuple[pd.DataFrame, list[str], list[str]]:
    data = pd.read_csv("data/store/train.csv", parse_dates=["week"], low_memory=False)

    sku_ids = sorted(data["sku_id"].unique())
    store_ids = sorted(data["store_id"].unique())
    sku_id2idx = {sku_id: sku_idx for sku_idx, sku_id in enumerate(sku_ids)}
    store_id2idx = {store_id: store_idx for store_idx, store_id in enumerate(store_ids)}
    data["sku_idx"] = data["sku_id"].map(sku_id2idx)
    data["store_idx"] = data["store_id"].map(store_id2idx)
    data = data.drop(columns=["store_id", "sku_id"])

    data["total_price"] = data["total_price"].apply(
        lambda x: float(str(x).strip()) if len(str(x).strip()) > 0 else None,
    )
    data["week_of_month"] = data["week"].apply(lambda d: (d.day - 1) // 7 + 1)
    data["month_of_year"] = data["week"].dt.month

    # Impute missing values by the closest available store-sku pair
    grouped_data = data.groupby(["store_idx", "sku_idx"], group_keys=False)
    data = grouped_data.apply(lambda group: group.bfill().ffill())

    input_columns = [
        "month_of_year",
        "week_of_month",
        "is_display_sku",
        "is_featured_sku",
        "base_price",
        "total_price",
    ]
    output_columns = ["units_sold"]

    return data, input_columns, output_columns
