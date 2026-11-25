from pathlib import Path

import pandas as pd

# Orders that never completed. Olist has no separate return table.
EXCLUDED_ORDER_STATUSES = ["canceled", "unavailable"]


def clean_olist(orders: pd.DataFrame, items: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """One row per order line, keyed by the person-level `customer_unique_id`."""
    orders = orders[~orders["order_status"].isin(EXCLUDED_ORDER_STATUSES)]
    orders = orders[["order_id", "customer_id", "order_purchase_timestamp"]]

    df = items.merge(orders, on="order_id", how="inner")
    df = df.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="inner")
    df = df[df["price"] > 0]

    out = pd.DataFrame(
        {
            "customer_id": df["customer_unique_id"].astype(str),
            "invoice": df["order_id"].astype(str),
            "invoice_date": pd.to_datetime(df["order_purchase_timestamp"]),
            "product_id": df["product_id"].astype(str),
            "freight": df["freight_value"].astype(float),
            "revenue": df["price"].astype(float),
        }
    )
    return out.reset_index(drop=True)


def load_olist(input_dir: str | Path) -> pd.DataFrame:
    d = Path(input_dir)
    orders = pd.read_csv(d / "olist_orders_dataset.csv")
    items = pd.read_csv(d / "olist_order_items_dataset.csv")
    customers = pd.read_csv(d / "olist_customers_dataset.csv")
    return clean_olist(orders, items, customers)
