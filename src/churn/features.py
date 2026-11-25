import pandas as pd

BASELINE_FEATURES = ["recency_days", "frequency", "monetary", "tenure_days", "avg_order_value"]
BEHAVIOURAL_FEATURES = ["mean_days_between_orders", "mean_freight", "distinct_products"]
FEATURE_COLUMNS = BASELINE_FEATURES + BEHAVIOURAL_FEATURES


def mean_days_between_orders(past: pd.DataFrame) -> pd.Series:
    """Mean gap in days between consecutive orders per customer; NaN for single-order customers."""
    orders = (
        past[["customer_id", "invoice", "invoice_date"]]
        .drop_duplicates("invoice")
        .sort_values(["customer_id", "invoice_date"])
    )
    gaps = orders.groupby("customer_id")["invoice_date"].diff().dt.days
    return gaps.groupby(orders["customer_id"]).mean()


def rfm_features(past: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Recency, frequency, monetary and behavioural features from transactions before the snapshot."""
    snapshot = pd.Timestamp(snapshot)
    grouped = past.groupby("customer_id")
    out = pd.DataFrame(
        {
            "recency_days": (snapshot - grouped["invoice_date"].max()).dt.days,
            "frequency": grouped["invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
            "tenure_days": (snapshot - grouped["invoice_date"].min()).dt.days,
            "mean_freight": grouped["freight"].mean(),
            "distinct_products": grouped["product_id"].nunique(),
        }
    )
    out["avg_order_value"] = out["monetary"] / out["frequency"]
    out["mean_days_between_orders"] = mean_days_between_orders(past)
    return out
