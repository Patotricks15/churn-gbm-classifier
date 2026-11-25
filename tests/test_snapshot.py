import pandas as pd

from churn.data import clean_olist
from churn.features import FEATURE_COLUMNS, rfm_features
from churn.labels import build_snapshot


def _tx(rows):
    """rows: (customer_id, invoice, date, product_id, freight, revenue)"""
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "product_id", "freight", "revenue"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def test_clean_olist_drops_cancelled_orders_and_links_unique_customer():
    orders = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_id": ["c1", "c2", "c3"],
        "order_status": ["delivered", "canceled", "delivered"],
        "order_purchase_timestamp": ["2017-01-01 10:00", "2017-01-02 10:00", "2017-01-03 10:00"],
    })
    items = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "product_id": ["p1", "p2", "p3"],
        "price": [10.0, 20.0, 0.0],
        "freight_value": [1.0, 2.0, 3.0],
    })
    customers = pd.DataFrame({"customer_id": ["c1", "c2", "c3"], "customer_unique_id": ["u1", "u2", "u3"]})
    clean = clean_olist(orders, items, customers)
    assert list(clean["customer_id"]) == ["u1"]
    assert list(clean["invoice"]) == ["o1"]
    assert clean.loc[0, "revenue"] == 10.0


def test_features_use_only_rows_before_snapshot():
    tx = _tx([
        ("1", "A", "2017-01-10 10:00", "p", 1.0, 10.0),
        ("1", "B", "2017-02-10 10:00", "q", 5.0, 50.0),
    ])
    feats = rfm_features(tx[tx["invoice_date"] < "2017-02-01"], pd.Timestamp("2017-02-01"))
    assert feats.loc["1", "frequency"] == 1
    assert feats.loc["1", "recency_days"] == 21
    assert feats.loc["1", "monetary"] == 10.0
    assert feats.loc["1", "mean_freight"] == 1.0
    assert pd.isna(feats.loc["1", "mean_days_between_orders"])


def test_mean_days_between_orders_is_hand_checkable():
    tx = _tx([
        ("1", "A", "2017-01-01", "p", 0.0, 10.0),
        ("1", "B", "2017-01-11", "p", 0.0, 10.0),  # gap 10 days
        ("1", "C", "2017-01-31", "p", 0.0, 10.0),  # gap 20 days; mean 15
        ("1", "C", "2017-01-31", "q", 0.0, 10.0),  # second line of order C: not a new order
    ])
    feats = rfm_features(tx, pd.Timestamp("2017-02-01"))
    assert feats.loc["1", "mean_days_between_orders"] == 15.0
    assert feats.loc["1", "distinct_products"] == 2


def test_snapshot_cut_is_not_affected_by_future_transactions():
    base = _tx([
        ("1", "A", "2017-01-10", "p", 1.0, 10.0),
        ("2", "B", "2017-01-20", "p", 1.0, 10.0),
        ("3", "C", "2017-06-01", "p", 1.0, 10.0),
        ("3", "D", "2017-06-05", "p", 1.0, 10.0),
    ])
    future_extra = _tx([
        ("1", "E", "2017-02-15", "q", 9.0, 99.0),
        ("2", "F", "2017-03-01", "q", 9.0, 99.0),
    ])
    snap_a = build_snapshot(base, "2017-02-01", 60).set_index("customer_id")
    snap_b = build_snapshot(pd.concat([base, future_extra]), "2017-02-01", 60).set_index("customer_id")

    pd.testing.assert_frame_equal(snap_a.loc[["1", "2"], FEATURE_COLUMNS], snap_b.loc[["1", "2"], FEATURE_COLUMNS])
    assert snap_a.loc["1", "churned"] == 1
    assert snap_b.loc["1", "churned"] == 0
