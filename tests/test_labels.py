import pandas as pd
import pytest

from churn.labels import build_snapshot, check_label_gap


def _tx(rows):
    """rows: (customer_id, invoice, date, revenue)"""
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "revenue"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["product_id"] = "P"
    df["freight"] = 0.0
    return df


def test_customer_with_purchase_in_window_is_retained():
    tx = _tx([
        ("1", "A1", "2010-01-05", 10.0),
        ("1", "A2", "2010-03-10", 10.0),  # inside [snapshot, snapshot + 30d)
        ("2", "B1", "2010-01-05", 10.0),  # no purchase in window
        ("2", "B0", "2010-01-20", 10.0),
        ("3", "C1", "2010-03-30", 10.0),
        ("3", "C0", "2010-01-01", 10.0),
        ("4", "D0", "2010-02-01", 10.0),
        ("4", "D1", "2010-03-01", 10.0),  # bought exactly on snapshot: counts in window
        ("5", "E1", "2010-06-01", 10.0),  # after the window, provides the data end
    ])
    snap = build_snapshot(tx, "2010-03-01", 30).set_index("customer_id")
    assert snap.loc["1", "churned"] == 0
    assert snap.loc["2", "churned"] == 1
    assert snap.loc["3", "churned"] == 0
    assert snap.loc["4", "churned"] == 0


def test_purchase_exactly_at_window_end_does_not_count():
    tx = _tx([
        ("1", "A0", "2010-01-01", 10.0),
        ("1", "A1", "2010-03-31", 10.0),  # snapshot + 30 days, outside the window
        ("2", "B0", "2010-01-01", 10.0),
        ("2", "B1", "2010-03-01", 10.0),
        ("9", "Z", "2010-06-01", 10.0),
    ])
    snap = build_snapshot(tx, "2010-03-01", 30).set_index("customer_id")
    assert snap.loc["1", "churned"] == 1
    assert snap.loc["2", "churned"] == 0


def test_single_order_customers_are_flagged():
    tx = _tx([
        ("1", "A0", "2010-01-01", 10.0),
        ("2", "B0", "2010-01-02", 10.0),
        ("2", "B1", "2010-01-15", 10.0),
        ("9", "Z", "2010-06-01", 10.0),
    ])
    snap = build_snapshot(tx, "2010-02-01", 30).set_index("customer_id")
    assert bool(snap.loc["1", "single_order"]) is True
    assert bool(snap.loc["2", "single_order"]) is False


def test_window_beyond_observed_data_is_rejected():
    tx = _tx([("1", "A0", "2010-01-01", 10.0), ("1", "A1", "2010-02-01", 10.0)])
    with pytest.raises(ValueError, match="after the last observed date"):
        build_snapshot(tx, "2010-02-01", 90)


def test_training_labels_must_end_before_test_snapshots():
    check_label_gap(["2017-10-01"], "2018-01-01", ["2018-04-01"], 90)  # 2018-01-01 + 90d = 2018-04-01: ok
    with pytest.raises(ValueError, match="Move the test snapshots later"):
        check_label_gap(["2017-10-01"], "2018-01-01", ["2018-03-01"], 90)
