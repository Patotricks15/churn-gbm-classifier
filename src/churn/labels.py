import pandas as pd

from churn.features import rfm_features


def build_snapshot(tx: pd.DataFrame, snapshot: str | pd.Timestamp, h_days: int) -> pd.DataFrame:
    """One row per customer active before `snapshot`.

    Features use transactions in (-inf, snapshot). The label uses [snapshot, snapshot + h_days).
    """
    snapshot = pd.Timestamp(snapshot)
    window_end = snapshot + pd.Timedelta(days=h_days)
    data_end = tx["invoice_date"].max().normalize()
    if window_end > data_end:
        raise ValueError(
            f"Label window for snapshot {snapshot.date()} ends {window_end.date()}, "
            f"after the last observed date {data_end.date()}."
        )

    past = tx[tx["invoice_date"] < snapshot]
    future = tx[(tx["invoice_date"] >= snapshot) & (tx["invoice_date"] < window_end)]

    panel = rfm_features(past, snapshot)
    panel["churned"] = (~panel.index.isin(future["customer_id"].unique())).astype(int)
    panel["single_order"] = panel["frequency"] == 1
    panel["snapshot_date"] = snapshot
    return panel.rename_axis("customer_id").reset_index()


def check_label_gap(train_snapshots, validation_snapshot, test_snapshots, h_days: int) -> None:
    """Training label windows must end before the first test snapshot starts."""
    fit_snapshots = [pd.Timestamp(s) for s in [*train_snapshots, validation_snapshot]]
    last_label_end = max(fit_snapshots) + pd.Timedelta(days=h_days)
    first_test = min(pd.Timestamp(s) for s in test_snapshots)
    if last_label_end > first_test:
        raise ValueError(
            f"Training label window ends {last_label_end.date()}, after the first test snapshot "
            f"{first_test.date()}. Move the test snapshots later by at least H={h_days} days."
        )
