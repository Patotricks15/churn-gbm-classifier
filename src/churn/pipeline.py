import numpy as np
import pandas as pd

from churn.labels import build_snapshot, check_label_gap
from churn.metrics import evaluate
from churn.models import (
    fit_baseline_logistic,
    fit_tuned_lgbm,
    gain_importance,
    predict_baseline,
    predict_lgbm,
)


def build_panel(tx: pd.DataFrame, snapshots, h_days: int) -> pd.DataFrame:
    return pd.concat([build_snapshot(tx, snap, h_days) for snap in snapshots], ignore_index=True)


def run_experiment(tx, train_snapshots, validation_snapshot, test_snapshots, h_days: int, n_iter: int):
    """Tune on the validation snapshot, refit on train + validation, evaluate on later test snapshots.

    Single-order customers are excluded from training and from the main metrics.
    """
    check_label_gap(train_snapshots, validation_snapshot, test_snapshots, h_days)

    train = build_panel(tx, train_snapshots, h_days)
    validation = build_panel(tx, [validation_snapshot], h_days)
    test = build_panel(tx, test_snapshots, h_days)
    train = train[~train["single_order"]]
    validation = validation[~validation["single_order"]]

    baseline = fit_baseline_logistic(pd.concat([train, validation], ignore_index=True))
    lgbm, best_params = fit_tuned_lgbm(train, validation, n_iter)

    rows, preds = [], []
    for snap, part in test.groupby("snapshot_date"):
        single = part[part["single_order"]]
        main = part[~part["single_order"]].copy()
        y = main["churned"].to_numpy()
        base_p = predict_baseline(baseline, main)
        lgb_p = predict_lgbm(lgbm, main)

        common = {
            "snapshot_date": snap,
            "h_days": h_days,
            "n_single_order": len(single),
            "single_order_base_rate": float(single["churned"].mean()) if len(single) else np.nan,
        }
        rows.append({**common, "model": "logistic_baseline", **evaluate(y, base_p, base_p)})
        rows.append({**common, "model": "lightgbm_tuned", **evaluate(y, lgb_p, lgb_p)})

        main["proba_baseline"] = base_p
        main["proba_lgbm"] = lgb_p
        preds.append(main)

    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), gain_importance(lgbm), best_params


def h_sensitivity(tx, train_snapshots, validation_snapshot, test_snapshots, h_values, n_iter) -> pd.DataFrame:
    frames = [
        run_experiment(tx, train_snapshots, validation_snapshot, test_snapshots, h, n_iter)[0]
        for h in h_values
    ]
    return pd.concat(frames, ignore_index=True)


def monthly_churn_rate(tx: pd.DataFrame, h_days: int, start: str) -> pd.DataFrame:
    data_end = tx["invoice_date"].max().normalize()
    starts = pd.date_range(start, data_end - pd.Timedelta(days=h_days), freq="MS")
    rows = []
    for snap in starts:
        main = build_snapshot(tx, snap, h_days)
        main = main[~main["single_order"]]
        rows.append({"snapshot_date": snap, "n": len(main), "churn_rate": main["churned"].mean()})
    return pd.DataFrame(rows)
