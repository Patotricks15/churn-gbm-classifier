import argparse
from pathlib import Path

from churn.config import (
    H_DAYS,
    H_SENSITIVITY_DAYS,
    MODEL_VERSION,
    N_SEARCH_ITER,
    TEST_SNAPSHOTS,
    TRAIN_SNAPSHOTS,
    VALIDATION_SNAPSHOT,
)
from churn.data import load_olist
from churn.features import FEATURE_COLUMNS
from churn.metrics import decile_table
from churn.pipeline import h_sensitivity, monthly_churn_rate, run_experiment
from churn.plots import (
    plot_feature_importance,
    plot_lift_by_decile,
    plot_monthly_churn,
    plot_pr_curves,
    plot_reliability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CH02 LightGBM churn classifier.")
    parser.add_argument("--input-dir", required=True, help="Folder with the Olist CSV files.")
    parser.add_argument("--out-dir", default="reports", help="Directory for metrics and figures.")
    parser.add_argument("--h", type=int, default=H_DAYS, help="Churn window in days.")
    parser.add_argument("--n-iter", type=int, default=N_SEARCH_ITER, help="Random search iterations.")
    args = parser.parse_args()

    out = Path(args.out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    tx = load_olist(args.input_dir)
    results, preds, importance, best_params, lgbm = run_experiment(
        tx, TRAIN_SNAPSHOTS, VALIDATION_SNAPSHOT, TEST_SNAPSHOTS, args.h, args.n_iter
    )
    results.to_csv(out / "ch02_metrics.csv", index=False)
    importance.to_csv(out / "ch02_feature_importance.csv", index=False)
    (out / "ch02_best_params.txt").write_text(repr(best_params) + "\n")
    lgbm.booster_.save_model(str(out / "ch02_lgbm.txt"))
    preds[FEATURE_COLUMNS + ["snapshot_date", "churned", "proba_lgbm"]].to_csv(
        out / "ch02_test_scores.csv", index=False
    )

    sensitivity = h_sensitivity(
        tx, TRAIN_SNAPSHOTS, VALIDATION_SNAPSHOT, TEST_SNAPSHOTS, H_SENSITIVITY_DAYS, args.n_iter
    )
    sensitivity.to_csv(out / "ch02_h_sensitivity.csv", index=False)

    monthly = monthly_churn_rate(tx, args.h, start=TRAIN_SNAPSHOTS[0])
    monthly.to_csv(out / "ch02_monthly_churn.csv", index=False)

    latest = preds[preds["snapshot_date"] == preds["snapshot_date"].max()]
    latest_date = latest["snapshot_date"].iloc[0].date()
    tag = f"Olist | snapshot {latest_date} | H={args.h}d | {MODEL_VERSION}"

    plot_pr_curves(
        latest["churned"],
        {"logistic baseline": latest["proba_baseline"], "LightGBM tuned": latest["proba_lgbm"]},
        f"PR curve | {tag}",
        figures / "pr_curve.png",
    )
    plot_reliability(
        latest["churned"],
        {"logistic baseline": latest["proba_baseline"], "LightGBM tuned": latest["proba_lgbm"]},
        f"Reliability | {tag}",
        figures / "reliability.png",
    )
    decile = decile_table(latest["churned"], latest["proba_lgbm"])
    plot_lift_by_decile(decile, f"Lift by decile (LightGBM) | {tag}", figures / "lift_by_decile.png")
    plot_feature_importance(importance, f"Feature importance by gain | {tag}",
                            figures / "feature_importance.png")
    plot_monthly_churn(monthly, f"Monthly churn rate | Olist | H={args.h}d | {MODEL_VERSION}",
                       figures / "monthly_churn_rate.png")

    cols = ["snapshot_date", "model", "n", "base_rate", "pr_auc", "roc_auc", "lift_top_decile", "brier"]
    print("Per-snapshot results (single-order customers excluded):")
    print(results[cols].to_string(index=False))
    print(f"\nBest LightGBM parameters: {best_params}")


if __name__ == "__main__":
    main()
