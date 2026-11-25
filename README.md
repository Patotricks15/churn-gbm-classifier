# churn-gbm-classifier (CH02)

**Question:** Do richer behavioural features beat the baseline?

This repository trains a LightGBM churn classifier with `RandomizedSearchCV` on time-ordered snapshots of Olist Brazilian E-Commerce, and compares it with a logistic regression on RFM features (the CH01 baseline).

**Status:** the code is tested with unit tests and a smoke run on synthetic data. **Results are not yet produced.** The notebook has not been executed, because the data is not on this machine. Headline numbers will be added after a Kaggle run.

## Data

- **Dataset:** Olist Brazilian E-Commerce, Kaggle slug [`olistbr/brazilian-ecommerce`](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
- **License (per Kaggle metadata):** CC BY-NC-SA 4.0. Non-commercial use only.
- **Read path:** `/kaggle/input/brazilian-ecommerce/` (see [data/README.md](data/README.md)).
- **Metadata read:** 2026-11-08 (plan Appendix A). No files were downloaded.

## Churn definition

- **Churn window:** `H = 90` days. A customer churns if they make no purchase in `[snapshot, snapshot + 90 days)`.
- **Features:** only orders strictly before the snapshot. They are RFM features, tenure, mean days between orders, mean freight and distinct products.
- **Cleaning before labelling:** orders with status `canceled` or `unavailable` are removed. Olist has no separate return table. Order lines with price ≤ 0 are removed.
- **Population:** customers with at least one order before the snapshot, identified by `customer_unique_id`.
- **Single-order customers:** kept in the panel and reported separately. They are excluded from training and from the main metrics.

## Snapshots and validation

- **Train snapshots:** 2017-04-01, 2017-07-01, 2017-10-01.
- **Validation snapshot:** 2018-01-01. It tunes the LightGBM hyperparameters (search scored by PR-AUC).
- **Final fit:** train + validation snapshots, for both models.
- **Test snapshots:** 2018-04-01, 2018-06-01. The headline result is the latest test snapshot, 2018-06-01.
- **Time-ordered only.** The code checks that the last training label window (2018-01-01 + 90 days = 2018-04-01) ends before the first test snapshot. The check raises an error otherwise, so training labels never overlap the test period.
- The last label window (2018-06-01 + 90 days = 2018-08-30) fits inside the dataset.

## Models

| Model | Definition |
|---|---|
| Logistic regression (baseline) | `log1p` → standard scaling → `LogisticRegression(random_state=42)` on the five CH01 features. |
| LightGBM (tuned) | `LGBMClassifier` with `scale_pos_weight = n_neg / n_pos`, `random_state=42`. `RandomizedSearchCV` with 25 draws over trees, learning rate, leaves, minimum child samples, subsampling, column sampling and L2 regularisation. The search uses the fixed validation fold (`PredefinedSplit`). |

## Metrics

- **PR-AUC** (primary, churn is imbalanced) and **ROC-AUC**.
- **Brier score** for both models. `scale_pos_weight` inflates LightGBM probabilities, so its Brier score and reliability curve are expected to look worse. They are reported as-is, not recalibrated.
- **Lift in the top decile:** churn rate among the top 10% of scores divided by the base rate.

## Outputs

Running the script writes to `reports/`:

- `ch02_metrics.csv`: per-snapshot metrics for both models.
- `ch02_h_sensitivity.csv`: the same metrics for `H` in {30, 60, 90} days.
- `ch02_monthly_churn.csv`: monthly churn rate for the main population.
- `ch02_feature_importance.csv` and `ch02_best_params.txt`: gain importance and the selected hyperparameters.
- `figures/`: PR curves, reliability curves, lift by decile, feature importance by gain, and monthly churn rate. Titles state the dataset, snapshot date, `H`, and the model version.

## Results

**Not yet run on the real dataset.** The tests and a smoke run on randomly generated data only confirm that the code runs. Those numbers are not results and are not reported here.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make test
make reproduce INPUT_DIR=/kaggle/input/brazilian-ecommerce
```

The notebook [notebooks/ch02_gbm_classifier.ipynb](notebooks/ch02_gbm_classifier.ipynb) runs the same script inside a Kaggle Notebook with the dataset attached. It clones this repository, so internet access must be enabled in the Kaggle session. It has not been executed locally, because the data is not on this machine.

## Limitation

Olist is a Brazilian marketplace where most customers buy once. Only a small share of customers stay in the main population, so the main metrics rest on few repeat buyers. The results may not transfer to subscription businesses without validation.

## Layout

```text
src/churn/       data cleaning, features, snapshot labels, models, metrics, plots, pipeline
scripts/         run_gbm.py (end-to-end run)
notebooks/       Kaggle notebook that runs the script on /kaggle/input
tests/           label rule, snapshot cut (leakage), label gap, lift calculation, cleaning
reports/         metrics CSVs and figures
data/            git-ignored except data/README.md
```
