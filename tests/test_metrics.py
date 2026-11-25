import numpy as np
import pytest

from churn.metrics import decile_table, evaluate, top_fraction_lift


def test_lift_top_decile_matches_hand_calculation():
    y = np.array([1] * 5 + [0] * 95)
    score = np.arange(100)[::-1]  # the first 10 scores are the highest
    # top 10% = 10 rows, 5 churned: rate 0.5, base rate 0.05, lift 10
    assert top_fraction_lift(y, score) == pytest.approx(10.0)


def test_lift_uses_ceiling_for_small_samples():
    y = np.array([1, 0, 0])
    score = np.array([0.9, 0.5, 0.1])
    # ceil(0.1 * 3) = 1 row in top decile: rate 1.0, base rate 1/3, lift 3
    assert top_fraction_lift(y, score) == pytest.approx(3.0)


def test_decile_table_top_row_matches_top_decile_lift():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=1000)
    score = y + rng.normal(0, 0.5, size=1000)
    table = decile_table(y, score)
    assert list(table["decile"]) == list(range(1, 11))
    assert table["n"].sum() == 1000
    assert table.iloc[0]["lift"] == pytest.approx(top_fraction_lift(y, score), rel=1e-9)


def test_evaluate_reports_brier_only_with_probabilities():
    y = np.array([0, 1, 0, 1])
    score = np.array([0.1, 0.8, 0.3, 0.6])
    assert np.isnan(evaluate(y, score)["brier"])
    assert evaluate(y, score, score)["brier"] == pytest.approx(np.mean((score - y) ** 2))
