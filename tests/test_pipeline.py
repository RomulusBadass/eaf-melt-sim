import numpy as np
import pandas as pd
import pytest

from eafsim.generator import (
    COLUMNS,
    PIG_IRON_RANGE,
    SCRAP_2A_RANGE,
    SCRAP_6A_RANGE,
    simulate,
)
from eafsim.model import FEATURES, TARGET, add_targets, load_model, predict_min_dose, train


@pytest.fixture(scope="module")
def heats() -> pd.DataFrame:
    return simulate(n=600, seed=7)


def test_simulation_is_reproducible():
    pd.testing.assert_frame_equal(simulate(n=200, seed=1), simulate(n=200, seed=1))


def test_different_seeds_differ():
    assert not simulate(n=200, seed=1).equals(simulate(n=200, seed=2))


def test_columns_are_documented(heats):
    assert list(heats.columns) == list(COLUMNS)


def test_charge_is_fixed_and_within_ranges(heats):
    total = heats["scrap_2a_kg"] + heats["scrap_6a_kg"] + heats["pig_iron_kg"]
    assert total.between(3300 - 2, 3300 + 2).all()
    assert heats["scrap_2a_kg"].between(*SCRAP_2A_RANGE).all()
    assert heats["scrap_6a_kg"].between(*SCRAP_6A_RANGE).all()
    assert heats["pig_iron_kg"].between(*PIG_IRON_RANGE).all()


def test_both_outcomes_present(heats):
    rate = heats["gost_pass"].mean()
    assert 0.5 < rate < 1.0


def test_recovery_has_many_distinct_values(heats):
    assert heats["mn_recovery"].nunique() > 50


def test_min_dose_is_below_actual_dose_on_average(heats):
    enriched = add_targets(heats)
    assert (enriched["femn78_kg"] - enriched[TARGET]).mean() > 0


def test_train_save_load_predict(tmp_path, heats):
    model, metrics = train(heats, seed=0, iterations=150)
    assert np.isfinite(metrics["test_mae_kg"])
    assert metrics["test_mae_kg"] < metrics["baseline_mae_kg"]

    path = tmp_path / "model.cbm"
    model.save_model(str(path))
    restored = load_model(path)

    row = heats.iloc[0][FEATURES].to_dict()
    assert predict_min_dose(restored, **row) == pytest.approx(predict_min_dose(model, **row))
