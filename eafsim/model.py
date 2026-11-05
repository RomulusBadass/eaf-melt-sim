"""Minimum FeMn78 dose model.

Target: the FeMn78 dose that would bring manganese in the ladle exactly to the
lower GOST 977-88 limit, given the manganese recovery the heat actually had.
Features are limited to what is known before the ferroalloy is added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .generator import (
    FEMN_MN_CONTENT,
    GOST_MANGANESE,
    LIQUID_STEEL_KG,
    RESIDUAL_MN,
)

FEATURES = [
    "scrap_2a_kg",
    "scrap_6a_kg",
    "pig_iron_kg",
    "oxygen_m3",
    "slag_basicity",
    "tap_temp_c",
    "carbon_pct",
]
TARGET = "femn78_min_kg"
HEADROOM = "femn78_headroom_kg"


def min_femn_dose(mn_recovery):
    """FeMn78 dose (kg) that lands manganese on the lower GOST limit."""
    needed_pct = GOST_MANGANESE[0] - RESIDUAL_MN
    return needed_pct / 100.0 * LIQUID_STEEL_KG / (FEMN_MN_CONTENT * mn_recovery)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = min_femn_dose(out["mn_recovery"])
    out[HEADROOM] = out["femn78_kg"] - out[TARGET]
    return out


def train(
    df: pd.DataFrame,
    seed: int = 42,
    iterations: int = 1000,
) -> tuple[CatBoostRegressor, dict]:
    """Fit CatBoost on a train/validation/test split and report test metrics.

    Early stopping uses the validation part only; metrics are computed on a
    test part the model never saw.
    """
    data = add_targets(df)
    dev, test = train_test_split(data, test_size=0.2, random_state=seed)
    train_part, val = train_test_split(dev, test_size=0.2, random_state=seed)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )
    model.fit(
        Pool(train_part[FEATURES], train_part[TARGET]),
        eval_set=Pool(val[FEATURES], val[TARGET]),
        early_stopping_rounds=100,
        use_best_model=True,
    )

    pred = model.predict(test[FEATURES])
    mse = mean_squared_error(test[TARGET], pred)
    baseline = np.full(len(test), train_part[TARGET].median())

    importance = model.get_feature_importance()
    metrics = {
        "n_rows": int(len(data)),
        "n_train": int(len(train_part)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "best_iteration": int(model.get_best_iteration()),
        "test_mae_kg": float(mean_absolute_error(test[TARGET], pred)),
        "test_rmse_kg": float(np.sqrt(mse)),
        "test_r2": float(r2_score(test[TARGET], pred)),
        "baseline_mae_kg": float(mean_absolute_error(test[TARGET], baseline)),
        "target_std_kg": float(data[TARGET].std()),
        "gost_pass_rate": float(data["gost_pass"].mean()),
        "headroom_true_mean_kg": float(data[HEADROOM].mean()),
        "headroom_predicted_test_mean_kg": float((test["femn78_kg"] - pred).mean()),
        "feature_importance": {
            name: float(value) for name, value in zip(FEATURES, importance)
        },
    }
    return model, metrics


def load_model(path: str | Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def predict_min_dose(model: CatBoostRegressor, **features: float) -> float:
    row = pd.DataFrame([features], columns=FEATURES)
    return float(model.predict(row)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the minimum FeMn78 dose model.")
    parser.add_argument("--data", type=Path, default=Path("data/eaf_melts_2000.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/fmn78_model.cbm"))
    parser.add_argument("--metrics", type=Path, default=Path("models/metrics.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    model, metrics = train(df, seed=args.seed)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model))
    args.metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"test MAE:      {metrics['test_mae_kg']:.3f} kg")
    print(f"test RMSE:     {metrics['test_rmse_kg']:.3f} kg")
    print(f"test R2:       {metrics['test_r2']:.3f}")
    print(f"baseline MAE:  {metrics['baseline_mae_kg']:.3f} kg (train median)")
    print(f"saved model to {args.model}, metrics to {args.metrics}")


if __name__ == "__main__":
    main()
