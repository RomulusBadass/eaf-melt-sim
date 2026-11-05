"""Synthetic heat records for a 3 t electric arc furnace with basic lining.

The simulated route is: charge -> melting -> oxygen blowing -> slag adjustment
-> ferroalloy addition in the ladle -> tapping -> quality check against
GOST 977-88 (steel 35L, group 1 castings, 3 t).

At every stage the operator aims at a value derived from the previous
measurements and adds instrument or handling noise. Ferroalloy recovery in the
ladle degrades smoothly with overheating, excess oxygen and low slag
basicity, and carries an unexplained heat-to-heat residual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHARGE_KG = 3300.0
LIQUID_STEEL_KG = 3000.0

SCRAP_2A_RANGE = (1600.0, 2500.0)
SCRAP_6A_RANGE = (400.0, 1300.0)
PIG_IRON_RANGE = (200.0, 400.0)
OXYGEN_RANGE = (35.0, 75.0)
BASICITY_RANGE = (1.8, 2.9)
TAP_TEMP_RANGE = (1530.0, 1660.0)
FESI_RANGE = (15.0, 30.0)
FEMN_RANGE = (20.0, 40.0)

PIG_IRON_CARBON = 0.042
SCRAP_CARBON = 0.0015
CARBON_REMOVED_PER_M3 = 0.0025

FESI_SI_CONTENT = 0.75
FEMN_MN_CONTENT = 0.78
NOMINAL_SI_RECOVERY = 0.90
NOMINAL_MN_RECOVERY = 0.80
RESIDUAL_SI = 0.05
RESIDUAL_MN = 0.12

TARGET_CARBON = 0.36
TARGET_SI = 0.36
TARGET_MN = 0.68
TARGET_SUPERHEAT = 57.5
TARGET_SP = 0.024

SP_BASE = 0.010
SP_PER_6A_SHARE = 0.14
SP_BASICITY_SLOPE = 0.010

GOST_CARBON = (0.32, 0.40)
GOST_SILICON = (0.20, 0.52)
GOST_MANGANESE = (0.45, 0.90)
GOST_SP_MAX = 0.040
GOST_SUPERHEAT = (45.0, 70.0)

COLUMNS: dict[str, dict[str, str]] = {
    "melt_id": {
        "unit": "-",
        "en": "Heat identifier.",
        "ru": "Номер плавки.",
    },
    "scrap_2a_kg": {
        "unit": "kg",
        "en": "Bulk scrap, category 2A, in the charge.",
        "ru": "Кусковой лом категории 2А в шихте.",
    },
    "scrap_6a_kg": {
        "unit": "kg",
        "en": "Scrap category 6A in the charge; Briquetted steel turnings.",
        "ru": "Лом категории 6А в шихте; Брикеты стальной стружки.",
    },
    "pig_iron_kg": {
        "unit": "kg",
        "en": "Pig iron used as carburiser.",
        "ru": "Передельный чугун как науглероживатель.",
    },
    "carbon_pct": {
        "unit": "%",
        "en": "Carbon in the bath after oxygen blowing.",
        "ru": "Углерод в ванне после продувки кислородом.",
    },
    "oxygen_m3": {
        "unit": "m3",
        "en": "Oxygen blown during the boil.",
        "ru": "Объём кислорода, вдутого на этапе кипения.",
    },
    "slag_basicity": {
        "unit": "-",
        "en": "Slag basicity before tapping.",
        "ru": "Основность шлака перед выпуском.",
    },
    "sulfur_phosphorus_pct": {
        "unit": "%",
        "en": "Combined sulfur and phosphorus in the steel.",
        "ru": "Суммарное содержание серы и фосфора в стали.",
    },
    "fesi75_kg": {
        "unit": "kg",
        "en": "FeSi75 added to the ladle.",
        "ru": "Ферросилиций ФС75, добавленный в ковш.",
    },
    "femn78_kg": {
        "unit": "kg",
        "en": "FeMn78 added to the ladle.",
        "ru": "Ферромарганец ФМн78, добавленный в ковш.",
    },
    "tap_temp_c": {
        "unit": "C",
        "en": "Tapping temperature.",
        "ru": "Температура выпуска.",
    },
    "mn_recovery": {
        "unit": "fraction",
        "en": "Manganese recovery from FeMn78 (not observable before tapping).",
        "ru": "Усвоение марганца из ФМн78 (до выпуска наблюдать нельзя).",
    },
    "si_recovery": {
        "unit": "fraction",
        "en": "Silicon recovery from FeSi75 (not observable before tapping).",
        "ru": "Усвоение кремния из ФС75 (до выпуска наблюдать нельзя).",
    },
    "silicon_pct": {
        "unit": "%",
        "en": "Final silicon in the ladle.",
        "ru": "Итоговое содержание кремния в ковше.",
    },
    "manganese_pct": {
        "unit": "%",
        "en": "Final manganese in the ladle.",
        "ru": "Итоговое содержание марганца в ковше.",
    },
    "liquidus_c": {
        "unit": "C",
        "en": "Liquidus temperature from the final composition.",
        "ru": "Температура ликвидус по итоговому химическому составу.",
    },
    "superheat_c": {
        "unit": "C",
        "en": "Tapping temperature minus liquidus.",
        "ru": "Температура выпуска минус ликвидус.",
    },
    "gost_pass": {
        "unit": "0/1",
        "en": "1 if carbon, silicon, manganese, sulfur+phosphorus and superheat are all within limits.",
        "ru": "1, если углерод, кремний, марганец, сера+фосфор и перегрев одновременно в допуске.",
    },
}


def _sigmoid_penalty(x, center: float, scale: float, max_penalty: float):
    return max_penalty / (1.0 + np.exp(-(x - center) / scale))


def _dose(target: float, residual: float, content: float, recovery: float) -> float:
    return (target - residual) / 100.0 * LIQUID_STEEL_KG / (content * recovery)


def _pickup(dose, content: float, recovery) -> float:
    return dose * content * recovery / LIQUID_STEEL_KG * 100.0


def _liquidus(carbon, silicon, manganese):
    return 1538.0 - 88.0 * carbon - 8.0 * silicon - 5.0 * manganese


def _sample_charge(rng: np.random.Generator, n: int):
    scrap_2a: list[float] = []
    scrap_6a: list[float] = []
    pig_iron: list[float] = []
    carbon_per_kg_pig = PIG_IRON_CARBON - SCRAP_CARBON
    while len(scrap_2a) < n:
        batch = n * 3
        target_carbon = rng.uniform(0.42, 0.50, batch)
        pig_calc = (target_carbon / 100.0 - SCRAP_CARBON) * CHARGE_KG / carbon_per_kg_pig
        pig = np.clip(pig_calc + rng.normal(0, 4, batch), *PIG_IRON_RANGE)
        s6 = rng.uniform(*SCRAP_6A_RANGE, batch)
        s2 = CHARGE_KG - pig - s6
        keep = (s2 >= SCRAP_2A_RANGE[0]) & (s2 <= SCRAP_2A_RANGE[1])
        scrap_2a.extend(s2[keep])
        scrap_6a.extend(s6[keep])
        pig_iron.extend(pig[keep])
    return (
        np.array(scrap_2a[:n]),
        np.array(scrap_6a[:n]),
        np.array(pig_iron[:n]),
    )


def simulate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` heats. The output is fully determined by ``seed``."""
    rng = np.random.default_rng(seed)

    scrap_2a, scrap_6a, pig_iron = _sample_charge(rng, n)
    carbon_melt = (
        (pig_iron * PIG_IRON_CARBON + (scrap_2a + scrap_6a) * SCRAP_CARBON)
        / CHARGE_KG
        * 100.0
    )
    contamination = rng.uniform(0.7, 1.4, n)

    oxygen_plan = (carbon_melt - TARGET_CARBON) / CARBON_REMOVED_PER_M3
    oxygen = np.clip(oxygen_plan + rng.normal(0, 2, n), *OXYGEN_RANGE)
    carbon = carbon_melt - oxygen * CARBON_REMOVED_PER_M3

    contamination_assay = contamination + rng.normal(0, 0.04, n)
    sp_load = scrap_6a / CHARGE_KG * SP_PER_6A_SHARE
    basicity_plan = (SP_BASE + sp_load * contamination_assay - TARGET_SP) / SP_BASICITY_SLOPE
    basicity = np.clip(basicity_plan + rng.normal(0, 0.08, n), *BASICITY_RANGE)
    sulfur_phosphorus = np.clip(
        SP_BASE + sp_load * contamination - basicity * SP_BASICITY_SLOPE, 0.006, None
    )

    fesi_plan = _dose(TARGET_SI, RESIDUAL_SI, FESI_SI_CONTENT, NOMINAL_SI_RECOVERY)
    femn_plan = _dose(TARGET_MN, RESIDUAL_MN, FEMN_MN_CONTENT, NOMINAL_MN_RECOVERY)
    fesi = np.clip(fesi_plan + rng.normal(0, 1.2, n), *FESI_RANGE)
    femn = np.clip(femn_plan + rng.normal(0, 1.8, n), *FEMN_RANGE)

    si_plan = RESIDUAL_SI + _pickup(fesi_plan, FESI_SI_CONTENT, NOMINAL_SI_RECOVERY)
    mn_plan = RESIDUAL_MN + _pickup(femn_plan, FEMN_MN_CONTENT, NOMINAL_MN_RECOVERY)
    liquidus_plan = _liquidus(TARGET_CARBON, si_plan, mn_plan)
    tap_temp = np.clip(
        liquidus_plan + TARGET_SUPERHEAT + rng.normal(0, 5, n), *TAP_TEMP_RANGE
    )

    oxygen_r = np.round(oxygen, 1)
    basicity_r = np.round(basicity, 2)
    tap_temp_r = np.round(tap_temp, 0)

    mn_loss = (
        _sigmoid_penalty(tap_temp_r, 1640, 8, 0.15)
        + _sigmoid_penalty(oxygen_r, 60, 5, 0.15)
        + _sigmoid_penalty(-basicity_r, -2.0, 0.1, 0.10)
    )
    si_loss = (
        _sigmoid_penalty(tap_temp_r, 1640, 8, 0.10)
        + _sigmoid_penalty(oxygen_r, 60, 5, 0.15)
        + _sigmoid_penalty(-basicity_r, -2.0, 0.1, 0.05)
    )
    furnace_noise = rng.normal(-0.01, 0.02, n)
    mn_recovery = np.clip(NOMINAL_MN_RECOVERY - mn_loss + furnace_noise, 0.20, NOMINAL_MN_RECOVERY)
    si_recovery = np.clip(NOMINAL_SI_RECOVERY - si_loss + furnace_noise * 0.7, 0.20, NOMINAL_SI_RECOVERY)

    silicon = RESIDUAL_SI + _pickup(fesi, FESI_SI_CONTENT, si_recovery)
    manganese = RESIDUAL_MN + _pickup(femn, FEMN_MN_CONTENT, mn_recovery)

    carbon_r = np.round(carbon, 3)
    silicon_r = np.round(silicon, 3)
    manganese_r = np.round(manganese, 3)
    liquidus = np.round(_liquidus(carbon_r, silicon_r, manganese_r), 1)
    superheat = np.round(tap_temp_r - liquidus, 1)

    df = pd.DataFrame(
        {
            "melt_id": np.arange(1, n + 1),
            "scrap_2a_kg": np.round(scrap_2a, 0),
            "scrap_6a_kg": np.round(scrap_6a, 0),
            "pig_iron_kg": np.round(pig_iron, 0),
            "carbon_pct": carbon_r,
            "oxygen_m3": oxygen_r,
            "slag_basicity": basicity_r,
            "sulfur_phosphorus_pct": np.round(sulfur_phosphorus, 3),
            "fesi75_kg": np.round(fesi, 1),
            "femn78_kg": np.round(femn, 1),
            "tap_temp_c": tap_temp_r,
            "mn_recovery": np.round(mn_recovery, 3),
            "si_recovery": np.round(si_recovery, 3),
            "silicon_pct": silicon_r,
            "manganese_pct": manganese_r,
            "liquidus_c": liquidus,
            "superheat_c": superheat,
        }
    )
    passed = (
        df["carbon_pct"].between(*GOST_CARBON)
        & df["silicon_pct"].between(*GOST_SILICON)
        & df["manganese_pct"].between(*GOST_MANGANESE)
        & (df["sulfur_phosphorus_pct"] <= GOST_SP_MAX)
        & df["superheat_c"].between(*GOST_SUPERHEAT)
    )
    df["gost_pass"] = passed.astype(int)
    return df[list(COLUMNS)]


def summarize(df: pd.DataFrame) -> str:
    lines = [f"heats: {len(df)}", f"gost_pass rate: {df['gost_pass'].mean():.1%}", "", "reject causes (share of heats outside the limit):"]
    checks = {
        "carbon": ~df["carbon_pct"].between(*GOST_CARBON),
        "silicon": ~df["silicon_pct"].between(*GOST_SILICON),
        "manganese": ~df["manganese_pct"].between(*GOST_MANGANESE),
        "sulfur+phosphorus": df["sulfur_phosphorus_pct"] > GOST_SP_MAX,
        "superheat": ~df["superheat_c"].between(*GOST_SUPERHEAT),
    }
    for name, mask in checks.items():
        lines.append(f"  {name:<18} {mask.mean():6.1%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic EAF heat records.")
    parser.add_argument("--n", type=int, default=2000, help="number of heats")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=Path, default=Path("data/eaf_melts_2000.csv"))
    args = parser.parse_args()

    df = simulate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved {len(df)} rows to {args.out}\n")
    print(summarize(df))


if __name__ == "__main__":
    main()
