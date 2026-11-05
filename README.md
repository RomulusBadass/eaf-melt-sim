# eaf-melt-sim

A simulator of electric arc furnace heats and a small CatBoost model that predicts
the minimum ferromanganese (FeMn78) dose needed to reach the lower manganese limit.

[Русская версия](README.ru.md) · [Live demo (Hugging Face)](https://huggingface.co/spaces/RomulusBadass/eaf-melt-sim) · [Dataset (Kaggle)](https://www.kaggle.com/datasets/romulusbadass/synthetic-eaf-steel-melts-35l-gost-977-88)

## What this is

The generator produces synthetic heats of steel 35L (GOST 977-88, group 1 castings,
3 t) in a 3 t electric arc furnace with basic lining. The charge is scrap of
categories 2A and 6A (GOST 2787-2019) plus pig iron, fixed at 3300 kg. At each stage the
simulated operator aims at a value computed from the previous measurement and adds
noise; ferroalloy recovery then degrades with overheating, excess oxygen and low slag
basicity. Each heat is checked against five GOST 977-88 conditions.

The model uses only quantities known before the ferroalloy is added and predicts the
FeMn78 dose that lands manganese exactly on the lower limit (0.45 %).

**This is a simulator, not a measurement.** The data comes from hand-written rules, so
the model learns those rules. It is a starting point for method development and teaching,
not evidence about any real furnace.

## Results on the bundled dataset

2000 heats, seed 42, test split of 400 heats not used for training or early stopping.

| Metric | Value |
| --- | --- |
| Heats passing all GOST 977-88 checks | 89.4 % |
| Test MAE of minimum dose | 0.352 kg |
| MAE of always predicting the train median | 0.873 kg |
| Test R² | 0.836 |
| Mean headroom over the minimum dose | 9.84 kg per heat |

Headroom is the dose actually added minus the dose that lands exactly on 0.45 % Mn. It
leaves no safety margin, so it is an upper bound on possible savings, not a forecast.

Slag basicity carries about 70 % of the feature importance, oxygen about 11 %.

## Quick start

```bash
pip install -r requirements.txt

python -m eafsim.generator --n 2000 --seed 42   # writes data/eaf_melts_2000.csv
python -m eafsim.model                          # writes models/fmn78_model.cbm and metrics.json
streamlit run app.py                            # interactive app
pytest                                          # tests
```

## Layout

```
eafsim/generator.py   simulator and column dictionary
eafsim/model.py       target definition, training, inference
app.py                Streamlit app (English and Russian)
data/                 generated heats
models/               trained model (CatBoost native format) and metrics
tests/                pipeline tests
```

## Limitations

- In the generated data the tapping temperature stays below the 1640 °C threshold at
  which the overheating penalty starts, and oxygen almost never exceeds 60 m³. Those
  loss mechanisms exist in the code but are barely active; slag basicity dominates.
- FeSi75 doses sit almost entirely at the 15 kg floor, so silicon has little variation.
- The headroom figure assumes manganese is brought exactly to the lower limit.
- No real furnace data was used to calibrate any coefficient.

## License

Code: MIT, see [LICENSE](LICENSE). The generated dataset is released under CC0 1.0.
