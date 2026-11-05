import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from eafsim.generator import CHARGE_KG, COLUMNS, FEMN_RANGE, SCRAP_2A_RANGE
from eafsim.model import FEATURES, HEADROOM, add_targets, load_model, predict_min_dose

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "eaf_melts_2000.csv"
MODEL_PATH = ROOT / "models" / "fmn78_model.cbm"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="EAF melt simulator", page_icon="🔥", layout="wide")

TEXT = {
    "en": {
        "title": "EAF melt simulator: minimum FeMn78 dose",
        "subtitle": (
            "Synthetic heats of steel 35L in a 3 t electric arc furnace, and a CatBoost "
            "model that predicts the smallest FeMn78 dose that still reaches the lower "
            "manganese limit."
        ),
        "tab_overview": "Overview",
        "tab_dataset": "Dataset",
        "tab_calc": "Calculator",
        "tab_about": "About",
        "m_heats": "Heats",
        "m_pass": "GOST pass rate",
        "m_mae": "Test MAE, kg",
        "m_mae_help": "Mean absolute error of the minimum dose on a held-out test split.",
        "m_baseline": "Baseline MAE, kg",
        "m_baseline_help": "Error of always predicting the training median.",
        "m_headroom": "Mean headroom, kg",
        "m_headroom_help": (
            "Mean of (dose actually added - dose that lands exactly on the lower "
            "manganese limit). An upper bound: it leaves no safety margin."
        ),
        "importance": "Feature importance",
        "importance_x": "Importance, %",
        "headroom_hist": "Headroom per heat",
        "headroom_x": "Headroom, kg FeMn78",
        "overview_note": (
            "All figures come from synthetic data produced by the generator in this "
            "repository. They describe the simulator, not a real furnace."
        ),
        "dataset_header": "Generated heats",
        "download": "Download CSV",
        "columns_header": "Columns",
        "col_name": "column",
        "col_unit": "unit",
        "col_desc": "description",
        "calc_header": "Minimum FeMn78 dose for a heat",
        "calc_hint": (
            "Inputs are known before the ferroalloy is added. The features are "
            "correlated in the training data, so unusual combinations are extrapolation."
        ),
        "s_6a": "Scrap 6A, kg",
        "s_pig": "Pig iron, kg",
        "s_o2": "Oxygen, m3",
        "s_basicity": "Slag basicity",
        "s_temp": "Tapping temperature, C",
        "s_carbon": "Carbon after blowing, %",
        "derived_2a": "Scrap 2A (charge fixed at 3300 kg), kg",
        "warn_2a": "Scrap 2A is outside the range seen in training; the estimate is less reliable.",
        "ref_dose": "Reference dose actually used, kg",
        "margin": "Safety margin, kg",
        "r_min": "Predicted minimum dose",
        "r_margin": "With safety margin",
        "r_delta": "Reference minus recommended",
        "warn_low": (
            "The generator only produces doses in the {lo:.0f}-{hi:.0f} kg range. A "
            "recommendation below {lo:.0f} kg is outside what the data covers."
        ),
        "about": """
### What the data represents

Synthetic data generated for casting of group 1 made of steel 35L per GOST 977-88,
3 t, melted in a 3 t electric arc furnace with basic lining. Charge: scrap
categories 2A and 6A per GOST 2787-2019, plus pig iron as a carburiser; total charge
is fixed at 3300 kg.

At each stage the simulated operator aims at a value computed from the previous
measurement and adds noise: pig iron for the target carbon, oxygen for the target
carbon after blowing, slag basicity for the contamination of the scrap, ferroalloy
doses for the target composition at nominal recovery, tapping temperature for the
liquidus plus a target superheat. Actual recovery of Mn and Si then drops smoothly
with overheating, excess oxygen and low basicity, plus an unexplained residual.

### What the model predicts

The FeMn78 dose that lands manganese exactly on the lower limit (0.45 %), given the
recovery the heat actually had. Features are limited to what is known before the
ferroalloy is added.

### Limitations

- The data is produced by hand-written rules, not measured. The model learns those
  rules; it says nothing about a real furnace until it is retrained on real heats.
- In the generated data the tapping temperature stays below the 1640 C threshold and
  oxygen almost never exceeds 60 m3, so those loss mechanisms are present in the code
  but hardly ever active. Slag basicity dominates the result.
- FeSi75 doses sit almost entirely at the 15 kg floor, so silicon has little variation.
- Headroom is measured against the exact lower manganese limit. A real process needs a
  margin, so the headroom is an upper bound on possible savings, not a forecast.
""",
    },
    "ru": {
        "title": "Симулятор плавки в ДСП: минимальная добавка ФМн78",
        "subtitle": (
            "Синтетические плавки стали 35Л в трёхтонной дуговой печи и модель CatBoost, "
            "которая предсказывает наименьшую добавку ФМн78, достаточную для нижней "
            "границы по марганцу."
        ),
        "tab_overview": "Обзор",
        "tab_dataset": "Данные",
        "tab_calc": "Калькулятор",
        "tab_about": "О проекте",
        "m_heats": "Плавок",
        "m_pass": "Доля годных по ГОСТ",
        "m_mae": "MAE на тесте, кг",
        "m_mae_help": "Средняя абсолютная ошибка минимальной добавки на отложенной тестовой выборке.",
        "m_baseline": "MAE базовой модели, кг",
        "m_baseline_help": "Ошибка, если всегда предсказывать медиану обучающей выборки.",
        "m_headroom": "Средний запас, кг",
        "m_headroom_help": (
            "Среднее от (фактически добавленная доза - доза, ровно доводящая марганец до "
            "нижней границы). Это верхняя оценка: запаса надёжности в ней нет."
        ),
        "importance": "Важность признаков",
        "importance_x": "Важность, %",
        "headroom_hist": "Запас по плавкам",
        "headroom_x": "Запас, кг ФМн78",
        "overview_note": (
            "Все цифры получены на синтетических данных генератора из этого репозитория. "
            "Они описывают симулятор, а не реальную печь."
        ),
        "dataset_header": "Сгенерированные плавки",
        "download": "Скачать CSV",
        "columns_header": "Колонки",
        "col_name": "колонка",
        "col_unit": "единицы",
        "col_desc": "описание",
        "calc_header": "Минимальная доза ФМн78 для плавки",
        "calc_hint": (
            "Входные значения известны до добавки ферросплава. В обучающих данных признаки "
            "связаны между собой, поэтому необычные сочетания - это экстраполяция."
        ),
        "s_6a": "Лом 6А, кг",
        "s_pig": "Чугун, кг",
        "s_o2": "Кислород, м3",
        "s_basicity": "Основность шлака",
        "s_temp": "Температура выпуска, C",
        "s_carbon": "Углерод после продувки, %",
        "derived_2a": "Лом 2А (завалка фиксирована 3300 кг), кг",
        "warn_2a": "Лом 2А вне диапазона обучающих данных; оценка менее надёжна.",
        "ref_dose": "Фактически используемая доза, кг",
        "margin": "Запас надёжности, кг",
        "r_min": "Предсказанная минимальная доза",
        "r_margin": "С запасом надёжности",
        "r_delta": "Фактическая минус рекомендуемая",
        "warn_low": (
            "Генератор выдаёт дозы только в диапазоне {lo:.0f}-{hi:.0f} кг. Рекомендация "
            "ниже {lo:.0f} кг выходит за пределы данных."
        ),
        "about": """
### Что описывают данные

Синтетические данные сгенерированы под выплавку отливки 1 группы из стали 35Л по
ГОСТ 977-88 массой 3 т в трёхтонной дуговой печи с основной футеровкой. Шихта: лом
категорий 2А и 6А по ГОСТ 2787-2019 и передельный чугун как науглероживатель; общая
завалка фиксирована и равна 3300 кг.

На каждом этапе симулированный сталевар целится в значение, рассчитанное по предыдущему
измерению, и добавляет шум: чугун под целевой углерод, кислород под целевой углерод после
продувки, основность шлака под загрязнённость лома, дозы ферросплавов под целевой состав
при номинальном усвоении, температуру выпуска под ликвидус плюс целевой перегрев.
Фактическое усвоение Mn и Si затем плавно падает при перегреве, избытке кислорода и низкой
основности, плюс необъяснимый остаток.

### Что предсказывает модель

Дозу ФМн78, которая ровно доводит марганец до нижней границы (0,45 %) при том усвоении,
которое плавка реально имела. Признаки ограничены тем, что известно до добавки ферросплава.

### Ограничения

- Данные получены по написанным вручную правилам, а не измерены. Модель учит эти правила
  и ничего не говорит о реальной печи, пока её не переобучат на реальных плавках.
- В сгенерированных данных температура выпуска остаётся ниже порога 1640 C, а кислород
  почти никогда не превышает 60 м3, поэтому эти механизмы потерь есть в коде, но почти не
  работают. Результат определяет основность шлака.
- Дозы ФС75 почти целиком лежат на нижней границе 15 кг, поэтому у кремния мало вариации.
- Запас считается относительно точной нижней границы по марганцу. В реальном процессе нужен
  запас надёжности, поэтому это верхняя оценка возможной экономии, а не прогноз.
""",
    },
}


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def get_model(path: Path):
    return load_model(path)


missing = [p.name for p in (DATA_PATH, MODEL_PATH, METRICS_PATH) if not p.exists()]
if missing:
    st.error(
        f"Missing files: {', '.join(missing)}. Run "
        "`python -m eafsim.generator` and `python -m eafsim.model` first."
    )
    st.stop()

lang_label = st.radio("Language / Язык", ["English", "Русский"], horizontal=True)
t = TEXT["en" if lang_label == "English" else "ru"]
lang = "en" if lang_label == "English" else "ru"

df = load_data(DATA_PATH)
model = get_model(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text())
enriched = add_targets(df)

st.title(t["title"])
st.markdown(t["subtitle"])

tab_overview, tab_dataset, tab_calc, tab_about = st.tabs(
    [t["tab_overview"], t["tab_dataset"], t["tab_calc"], t["tab_about"]]
)

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(t["m_heats"], f"{metrics['n_rows']:,}".replace(",", " "))
    c2.metric(t["m_pass"], f"{metrics['gost_pass_rate']:.1%}")
    c3.metric(t["m_mae"], f"{metrics['test_mae_kg']:.3f}", help=t["m_mae_help"])
    c4.metric(t["m_baseline"], f"{metrics['baseline_mae_kg']:.3f}", help=t["m_baseline_help"])
    c5.metric(t["m_headroom"], f"{metrics['headroom_true_mean_kg']:.2f}", help=t["m_headroom_help"])

    left, right = st.columns(2)
    with left:
        st.subheader(t["importance"])
        imp = pd.DataFrame(
            {
                "feature": list(metrics["feature_importance"]),
                "importance": list(metrics["feature_importance"].values()),
            }
        )
        fig = px.bar(imp, x="importance", y="feature", orientation="h", labels={"importance": t["importance_x"], "feature": ""})
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=340, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader(t["headroom_hist"])
        fig = px.histogram(enriched, x=HEADROOM, nbins=40, labels={HEADROOM: t["headroom_x"]})
        fig.update_layout(height=340, margin={"l": 0, "r": 0, "t": 10, "b": 0}, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    st.caption(t["overview_note"])

with tab_dataset:
    st.subheader(t["dataset_header"])
    st.dataframe(df.head(200), width="stretch")
    st.download_button(t["download"], df.to_csv(index=False).encode("utf-8"), "eaf_melts_2000.csv", "text/csv")
    st.subheader(t["columns_header"])
    schema = pd.DataFrame(
        [
            {t["col_name"]: name, t["col_unit"]: spec["unit"], t["col_desc"]: spec[lang]}
            for name, spec in COLUMNS.items()
        ]
    )
    st.dataframe(schema, width="stretch", hide_index=True)

with tab_calc:
    st.subheader(t["calc_header"])
    st.caption(t["calc_hint"])

    col_a, col_b = st.columns(2)
    with col_a:
        scrap_6a = st.slider(t["s_6a"], 400, 1300, 900, 10)
        pig_iron = st.slider(t["s_pig"], 200, 400, 250, 5)
        scrap_2a = CHARGE_KG - scrap_6a - pig_iron
        st.metric(t["derived_2a"], f"{scrap_2a:.0f}")
        if not SCRAP_2A_RANGE[0] <= scrap_2a <= SCRAP_2A_RANGE[1]:
            st.warning(t["warn_2a"])
        carbon = st.slider(t["s_carbon"], 0.30, 0.42, 0.36, 0.005)
    with col_b:
        oxygen = st.slider(t["s_o2"], 35.0, 75.0, 42.0, 0.5)
        basicity = st.slider(t["s_basicity"], 1.8, 2.9, 2.4, 0.05)
        tap_temp = st.slider(t["s_temp"], 1530, 1660, 1558, 1)
        reference = st.number_input(t["ref_dose"], value=27.0, step=0.5)
        margin = st.slider(t["margin"], 0.0, 3.0, 0.5, 0.1)

    recommended = predict_min_dose(
        model,
        scrap_2a_kg=scrap_2a,
        scrap_6a_kg=scrap_6a,
        pig_iron_kg=pig_iron,
        oxygen_m3=oxygen,
        slag_basicity=basicity,
        tap_temp_c=tap_temp,
        carbon_pct=carbon,
    )
    with_margin = recommended + margin

    r1, r2, r3 = st.columns(3)
    r1.metric(t["r_min"], f"{recommended:.2f} kg")
    r2.metric(t["r_margin"], f"{with_margin:.2f} kg")
    r3.metric(t["r_delta"], f"{reference - with_margin:.2f} kg")
    if recommended < FEMN_RANGE[0]:
        st.info(t["warn_low"].format(lo=FEMN_RANGE[0], hi=FEMN_RANGE[1]))

with tab_about:
    st.markdown(t["about"])
