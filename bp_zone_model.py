"""Demographic blood-pressure zone model.

Given a person's age, gender and BMI, this module predicts their most likely
blood pressure (the centre) and an individual prediction range (the zone), for
both systolic (SBP) and diastolic (DBP) pressure. It is trained on the
unmedicated adult cohort of the pooled NHANES cycles and serves as the
population reference layer for the demographic vital-sign model.

Run as a script to print example profiles, or import `load_cohort`,
`fit_bp_models` and `bp_zone` to use the model elsewhere (for example, the
Streamlit app).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

DEFAULT_DATA = "data/nhanes_cohort.csv"

REQUIRED_COLUMNS = [
    "age_years", "gender_group", "bmi_kg_m2",
    "sbp_mean_mmhg", "dbp_mean_mmhg", "on_bp_medication",
]


def load_cohort(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Loading the unmedicated adult cohort used to train the models.

    Keeping adults (18-85) with valid blood pressure, BMI and gender, and
    excluding anyone on blood-pressure medication.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at '{path}'. "
            "See data/README.md for the expected file and columns."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data file is missing required columns: {missing}")

    # keeping valid adults only
    df = df[(df.age_years >= 18) & (df.age_years <= 85)]
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # keeping people not on blood-pressure medication
    cohort = df[df.on_bp_medication == 0].copy()
    return cohort


def fit_bp_models(
    cohort: pd.DataFrame,
) -> tuple[RegressionResultsWrapper, RegressionResultsWrapper]:
    """Fitting one linear model per outcome.

    Blood pressure is explained by age, age squared (because pressure rises
    faster at older ages), BMI and gender.
    """
    formula = ("{y} ~ age_years + I(age_years**2) "
               "+ bmi_kg_m2 + C(gender_group)")
    sbp_model = smf.ols(formula.format(y="sbp_mean_mmhg"), data=cohort).fit()
    dbp_model = smf.ols(formula.format(y="dbp_mean_mmhg"), data=cohort).fit()
    return sbp_model, dbp_model


def bp_zone(
    sbp_model: RegressionResultsWrapper,
    dbp_model: RegressionResultsWrapper,
    age: float,
    gender: str,
    bmi: float,
    coverage: float = 0.80,
) -> dict:
    """Returning the centre value and individual range for SBP and DBP.

    The range is a prediction interval (where an individual is likely to fall),
    not a confidence interval of the average, so it reflects real person-to-
    person spread.
    """
    alpha = 1 - coverage
    person = pd.DataFrame([{
        "age_years": age, "gender_group": gender, "bmi_kg_m2": bmi,
    }])

    result = {}
    for name, model in [("SBP", sbp_model), ("DBP", dbp_model)]:
        pred = model.get_prediction(person).summary_frame(alpha=alpha)
        result[name] = {
            "center": round(float(pred["mean"].iloc[0])),
            "low": round(float(pred["obs_ci_lower"].iloc[0])),
            "high": round(float(pred["obs_ci_upper"].iloc[0])),
        }
    return result


def _print_profile(models, age, gender, bmi) -> None:
    """Printing a readable summary for one person."""
    z = bp_zone(models[0], models[1], age, gender, bmi)
    s, d = z["SBP"], z["DBP"]
    print(f"{gender}, age {age}, BMI {bmi}")
    print(f"  SBP: most likely {s['center']}, zone {s['low']} to {s['high']}")
    print(f"  DBP: most likely {d['center']}, zone {d['low']} to {d['high']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="path to the NHANES CSV file")
    args = parser.parse_args()

    cohort = load_cohort(args.data)
    models = fit_bp_models(cohort)

    print(f"Model trained on {len(cohort):,} unmedicated adults.\n")
    print(f"SBP R-squared: {models[0].rsquared:.3f} | "
          f"DBP R-squared: {models[1].rsquared:.3f}\n")

    # printing a few example profiles
    _print_profile(models, 44, "Female", 28)
    _print_profile(models, 52, "Female", 33)
    _print_profile(models, 25, "Male", 22)
    _print_profile(models, 70, "Male", 30)


if __name__ == "__main__":
    main()
