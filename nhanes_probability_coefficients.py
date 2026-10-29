"""NHANES probability coefficients for high blood pressure.

This module quantifies how much age, gender and BMI change the chance of high
blood pressure, using the unmedicated adult cohort of the pooled NHANES cycles.
It produces two tables:

  1. Likelihood factors: the unadjusted prevalence ratio for each demographic
     group versus a reference group (one factor at a time).
  2. Adjusted odds ratios: the effect of each factor after holding the others
     steady, with 95% confidence intervals, from a single logistic regression.

These coefficients act as the reference prior that later small-sample models
are checked against. Run as a script to compute and save both tables, or import
the functions (for example, for the Streamlit app).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import BinaryResultsWrapper

from bp_zone_model import load_cohort, DEFAULT_DATA

# high-BP threshold (hypertensive range) and category edges
HIGH_SBP, HIGH_DBP = 140, 90
BMI_BINS = [0, 18.5, 25, 30, 200]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]
AGE_BINS = [18, 30, 45, 60, 200]
AGE_LABELS = ["18-29", "30-44", "45-59", "60-85"]

REF_BMI, REF_AGE, REF_GENDER = "Normal", "18-29", "Female"


def add_derived(cohort: pd.DataFrame) -> pd.DataFrame:
    """Adding the high-BP outcome and the BMI and age categories."""
    cohort = cohort.copy()
    cohort["high_bp"] = ((cohort.sbp_mean_mmhg >= HIGH_SBP) |
                         (cohort.dbp_mean_mmhg >= HIGH_DBP)).astype(int)
    cohort["bmi_cat"] = pd.cut(cohort.bmi_kg_m2, BMI_BINS,
                               labels=BMI_LABELS, right=False).astype(str)
    cohort["age_band"] = pd.cut(cohort.age_years, AGE_BINS,
                                labels=AGE_LABELS, right=False).astype(str)
    return cohort


def likelihood_factors(cohort: pd.DataFrame) -> pd.DataFrame:
    """Computing the unadjusted prevalence ratio versus a reference level."""
    rows = []
    factors = [("bmi_cat", REF_BMI), ("age_band", REF_AGE),
               ("gender_group", REF_GENDER)]
    for col, ref in factors:
        grp = cohort.groupby(col, observed=True)["high_bp"].agg(["mean", "count"])
        ref_prevalence = grp.loc[ref, "mean"]
        for level, row in grp.iterrows():
            rows.append({
                "factor": col,
                "level": level,
                "reference": ref,
                "n": int(row["count"]),
                "prevalence_%": round(row["mean"] * 100, 1),
                "likelihood_vs_ref": round(row["mean"] / ref_prevalence, 2),
            })
    return pd.DataFrame(rows)


def fit_logit_model(cohort: pd.DataFrame) -> BinaryResultsWrapper:
    """Fitting one logistic model with age, gender and BMI together."""
    formula = (
        f"high_bp ~ C(bmi_cat, Treatment('{REF_BMI}')) "
        f"+ C(age_band, Treatment('{REF_AGE}')) "
        f"+ C(gender_group, Treatment('{REF_GENDER}'))"
    )
    return smf.logit(formula, data=cohort).fit(disp=0)


def adjusted_odds_ratios(model: BinaryResultsWrapper) -> pd.DataFrame:
    """Turning the logistic coefficients into odds ratios with 95% ranges."""
    conf = model.conf_int()
    table = pd.DataFrame({
        "log_odds": model.params,
        "odds_ratio": np.exp(model.params),
        "or_low": np.exp(conf[0]),
        "or_high": np.exp(conf[1]),
    }).drop(index="Intercept").round(3)
    return table.reset_index().rename(columns={"index": "term"})


def _categorize(age: float, bmi: float) -> tuple[str, str]:
    """Mapping a single person's age and BMI to their band and category."""
    age_band = pd.cut([age], AGE_BINS, labels=AGE_LABELS, right=False)[0]
    bmi_cat = pd.cut([bmi], BMI_BINS, labels=BMI_LABELS, right=False)[0]
    return str(age_band), str(bmi_cat)


def person_report(model: BinaryResultsWrapper, age: float, gender: str,
                  bmi: float) -> dict:
    """Reporting one person's high-BP probability and combined odds ratio.

    The probability comes with an 80% range; the combined odds ratio (versus a
    young, normal-weight woman) comes with a 95% range. Both ranges use the
    model's own covariance, so they are exact for this profile.
    """
    age_band, bmi_cat = _categorize(age, bmi)
    person = pd.DataFrame([{
        "bmi_cat": bmi_cat, "age_band": age_band, "gender_group": gender,
    }])

    # probability of high BP with an 80% range (confidence in the estimate)
    frame = model.get_prediction(person).summary_frame(alpha=0.20)
    prob = float(frame["predicted"].iloc[0])
    prob_low = float(frame["ci_lower"].iloc[0])
    prob_high = float(frame["ci_upper"].iloc[0])

    # building the design row that matches the fitted model
    design_info = model.model.data.orig_exog.design_info
    x = np.asarray(patsy.build_design_matrices([design_info], person)[0])[0]
    params = model.params.values
    cov = model.cov_params().values

    # combined odds ratio versus baseline (zeroing the intercept term)
    intercept_idx = list(model.params.index).index("Intercept")
    contrast = x.copy()
    contrast[intercept_idx] = 0.0
    log_or = float(contrast @ params)
    se_or = float(np.sqrt(contrast @ cov @ contrast))
    odds_ratio = float(np.exp(log_or))
    or_low = float(np.exp(log_or - 1.96 * se_or))
    or_high = float(np.exp(log_or + 1.96 * se_or))

    return {
        "age_band": age_band,
        "bmi_cat": bmi_cat,
        "gender": gender,
        "prob_high_bp": prob,
        "prob_low": prob_low,
        "prob_high": prob_high,
        "odds_ratio_vs_baseline": odds_ratio,
        "or_low": or_low,
        "or_high": or_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="path to the NHANES CSV file")
    parser.add_argument("--output-dir", default="outputs",
                        help="directory for the output tables")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = add_derived(load_cohort(args.data))
    n = len(cohort)
    overall = cohort.high_bp.mean()

    print(f"Unmedicated adult cohort: n = {n:,}")
    print(f"Overall high-BP (>={HIGH_SBP}/{HIGH_DBP}) prevalence: "
          f"{overall * 100:.1f}%\n")

    factors = likelihood_factors(cohort)
    print("Likelihood factors (unadjusted prevalence ratios):")
    print(factors.to_string(index=False), "\n")

    model = fit_logit_model(cohort)
    odds = adjusted_odds_ratios(model)
    print("Adjusted odds ratios (age + gender + BMI together):")
    print(odds.to_string(index=False), "\n")

    # saving tables as CSV (viewable on GitHub) and as one Excel workbook
    factors.to_csv(out_dir / "likelihood_factors.csv", index=False)
    odds.to_csv(out_dir / "adjusted_odds_ratios.csv", index=False)
    with pd.ExcelWriter(out_dir / "probability_coefficients.xlsx") as xl:
        factors.to_excel(xl, sheet_name="likelihood_factors", index=False)
        odds.to_excel(xl, sheet_name="adjusted_odds_ratios", index=False)
        pd.DataFrame({"metric": ["cohort_n", "overall_high_bp_%"],
                      "value": [n, round(overall * 100, 1)]}).to_excel(
            xl, sheet_name="cohort", index=False)

    print(f"Saved tables to {out_dir}/")


if __name__ == "__main__":
    main()
