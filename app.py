"""Streamlit tool for the demographic blood-pressure model.

A reviewer enters a profile (age, gender, BMI) and the app returns the
predicted blood pressure with its individual range, the probability of high
blood pressure, and the combined odds ratio versus a young, normal-weight
reference. Run with:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from bp_zone_model import load_cohort, fit_bp_models, bp_zone, DEFAULT_DATA
from nhanes_probability_coefficients import (
    add_derived, fit_logit_model, adjusted_odds_ratios, person_report,
)

st.set_page_config(page_title="Demographic BP Model", page_icon="🩺",
                   layout="centered")


@st.cache_resource(show_spinner="Training models on the NHANES cohort...")
def build_models(data_path: str):
    """Loading the cohort and fitting every model once, then caching them."""
    cohort = load_cohort(data_path)
    sbp_model, dbp_model = fit_bp_models(cohort)
    logit = fit_logit_model(add_derived(cohort))
    odds = adjusted_odds_ratios(logit)
    return cohort, sbp_model, dbp_model, logit, odds


st.title("Demographic Blood-Pressure Model")
st.caption("A population reference for first-time (uncalibrated) readings, "
           "built on unmedicated adults from NHANES.")

# checking the data file is present before anything else
data_path = DEFAULT_DATA
if not Path(data_path).exists():
    st.error(
        f"Data file not found at `{data_path}`.\n\n"
        "Place the NHANES CSV there (see `data/README.md`) and reload."
    )
    st.stop()

cohort, sbp_model, dbp_model, logit, odds = build_models(data_path)

# input controls
with st.sidebar:
    st.header("Enter a profile")
    age = st.slider("Age (years)", 18, 85, 44)
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 28.0, step=0.5)
    coverage = st.select_slider("Range coverage", [0.80, 0.95], value=0.80,
                                format_func=lambda c: f"{int(c*100)}%")
    st.caption(f"Model trained on {len(cohort):,} unmedicated adults.")

# predictions
zone = bp_zone(sbp_model, dbp_model, age, gender, bmi, coverage=coverage)
report = person_report(logit, age, gender, bmi)

st.subheader(f"{gender}, age {age}, BMI {bmi:g}")

# predicted blood pressure with range
c1, c2 = st.columns(2)
with c1:
    s = zone["SBP"]
    st.metric("Systolic (SBP), most likely", f"{s['center']} mmHg")
    st.caption(f"{int(coverage*100)}% range: {s['low']} to {s['high']} mmHg")
with c2:
    d = zone["DBP"]
    st.metric("Diastolic (DBP), most likely", f"{d['center']} mmHg")
    st.caption(f"{int(coverage*100)}% range: {d['low']} to {d['high']} mmHg")

st.divider()

# probability of high BP and combined odds ratio
c3, c4 = st.columns(2)
with c3:
    st.metric("Chance of high BP (≥140/90)",
              f"{report['prob_high_bp']*100:.1f}%")
    st.caption(f"80% range: {report['prob_low']*100:.1f}% "
               f"to {report['prob_high']*100:.1f}%")
with c4:
    st.metric("Odds ratio vs reference",
              f"{report['odds_ratio_vs_baseline']:.2f}×")
    st.caption(f"95% range: {report['or_low']:.2f} to {report['or_high']:.2f} "
               "(vs young, normal-weight woman)")

st.divider()

# how each factor contributes, from the adjusted odds-ratio table
st.markdown("**How each factor contributes (adjusted odds ratios)**")
terms = {
    f"C(bmi_cat, Treatment('Normal'))[T.{report['bmi_cat']}]": f"BMI: {report['bmi_cat']}",
    f"C(age_band, Treatment('18-29'))[T.{report['age_band']}]": f"Age: {report['age_band']}",
    "C(gender_group, Treatment('Female'))[T.Male]": "Gender: Male",
}
rows = []
for term, label in terms.items():
    match = odds[odds.term == term]
    if not match.empty:
        r = match.iloc[0]
        rows.append({"Factor": label, "Odds ratio": r.odds_ratio,
                     "Low": r.or_low, "High": r.or_high})
if rows:
    st.dataframe(pd.DataFrame(rows).set_index("Factor"))
else:
    st.caption("This profile matches the reference group on every factor, "
               "so no adjustment applies.")
