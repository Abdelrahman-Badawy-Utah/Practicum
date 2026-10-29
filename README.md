# Demographic Blood-Pressure Reference Model

A demographic reference model for smartphone-based (rPPG) blood-pressure
estimation. When a device measures a person for the first time and has no
personal cuff reading to anchor on, it must fall back on population guidance,
because age, gender and BMI all shift blood pressure. This repository builds
that guidance from the pooled NHANES cycles and exposes it as:

1. **Probability coefficients** that quantify how much each demographic factor
   changes the chance of high blood pressure.
2. **A blood-pressure zone model** that predicts a person's most likely
   systolic and diastolic pressure, with an honest individual range.
3. **An interactive tool** (Streamlit) to try any profile.

All models are trained on the **unmedicated** adult cohort, because the device
does not ask which medication a person takes, so the reference should reflect
people who are not on blood-pressure medication.

---

## Repository structure

```
.
├── bp_zone_model.py                  # BP zone model (centre + individual range)
├── nhanes_probability_coefficients.py# likelihood factors + adjusted odds ratios
├── app.py                            # Streamlit tool to try any profile
├── requirements.txt
├── data/
│   ├── nhanes_cohort.csv             # trimmed NHANES extract (included)
│   └── README.md                     # data description
└── outputs/                          # generated tables (CSV + Excel)
```

## Setup

```bash
pip install -r requirements.txt
```

A trimmed NHANES extract is included at `data/nhanes_cohort.csv`, so everything
runs after cloning with no extra setup. To use a full NHANES export instead,
pass its path with `--data` (see `data/README.md`).

## How to run

Compute and save the probability coefficient tables:

```bash
python nhanes_probability_coefficients.py
```

Print the blood-pressure zone model and example profiles:

```bash
python bp_zone_model.py
```

Launch the interactive tool:

```bash
streamlit run app.py
```

---

## The cohort

After keeping adults aged 18 to 85 with valid blood pressure, BMI and gender,
and excluding anyone on blood-pressure medication, the cohort is **46,041
people**. Among them, **13.0%** have high blood pressure (140/90 or above) with
no medication.

---

## Table 1: Likelihood factors (unadjusted)

This table looks at **one factor at a time**. For each group it gives the share
with high blood pressure (`prevalence_%`) and how many times more likely that is
than the reference group (`likelihood_vs_ref`). The reference group always
equals 1.00.

| Factor | Level       | n      | Prevalence % | Likelihood vs reference |
|--------|-------------|--------|--------------|-------------------------|
| BMI    | Underweight | 1,058  | 9.4          | 0.92                    |
| BMI    | Normal      | 15,110 | 10.2         | 1.00 (reference)        |
| BMI    | Overweight  | 15,054 | 13.5         | 1.33                    |
| BMI    | Obese       | 14,819 | 15.5         | 1.52                    |
| Age    | 18-29       | 13,341 | 2.5          | 1.00 (reference)        |
| Age    | 30-44       | 13,335 | 7.8          | 3.16                    |
| Age    | 45-59       | 9,912  | 16.6         | 6.71                    |
| Age    | 60-85       | 9,453  | 31.1         | 12.52                   |
| Gender | Female      | 23,455 | 11.3         | 1.00 (reference)        |
| Gender | Male        | 22,586 | 14.7         | 1.30                    |

**How to read it.** An overweight adult is about 1.33 times as likely to have
high blood pressure as a normal-weight adult. The 60 to 85 group is about 12.5
times as likely as the 18 to 29 group. Age is clearly the strongest factor.

**Caution.** These numbers mix causes together. Part of the overweight effect is
really age, because older people tend to weigh more. Table 2 separates them.

## Table 2: Adjusted odds ratios

This table puts age, gender and BMI into **one logistic regression**, so each
odds ratio is the effect of that factor **after holding the others steady**.
The low and high columns are the 95% confidence interval.

| Term            | Odds ratio | 95% low | 95% high |
|-----------------|------------|---------|----------|
| BMI Overweight  | 1.13       | 1.05    | 1.21     |
| BMI Obese       | 1.57       | 1.46    | 1.69     |
| BMI Underweight | 1.12       | 0.89    | 1.41     |
| Age 30-44       | 3.21       | 2.83    | 3.64     |
| Age 45-59       | 7.55       | 6.69    | 8.53     |
| Age 60-85       | 17.45      | 15.51   | 19.64    |
| Gender Male     | 1.32       | 1.25    | 1.40     |

**How to read it.** After accounting for age and gender, being obese raises the
odds of high blood pressure by about 57%. Notice the overweight effect drops
from 1.33 (Table 1) to 1.13 here, because the age hidden inside the raw BMI
effect has been removed.

**Confidence ranges matter.** The underweight range (0.89 to 1.41) crosses 1.0,
which means its effect is not statistically reliable, most likely because the
underweight group is small (n = 1,058). Every other factor is reliable.

**Takeaway ranking.** Age dominates (up to 17.5x), BMI is real but modest (obese
1.57x, overweight 1.13x once age is removed), gender is modest (1.32x).

---

## Table 3: Blood-pressure zone model

Two linear models predict the actual pressure value, one for systolic (SBP) and
one for diastolic (DBP). Blood pressure is explained by age, age squared
(pressure rises faster at older ages), BMI and gender. For any person the model
returns a **centre** (most likely value) and a **zone** (an 80% individual
prediction range, where a real person is likely to fall).

Example profiles:

| Profile                 | SBP centre | SBP zone  | DBP centre | DBP zone |
|-------------------------|------------|-----------|------------|----------|
| Female, 44, BMI 28      | 116        | 98 to 135 | 73         | 59 to 87 |
| Female, 52, BMI 33      | 121        | 103 to 140| 75         | 60 to 89 |
| Male, 25, BMI 22        | 114        | 96 to 133 | 67         | 53 to 82 |
| Male, 70, BMI 30        | 136        | 117 to 154| 72         | 58 to 87 |

**Model fit.** The systolic model explains about 24% of blood-pressure variation
(R-squared 0.244); the diastolic model explains about 15% (R-squared 0.152).
The typical person-to-person scatter around the systolic line is about 14.6
mmHg, which is what makes the zone roughly 35 to 40 points wide.

**Why the zone is wide and honest.** Age, gender and BMI together explain only
about a quarter of why blood pressure differs between people. The rest is
individual biology that demographics cannot see. So the model gives a sensible
zone for a first-time reading, not an exact number, and a personal cuff reading
remains more accurate when one is available.

**A note on physiology.** The models reproduce known patterns on their own:
systolic pressure keeps rising with age, while diastolic pressure peaks around
age 50 and then declines. This widening gap (pulse pressure) is expected as
arteries stiffen, and it is a good sign the models capture something real.

---

## Interactive tool

`app.py` is a Streamlit tool. A reviewer enters a profile (age, gender, BMI) and
the app shows:

- predicted systolic and diastolic pressure, each with its individual range,
- the probability of high blood pressure for that profile, with a range,
- the combined odds ratio versus a young, normal-weight reference, with a range,
- the contribution of each factor, from the adjusted odds-ratio table.

Run it with `streamlit run app.py`.

---

## Method summary

| Step | What it does                                                        |
|------|---------------------------------------------------------------------|
| 1    | Keep the unmedicated adult cohort (not on blood-pressure medication) |
| 2    | Compute the NHANES probability coefficients (the reference prior)    |
| 3    | Predict the blood-pressure zone for any profile                      |

The next phase extends this reference with a mixed-effects model (to stabilise
small demographic groups), a validation against a separate measured dataset, and
a skin-tone correction for the rPPG signal.

---

## Data source

NHANES (National Health and Nutrition Examination Survey), pooled cycles,
published by the U.S. Centers for Disease Control and Prevention (CDC). Public
data, free to use.
