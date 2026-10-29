# Data

## Included file

`nhanes_cohort.csv` is a trimmed extract of the pooled NHANES cycles, kept small
so the scripts and the app run straight after cloning. It holds adults aged 18
to 85 with valid values, with only the columns the models need.

| Column             | Meaning                                         |
|--------------------|-------------------------------------------------|
| `age_years`        | Age in years                                    |
| `gender_group`     | `Male` or `Female`                              |
| `bmi_kg_m2`        | Body mass index (kg/m²)                         |
| `sbp_mean_mmhg`    | Mean systolic blood pressure (mmHg)             |
| `dbp_mean_mmhg`    | Mean diastolic blood pressure (mmHg)            |
| `on_bp_medication` | 1 if on blood-pressure medication, else 0       |

The scripts apply the final filtering themselves (excluding anyone on
blood-pressure medication), which leaves the unmedicated reference cohort of
about 46,000 people.

## Using the full raw file instead

To run against a full NHANES export instead, place it here and pass its path:

```bash
python bp_zone_model.py --data data/your_file.csv
```

Large raw files named `all_cycles*.csv` are not tracked (see `.gitignore`).

## Source

NHANES (National Health and Nutrition Examination Survey), published by the U.S.
Centers for Disease Control and Prevention (CDC). Public data, free to use.
