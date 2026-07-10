# US Airline Flight Delay & Diversion Analysis (2004–2008)

**Course:** ST2195 – Programming for Data Science
**Language:** Python 3 · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn`
**Dataset:** 2009 ASA Statistical Computing and Graphics Data Expo (Harvard Dataverse)
**Reproducibility:** Random seed = 42 throughout

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Loading & Cleaning](#2-data-loading--cleaning)
3. [Question (a) — Best Times & Days to Fly](#3-question-a--best-times--days-to-minimise-delays)
4. [Question (b) — Do Older Planes Suffer More Delays?](#4-question-b--do-older-planes-suffer-more-delays)
5. [Question (c) — Logistic Regression for Flight Diversion](#5-question-c--logistic-regression-for-pdiverted)
6. [Summary of Findings](#6-summary-of-findings)
7. [References](#7-references)

---

## 1. Overview

This report analyses roughly **35 million US domestic flight records spanning 2004–2008**, drawn from the 2009 ASA Data Expo dataset. The dataset — along with supporting reference tables — was obtained from the Harvard Dataverse:

> **Source:** [https://doi.org/10.7910/DVN/HG7NV7](https://doi.org/10.7910/DVN/HG7NV7)

Three questions are addressed, each treated as a separate year-by-year analysis so that trends over time can be identified:

| Question | What it answers |
|---|---|
| **2(a)** | What are the best times and days of the week to minimise delays? |
| **2(b)** | Do older aircraft suffer more delays? |
| **2(c)** | Can we model the probability that a flight is diverted, and which factors drive it? |

---

## 2. Data Loading & Cleaning

The full uncompressed dataset is roughly **12 GB**, which caused memory crashes when loaded naively. To work around this, each year's compressed CSV was read with `pandas.read_csv()` using the `usecols` parameter, restricting the load to only the **16 columns** required for this analysis. Numeric columns were then downcast to smaller dtypes (`int8`, `int16`) to further reduce memory overhead.

Two supporting reference files were also merged in:

- `plane-data.csv` — aircraft manufacture year, keyed on tail number
- `airports.csv` — airport latitude/longitude, keyed on IATA code

### Cleaning steps

| # | Operation | Reason |
|---|---|---|
| 1 | Removed rows where `Cancelled = 1` | Cancelled flights carry no meaningful delay information |
| 2 | Derived `Hour = DepTime // 100` | `DepTime` is stored as `HHMM` (e.g. 1435 = 14:35) |
| 3 | Downcast numeric columns to `int8`/`int16` | Reduces memory footprint |
| 4 | Converted `ArrDelay` / `DepDelay` to numeric | Some rows contained non-numeric strings |
| 5 | Dropped rows with missing delay values | Mean delay cannot be computed from missing values |
| 6 | Merged `plane-data.csv` on `TailNum`; dropped rows with missing `MfgYear` | Aircraft age requires a valid manufacture year |
| 7 | Removed rows with `PlaneAge < 0` or `> 40` | Extreme/invalid ages would skew results |
| 8 | Dropped rows with missing coordinates | Latitude/longitude are required inputs for the Q2(c) regression |

---

## 3. Question (a) — Best Times & Days to Minimise Delays

**Method:** Mean departure delay (`DepDelay`) was computed by grouping flights on `(Year, Hour)` and, separately, on `(Year, DayOfWeek)`. Negative `DepDelay` values (early departures) were kept, since they are valid observations — a negative mean simply indicates better-than-scheduled performance. Each year was analysed independently so that year-on-year patterns could be compared, and a combined day × hour heatmap was produced to capture the joint effect of both variables.

### Results by departure hour

![Mean departure delay by hour, 2004–2008](assets/fig1_delay_by_hour.png)

Across all five years, **early morning departures between 04:00 and 06:00** consistently produced the lowest mean delays — in some years (e.g. 2004) these flights departed early, on average. This makes operational sense: the first flights of the day start with a clean slate, with no delays inherited from previous rotations. Delays then build progressively through the day as knock-on effects accumulate, typically peaking between **17:00 and 20:00**.

### Results by day of week

![Mean departure delay by day of week, 2004–2008](assets/fig2_delay_by_day.png)

**Saturdays and mid-week days (Tuesday/Wednesday)** tend to produce the lowest average delay:

- Saturday was the best day to fly in **2004, 2005, and 2007**
- Tuesday was best in **2006**
- Wednesday was best in **2008**

These days typically carry lower flight volumes. Monday, Thursday, and Friday see a pickup from business travel, while **Friday and Sunday consistently show the highest delays**, driven by peak leisure-travel volume (outbound on Friday, return on Sunday).

### Combined day × hour view

![Combined heatmap of delay by day and hour](assets/fig3_day_hour_heatmap.png)

The heatmap confirms that **early mornings, on any day**, are the safest window — with the *best* combinations being early mornings on Saturdays or mid-week days. Blue cells indicate low/negative delay; red cells indicate high delay.

> **Conclusion (2a):** To minimise departure delays, fly early in the morning (04:00–06:00) on a Saturday, Tuesday, or Wednesday. This pattern is stable across all five years, reflecting structural dynamics of the US air traffic system — early flights avoid cascading network delays, and specific days simply see lower demand.

---

## 4. Question (b) — Do Older Planes Suffer More Delays?

**Method:** Flight records were merged with `plane-data.csv` on `TailNum` (the aircraft's unique registration number), and aircraft age at the time of each flight was computed as:

```
PlaneAge = Flight Year − Aircraft Manufacture Year
```

Rows with a missing manufacture year, or an implausible age (negative, or over 40 years), were removed. Mean delay was computed for every `(Year, PlaneAge)` combination, and a 3-point rolling average was applied when plotting to smooth noise arising from small sample sizes at extreme ages. To quantify the relationship, the **Pearson correlation coefficient** between `PlaneAge` and `ArrDelay` was computed per year on the individual flight records.

![Mean departure delay vs aircraft age, smoothed, 2004–2008](assets/fig4_delay_vs_aircraft_age.png)

### Pearson correlation results

| Year | r value | Interpretation |
|---|---|---|
| 2004 | 0.0024 | Negligible positive — effectively no relationship |
| 2005 | 0.0046 | Negligible positive — effectively no relationship |
| 2006 | −0.0088 | Negligible negative — effectively no relationship |
| 2007 | 0.0004 | Negligible positive — effectively no relationship |
| 2008 | 0.0074 | Negligible positive — effectively no relationship |

> **Conclusion (2b):** Correlation values sit consistently close to zero across every year (2006 even shows a negligible negative correlation). There is effectively **no relationship between an aircraft's age and its departure delay** — a finding backed up visually, since the plot shows no consistent upward trend in delay as aircraft get older.

---

## 5. Question (c) — Logistic Regression for P(Diverted)

**Method:** A separate binary logistic regression (`solver='lbfgs'`, `max_iter=1000`) was fitted per year to estimate the probability that a flight is diverted (`Diverted = 1`). Logistic regression is the natural choice here since the outcome is binary. Each fitted coefficient indicates direction of effect: **positive → increases diversion probability, negative → decreases it.**

**Modelling decisions:**

- **Features:** departure date attributes, scheduled departure/arrival time, origin & destination coordinates, flight distance, and one-hot encoded carrier
- **Standardisation:** all numeric features were scaled to zero mean / unit variance (`StandardScaler`) so coefficients are directly comparable
- **Class imbalance:** diversions are rare events, so `class_weight='balanced'` was used to automatically up-weight the minority class during training
- **Carrier encoding:** one-hot encoded independently within each year's loop, with `drop_first=True` to avoid multicollinearity

### Numeric feature coefficients across years

![Logistic regression coefficients for numeric features, 2004–2008](assets/fig5_logistic_regression_numeric_coefficients.png)

### Carrier coefficients across years

![Carrier coefficient heatmap, 2004–2008](assets/fig6_carrier_coefficients_heatmap.png)

> **Conclusion (2c):** Standardising the features makes coefficient magnitudes directly comparable year over year. The numeric-feature trend lines show broad **stability across the five-year period** — the factors driving diversions did not shift substantially between 2004 and 2008. Visualising the many carrier categories as a heatmap (rather than a crowded line chart) keeps the categorical comparison readable.

---

## 6. Summary of Findings

- **Timing:** Flying early morning (04:00–06:00), especially on a Saturday, Tuesday, or Wednesday, minimises expected departure delay — consistently across all five years.
- **Aircraft age:** Has effectively **no measurable effect** on departure delay (|r| < 0.01 in every year studied).
- **Diversions:** Are predictable to some extent from scheduling and geographic features, with **distance** and **destination longitude** showing the strongest, most consistent positive association with diversion probability, and the pattern of drivers stayed broadly stable from 2004–2008.

---

## 7. References

1. American Statistical Association (2009). *ASA Statistical Computing and Graphics Data Expo 2009.* Harvard Dataverse. https://doi.org/10.7910/DVN/HG7NV7
2. McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proceedings of the 9th Python in Science Conference, pp. 51–56.
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
4. Hunter, J.D. (2007). *Matplotlib: A 2D Graphics Environment.* Computing in Science and Engineering, 9(3), 90–95.
5. Waskom, M. (2021). *Seaborn: Statistical Data Visualization.* Journal of Open Source Software, 6(60), 3021.
6. Harris, C.R. et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362.
