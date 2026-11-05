"""Step 5: refit the chosen model on all data and forecast the next 3 months (for the dashboard).

The uncertainty band comes from the model's real errors in the walk-forward test (10th-90th percentile).
"""
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
market = pd.read_csv("data/processed/market_monthly.csv", parse_dates=["month"], index_col="month")
past = pd.read_csv("reports/market_forecasts.csv", parse_dates=["month"])
headline = json.load(open("reports/market_headline.json"))
SHOCKS = pd.to_datetime(["2020-04-01", "2020-05-01", "2021-05-01"])

rows = []
for series in ["cement", "steel"]:
    y = np.log(market[series]).asfreq("MS")
    for m in SHOCKS:                                  # same COVID treatment as in the backtest
        y[m] = y[m - pd.DateOffset(months=12)]
    fit = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).fit()
    point = np.exp(fit.forecast(3))
    for h, (month, value) in enumerate(point.items(), start=1):
        test_h = 1 if h == 1 else 3                    # month 2 uses the (wider) 3-month errors
        model = headline[f"{series}_h{test_h}"]["chosen_model"]
        errs = past[(past.series == series) & (past.period == "test") & (past.model == model) &
                    (past.horizon == test_h)].err_pct / 100
        lo, hi = value / (1 + errs.quantile(0.9)), value / (1 + errs.quantile(0.1))
        last_year = market[series].get(month - pd.DateOffset(months=12), np.nan)
        rows.append((series, month, h, round(value, 1), round(lo, 1), round(hi, 1),
                     round((value / last_year - 1) * 100, 1)))
out = pd.DataFrame(rows, columns=["series", "month", "months_ahead", "forecast", "low_80", "high_80",
                                  "growth_vs_last_year_pct"])
out.to_csv("reports/market_outlook.csv", index=False)
print(out.to_string(index=False))
