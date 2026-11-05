"""Step 2: forecast India's monthly cement & steel production 1 and 3 months ahead.

Every model is tested "walk-forward": for each month from April 2023 onwards we train only on the
data that existed at that time, forecast ahead, and compare with what really happened.
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
market = pd.read_csv("data/processed/market_monthly.csv", parse_dates=["month"], index_col="month")
FIRST_VALID = pd.Timestamp("2021-04-01")   # validation: Apr 2021 - Mar 2023 (choose the model)
FIRST_TEST = pd.Timestamp("2023-04-01")    # test: Apr 2023 onwards (report the score)
HORIZONS = (1, 3)

# ---------- calendar features (all known in advance) ----------
DIWALI = {2011: "10-26", 2012: "11-13", 2013: "11-03", 2014: "10-23", 2015: "11-11", 2016: "10-30",
          2017: "10-19", 2018: "11-07", 2019: "10-27", 2020: "11-14", 2021: "11-04", 2022: "10-24",
          2023: "11-12", 2024: "10-31", 2025: "10-20", 2026: "11-08"}
months = pd.date_range(market.index.min(), market.index.max() + pd.DateOffset(months=max(HORIZONS)), freq="MS")
cal = pd.DataFrame(index=months)
cal["diwali"] = cal.index.isin([pd.Timestamp(f"{y}-{d}").to_period("M").to_timestamp()
                                for y, d in DIWALI.items()]).astype(int)
cal["lockdown"] = cal.index.isin(pd.to_datetime(["2020-04-01", "2020-05-01"])).astype(int)
cal["second_wave"] = cal.index.isin(pd.to_datetime(["2021-05-01"])).astype(int)
cal["base_effect"] = cal["lockdown"].shift(12, fill_value=0)      # Apr-May 2021 vs collapsed 2020
cal["working_days"] = [int((pd.date_range(m, m + pd.offsets.MonthEnd(0)).dayofweek != 6).sum())
                       for m in cal.index]
EXOG = ["lockdown", "second_wave", "base_effect", "diwali"]
SHOCKS = pd.to_datetime(["2020-04-01", "2020-05-01", "2021-05-01"])


def tabular_rows(y, h):
    """One row per origin month t: features known at t, target = log value at t+h."""
    f = pd.DataFrame(index=y.index)
    f["target"] = y.shift(-h)
    f["now"] = y
    f["lag1"] = y.shift(1)
    f["lag2"] = y.shift(2)
    f["same_month_last_year"] = y.shift(12 - h)
    f["growth_12m"] = y - y.shift(12)
    tgt = cal.reindex(y.index + pd.DateOffset(months=h))
    tgt.index = y.index
    f = f.join(tgt)
    f = f.join(pd.get_dummies(tgt.index.month, prefix="m", dtype=float).set_index(f.index))
    return f


def ets_forecast(y_train, h):
    """Holt-Winters on logs; the shock months are replaced by the same month a year earlier."""
    y_clean = y_train.copy()
    for m in SHOCKS:
        if m in y_clean.index and m - pd.DateOffset(months=12) in y_clean.index:
            y_clean[m] = y_clean[m - pd.DateOffset(months=12)]
    fit = ExponentialSmoothing(y_clean, trend="add", seasonal="add", seasonal_periods=12).fit()
    return float(fit.forecast(h).iloc[-1])


rows = []
for series in ["cement", "steel"]:
    y = np.log(market[series]).asfreq("MS")
    tables = {h: tabular_rows(y, h) for h in HORIZONS}
    for t in y.index[(y.index >= FIRST_VALID - pd.DateOffset(months=1))]:
        sarima = SARIMAX(y.loc[:t], exog=cal.loc[:t, EXOG], order=(1, 1, 1),
                         seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        sar_path = sarima.forecast(max(HORIZONS), exog=cal.loc[t + pd.DateOffset(months=1):
                                                             t + pd.DateOffset(months=max(HORIZONS)), EXOG])
        for h in HORIZONS:
            target_month = t + pd.DateOffset(months=h)
            if target_month not in y.index or target_month < FIRST_VALID:
                continue
            tab = tables[h]
            train = tab.loc[:t - pd.DateOffset(months=h)].dropna()     # targets already known at t
            cols = [c for c in tab.columns if c != "target"]
            x_now = tab.loc[[t], cols]
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(train[cols], train.target)
            lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                                 verbose=-1).fit(train[cols], train.target)
            ly = y[target_month - pd.DateOffset(months=12)]                 # same month last year
            growth = y.loc[:t].iloc[-12:].mean() - y.loc[:t].iloc[-24:-12].mean()
            preds = {
                "Same month last year": ly,
                "Last year + recent growth": ly + growth,
                "Holt-Winters (ETS)": ets_forecast(y.loc[:t], h),
                "SARIMA": float(sar_path.iloc[h - 1]),
                "Ridge (lags + calendar)": float(ridge.predict(x_now)[0]),
                "LightGBM": float(lgbm.predict(x_now)[0]),
            }
            preds["Average of ETS, SARIMA and growth rule"] = np.mean(
                [preds["Holt-Winters (ETS)"], preds["SARIMA"], preds["Last year + recent growth"]])
            for model, p in preds.items():
                rows.append((series, h, target_month, model, float(np.exp(y[target_month])), float(np.exp(p))))

res = pd.DataFrame(rows, columns=["series", "horizon", "month", "model", "actual", "forecast"])
res["ape"] = (res.forecast - res.actual).abs() / res.actual * 100
res["err_pct"] = (res.forecast - res.actual) / res.actual * 100
res["period"] = np.where(res.month < FIRST_TEST, "validation", "test")
res.to_csv("reports/market_forecasts.csv", index=False)

valid = (res[res.period == "validation"].groupby(["series", "horizon", "model"]).ape.mean()
         .rename("valid_MAPE").round(2).reset_index())
summary = (res[res.period == "test"].groupby(["series", "horizon", "model"])
           .agg(MAPE=("ape", "mean"), bias=("err_pct", "mean"), months=("ape", "size"))
           .round(2).reset_index().merge(valid, on=["series", "horizon", "model"]))
summary.to_csv("reports/market_model_comparison.csv", index=False)
for (series, h), g in summary.groupby(["series", "horizon"]):
    print(f"\n=== {series.upper()} | {h} month(s) ahead | test = {g.months.iloc[0]} unseen months ===")
    print(g.sort_values("valid_MAPE")[["model", "valid_MAPE", "MAPE", "bias"]]
          .rename(columns={"MAPE": "test_MAPE", "bias": "test_bias"}).to_string(index=False))

# ---------- headline numbers + chart ----------
best = {}
for (series, h), g in summary.groupby(["series", "horizon"]):
    b = g.sort_values("valid_MAPE").iloc[0]          # chosen on validation, NOT on test
    base = g.set_index("model").loc["Last year + recent growth"]
    best[f"{series}_h{h}"] = {"chosen_model": b.model, "test_mape": b.MAPE,
                              "baseline": "Last year + recent growth", "baseline_mape": base.MAPE,
                              "naive_same_month_mape": g.set_index("model").loc["Same month last year"].MAPE,
                              "error_reduction_vs_baseline_pct": round((1 - b.MAPE / base.MAPE) * 100, 1)}
json.dump(best, open("reports/market_headline.json", "w"), indent=2)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, series in zip(axes, ["cement", "steel"]):
    hist = market[series].loc["2019-01-01":]
    f_all = res[(res.series == series) & (res.horizon == 1) & (res.period == "test")]
    ax.plot(hist.index, hist.values, color="black", lw=1.5, label="Actual (official)")
    b = best[f"{series}_h1"]["chosen_model"]
    f = f_all[f_all.model == b]
    ax.plot(f.month, f.forecast, color="tab:blue", lw=1.5, ls="--", label=f"1-month-ahead forecast: {b}")
    ax.axvline(FIRST_TEST, color="grey", lw=0.8)
    ax.set_title(f"India {series} production index (walk-forward test from Apr 2023)")
    ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("reports/figures/market_forecast.png", dpi=150)
print("\nHeadline:", json.dumps(best, indent=1))
