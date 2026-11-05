"""Step 4: operations analytics on 9,300 real ready-mix concrete orders.

Questions a supplier's planner asks:
  1. How many orders arrive on the same day, and with how much notice?
  2. When do deliveries peak?
  3. How many standby truck-loads should be kept each morning for same-day orders?
  4. Is customer behaviour stable over time, or do planning rules go stale (drift)?
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRUCK_M3 = 7.5            # volume per truck trip (parameter published with the dataset)
COST_IDLE, COST_MISS = 1.0, 3.0      # ASSUMPTION: a missed same-day load costs 3x an idle standby load
Q = COST_MISS / (COST_MISS + COST_IDLE)  # newsvendor critical ratio = 0.75

o = pd.read_csv("data/processed/orders.csv", parse_dates=["day", "ordered_time", "delivery_time"])
dyn = o[o.has_same_day_info].copy()
dyn["loads"] = np.ceil(dyn.quantity_m3 / TRUCK_M3)

# ---------- 1-2. descriptive facts ----------
facts = {
    "orders": int(len(o)), "days": int(o.day.nunique()),
    "period": f"{o.day.min():%b %Y} - {o.day.max():%b %Y}",
    "same_day_share_of_orders_pct": round(dyn.same_day.mean() * 100, 1),
    "same_day_share_of_volume_pct": round(dyn[dyn.same_day].quantity_m3.sum() / dyn.quantity_m3.sum() * 100, 1),
    "same_day_notice_hours_median": round(dyn.notice_hours.median(), 1),
    "same_day_notice_hours_p10": round(dyn.notice_hours.quantile(0.1), 1),
    "same_day_notice_hours_p90": round(dyn.notice_hours.quantile(0.9), 1),
    "order_size_m3_median_prebooked": float(dyn[~dyn.same_day].quantity_m3.median()),
    "order_size_m3_median_same_day": float(dyn[dyn.same_day].quantity_m3.median()),
    "deliveries_in_peak_hour_pct": round((o.delivery_time.dt.hour == 7).mean() * 100, 1),
}

# ---------- 3. daily standby planning (decided at the start of the day) ----------
day = dyn.groupby(["day", "same_day"]).loads.sum().unstack(fill_value=0)
day.columns = ["prebooked_loads", "same_day_loads"]
day = day[day.prebooked_loads > 0]
day["ratio"] = day.same_day_loads / day.prebooked_loads


def recent(train, years=3):
    return train[train.index >= train.index.max() - pd.DateOffset(years=years)]


RULES = {
    "Same standby every day (median)": lambda tr, te: np.full(len(te), np.round(tr.same_day_loads.median())),
    "Same standby every day (P75)": lambda tr, te: np.full(len(te), np.round(tr.same_day_loads.quantile(Q))),
    "Scaled to morning bookings (median)": lambda tr, te: np.round(tr.ratio.median() * te.prebooked_loads),
    "Scaled to morning bookings (P75, newsvendor)": lambda tr, te: np.round(tr.ratio.quantile(Q) * te.prebooked_loads),
    "Scaled, median, last 3 years only": lambda tr, te: np.round(recent(tr).ratio.median() * te.prebooked_loads),
    "Scaled, P75, last 3 years only": lambda tr, te: np.round(recent(tr).ratio.quantile(Q) * te.prebooked_loads),
}


def cost(plan, actual):
    return float((COST_IDLE * np.maximum(plan - actual, 0) + COST_MISS * np.maximum(actual - plan, 0)).sum())


rows = []
for period, years in {"validation": range(2014, 2017), "test": range(2017, 2022)}.items():
    for yr in years:                                  # each year is planned using only earlier years
        tr, te = day[day.index.year < yr], day[day.index.year == yr]
        rows.append((period, yr, "No standby", cost(0 * te.same_day_loads, te.same_day_loads), len(te), len(te)))
        for name, rule in RULES.items():
            plan = rule(tr, te)
            rows.append((period, yr, name, cost(plan, te.same_day_loads), int((te.same_day_loads > plan).sum()), len(te)))
bt = pd.DataFrame(rows, columns=["period", "year", "rule", "cost", "days_short", "days"])
table = bt.groupby(["period", "rule"]).agg(cost=("cost", "sum"), days_short=("days_short", "sum"),
                                           days=("days", "sum")).reset_index()
flat = table[table.rule == "Same standby every day (median)"].set_index("period").cost
table["cost_vs_flat_rule_pct"] = ((table.cost / table.period.map(flat) - 1) * 100).round(0)
table.to_csv("reports/standby_backtest.csv", index=False)
for period in ["validation", "test"]:
    t = table[table.period == period].sort_values("cost")
    print(f"\n=== Standby rules, {period} ({t.days.iloc[0]} real days) | cost vs 'same standby every day' ===")
    print(t[["rule", "cost_vs_flat_rule_pct", "days_short"]].to_string(index=False))

# ---------- 4. drift: did same-day ordering change? ----------
by_year = dyn.groupby(dyn.day.dt.year).same_day.mean().mul(100).round(1)
ref = dyn[dyn.day < "2017-01-01"].groupby("day").same_day.mean()          # reference period
share_by_day = dyn.groupby("day").same_day.mean()
rolling = share_by_day.rolling(8).mean()                                    # ~ last 3 months of sampled days
lower = ref.mean() - 3 * ref.std() / np.sqrt(8)
alarm = rolling[(rolling.index >= "2017-01-01") & (rolling < lower)]
facts.update({
    "same_day_share_by_year_pct": by_year.to_dict(),
    "same_day_share_2011_2018_pct": round(dyn[dyn.day.dt.year <= 2018].same_day.mean() * 100, 1),
    "same_day_share_2019_2021_pct": round(dyn[dyn.day.dt.year >= 2019].same_day.mean() * 100, 1),
    "drift_alarm_first_date": None if alarm.empty else f"{alarm.index[0]:%b %Y}",
})
best_valid = table[table.period == "validation"].sort_values("cost").iloc[0]
facts["standby"] = {
    "assumed_cost_ratio_miss_to_idle": COST_MISS / COST_IDLE,
    "best_rule_on_validation": best_valid.rule,
    "validation_cost_vs_flat_pct": float(best_valid.cost_vs_flat_rule_pct),
    "test_cost_vs_flat_pct_same_rule": float(table[(table.period == "test") &
                                                   (table.rule == best_valid.rule)].cost_vs_flat_rule_pct.iloc[0]),
    "test_cost_vs_flat_pct_last3y_median": float(table[(table.period == "test") &
                                                       (table.rule == "Scaled, median, last 3 years only")].cost_vs_flat_rule_pct.iloc[0]),
}
json.dump(facts, open("reports/orders_headline.json", "w"), indent=2, default=str)
print("\nFacts:", json.dumps(facts, indent=1, default=str))

# ---------- charts ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
hourly = o.delivery_time.dt.hour.value_counts().sort_index()
axes[0].bar(hourly.index, hourly.values / o.day.nunique(), color="tab:blue")
axes[0].set_title("Orders due per hour (average real day)")
axes[0].set_xlabel("hour of day")
axes[1].bar(by_year.index.astype(str), by_year.values, color=["tab:orange" if y >= 2019 else "tab:blue" for y in by_year.index])
axes[1].set_title("Share of orders placed the same day (%)")
plt.tight_layout()
plt.savefig("reports/figures/orders_patterns.png", dpi=150)
