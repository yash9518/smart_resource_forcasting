"""Decision dashboard. Run locally with:  streamlit run app.py
Reads only the saved results of the src/ scripts (no training happens here)."""
import json
import math

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Construction Demand Forecasting", layout="wide")


@st.cache_data
def load():
    market = pd.read_csv("data/processed/market_monthly.csv", parse_dates=["month"])
    outlook = pd.read_csv("reports/market_outlook.csv", parse_dates=["month"])
    head = json.load(open("reports/market_headline.json"))
    orders = pd.read_csv("data/processed/orders.csv", parse_dates=["day"])
    facts = json.load(open("reports/orders_headline.json"))
    return market, outlook, head, orders, facts


market, outlook, head, orders, facts = load()
st.title("🏗️ Construction Demand Forecasting")
st.caption("Built on real data: Government of India core-industry statistics (2011–2026) and "
           "9,300 real orders from a ready-mix concrete supplier (2011–2021).")
tab_market, tab_planner = st.tabs(["📈 Market outlook (India)", "🚚 Daily standby planner (supplier)"])

with tab_market:
    series = st.radio("Material", ["cement", "steel"], horizontal=True)
    out = outlook[outlook.series == series].reset_index(drop=True)
    h1 = head[f"{series}_h1"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Forecast for {out.month[0]:%b %Y}", f"{out.forecast[0]:.1f}",
              f"{out.growth_vs_last_year_pct[0]:+.1f}% vs same month last year")
    c2.metric("Likely range (80%)", f"{out.low_80[0]:.0f} – {out.high_80[0]:.0f}")
    c3.metric("Tested error (42 unseen months)", f"{h1['test_mape']:.1f}%",
              f"simple rule: {h1['baseline_mape']:.1f}%", delta_color="off")
    history = market.set_index("month")[series].loc["2019-01-01":].rename("actual")
    st.line_chart(pd.concat([history, out.set_index("month")[["forecast", "low_80", "high_80"]]], axis=1))
    st.caption("Production index, base 2011-12 = 100. Model: Holt-Winters, chosen on 2021-23 data and "
               "tested walk-forward on Apr 2023 – Sep 2026.")

with tab_planner:
    st.write("At the start of the day: how many **standby truck-loads** should we keep for same-day orders?")
    c1, c2, c3 = st.columns(3)
    booked = c1.number_input("Truck-loads already booked for today", min_value=1, max_value=300, value=60)
    miss_cost = c2.slider("Cost of a missed load vs an idle standby load", 1.0, 6.0, 3.0, 0.5)
    years = c3.slider("Learn from the last N years", 1, 5, 3)

    dyn = orders[orders.has_same_day_info].copy()
    dyn["loads"] = np.ceil(dyn.quantity_m3 / 7.5)
    day = dyn.groupby(["day", "same_day"]).loads.sum().unstack(fill_value=0).rename(
        columns={False: "prebooked", True: "same_day"})
    day = day[day.prebooked > 0]
    recent = day[day.index >= day.index.max() - pd.DateOffset(years=years)]
    ratio = recent.same_day / recent.prebooked
    q = miss_cost / (miss_cost + 1)                   # newsvendor critical ratio
    standby = math.ceil(ratio.quantile(q) * booked)

    k1, k2, k3 = st.columns(3)
    k1.metric("Recommended standby loads", standby)
    k2.metric("Typical same-day loads (median)", round(ratio.median() * booked))
    k3.metric("Chance of still running short", f"{(1 - q) * 100:.0f}%")
    st.caption(f"Based on {len(recent)} real days. Standby = the {q:.0%} point of past same-day demand "
               "(newsvendor rule: plan higher when running short costs more).")

    st.subheader("Why the rule must be retrained: same-day orders fell after 2018")
    share = pd.Series(facts["same_day_share_by_year_pct"], name="same-day orders (%)")
    st.bar_chart(share)
    st.caption(f"A simple monitoring alarm would have flagged this change in {facts['drift_alarm_first_date']}.")
