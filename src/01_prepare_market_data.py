"""Step 1: clean India's official monthly cement & steel production indices.

Source: Office of the Economic Adviser, DPIIT, Government of India - Index of Eight Core Industries
(https://eaindustry.nic.in/ici_download_data.asp). Two official series:
  * base 2011-12 = 100, April 2011 - May 2026 (the long history)
  * base 2022-23 = 100, April 2023 - latest (the new series)
We clean the dates, then extend the old series with the new one using a linking factor
(the average old/new ratio over the months both series cover).
"""
import pandas as pd

RAW = "data/raw/"
OLD_FILE = RAW + "Core_Industries_2011_12_20260720.xlsx"
NEW_FILE = RAW + "Core_Industries_2022_23_20261021.xlsx"

# --- old series (base 2011-12) ---
old = pd.read_excel(OLD_FILE, sheet_name="Index", header=0)
old = old.rename(columns={old.columns[0]: "month"})
old["month"] = pd.to_datetime(old["month"], errors="coerce")
old = old.dropna(subset=["month"])                      # drops the yearly summary rows at the bottom
# Data-quality fix: 51 dates are stored with a random day (e.g. 18-Jan-2020), so snap to month start
old["month"] = old["month"].dt.to_period("M").dt.to_timestamp()
expected = pd.date_range(old["month"].min(), old["month"].max(), freq="MS")
assert list(old["month"]) == list(expected), "months are missing or out of order"
old = old.set_index("month")
pick = lambda df, word: df[[c for c in df.columns if word in str(c)][0]].astype(float)
old = pd.DataFrame({"cement": pick(old, "Cement"), "steel": pick(old, "Steel")})

# --- new series (base 2022-23) ---
new = pd.read_excel(NEW_FILE, sheet_name="Index", header=1)
new = new.rename(columns={new.columns[0]: "month"})
new["month"] = pd.to_datetime(new["month"], format="%b-%y", errors="coerce")
new = new.dropna(subset=["month"]).set_index("month")
new = pd.DataFrame({"cement": pick(new, "Cement"), "steel": pick(new, "Steel")})

# --- link the two series ---
overlap = old.index.intersection(new.index)
factors = (old.loc[overlap] / new.loc[overlap]).mean()
spread = ((old.loc[overlap] / new.loc[overlap]).std() / factors * 100).round(2)
extra = new.loc[new.index > old.index.max()] * factors
market = pd.concat([old.assign(source="official base 2011-12"),
                    extra.assign(source="base 2022-23, linked")]).round(2)
market.index.name = "month"
market.to_csv("data/processed/market_monthly.csv")

print(f"Old series: {old.index.min():%b %Y} - {old.index.max():%b %Y} ({len(old)} months)")
print(f"New series: {new.index.min():%b %Y} - {new.index.max():%b %Y}; overlap {len(overlap)} months")
print("Linking factors (old/new):", factors.round(3).to_dict(), "| variation %:", spread.to_dict())
print(f"Final: {market.index.min():%b %Y} - {market.index.max():%b %Y} ({len(market)} months)")
print("Check vs official Apr-Jul 2012 cement (106.8, 107.1, 103.2, 101.6):",
      market.loc["2012-04":"2012-07", "cement"].round(1).tolist())
