"""Step 3: convert the real ready-mix concrete orders (MATLAB files) into one clean CSV.

Source: Tzanetos, A. & Blondin, M. (2023). Real operational data for the Concrete Delivery Problem.
Mendeley Data, V1. https://doi.org/10.17632/b78bc94d95.1 (licence CC BY 4.0).
A concrete producer in Quebec, Canada: 263 real days between 2011 and 2021, 9,300 orders.
The orders are stored as MATLAB "table" objects, which scipy cannot read, so we use mat-io.
"""
import glob
import os
import tempfile
import warnings
import zipfile

import matio
import pandas as pd

warnings.filterwarnings("ignore")
ZIP = "data/raw/concrete_delivery_orders_mendeley.zip"

with tempfile.TemporaryDirectory() as tmp:
    zipfile.ZipFile(ZIP).extractall(tmp)
    files = sorted(glob.glob(os.path.join(tmp, "*", "Instances", "*.mat")))
    frames = []
    for f in files:
        o = matio.load_from_mat(f)["orders"].copy()
        o["instance"] = os.path.basename(f)[:-4]
        frames.append(o)

orders = pd.concat(frames, ignore_index=True)
first = lambda v: str(v[0]) if hasattr(v, "__len__") and len(v) else ""
clean = pd.DataFrame({
    "day": orders["ExpectedDeliveryTime"].dt.normalize(),
    "instance": orders["instance"],
    "order_id": orders["OrderDayID"].astype(int),
    "usage": orders["Usage"].apply(first).str.upper().str.strip(),     # e.g. PLANCHER = floor slab
    "quantity_m3": orders["OrderedQuantity"].astype(float),
    "ordered_time": orders["OrderedTime"],
    "delivery_time": orders["ExpectedDeliveryTime"],
})
# Orders booked before the day carry a midnight timestamp; same-day orders carry the real time
clean["same_day"] = clean["ordered_time"].dt.strftime("%H:%M:%S") != "00:00:00"
# "s" instances were published without same-day information, so keep them out of same-day analysis
clean["has_same_day_info"] = ~clean["instance"].str.endswith("s")
clean["notice_hours"] = ((clean.delivery_time - clean.ordered_time).dt.total_seconds() / 3600).where(clean.same_day)
clean.to_csv("data/processed/orders.csv", index=False)

print(f"{len(clean):,} orders on {clean.day.nunique()} real days, "
      f"{clean.day.min():%b %Y} - {clean.day.max():%b %Y}")
