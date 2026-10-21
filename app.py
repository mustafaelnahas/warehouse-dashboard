import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Warehouse & Delivery Performance", layout="wide")
st.title("Warehouse & Delivery Performance")
st.caption("Training project • Synthetic delivery data • EGP")

@st.cache_data
def load_data():
    return pd.read_csv("warehouse_delivery_raw.csv")

raw = load_data()
df = raw.copy()
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")
rows_before = len(df)
df = df.drop_duplicates()
df.loc[~df["Customer_Rating"].between(1, 5), "Customer_Rating"] = np.nan
df["Courier"] = df["Courier"].fillna("Unknown")
df["Shipping_Cost_EGP"] = pd.to_numeric(df["Shipping_Cost_EGP"], errors="coerce")
df["Shipping_Cost_EGP"] = df["Shipping_Cost_EGP"].clip(upper=df["Shipping_Cost_EGP"].quantile(.99))
df["Shipping_Cost_EGP"] = df["Shipping_Cost_EGP"].fillna(df["Shipping_Cost_EGP"].median())
df["Customer_Rating"] = df["Customer_Rating"].fillna(df["Customer_Rating"].median())
df["Delivery_Days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.total_seconds() / 86400
df.loc[(df["Delivery_Days"] < 0) | (df["Delivery_Days"] > 30), "Delivery_Days"] = np.nan
df["SLA_Met"] = np.where(
    df["Delivery_Status"].isin(["Delivered On Time", "Delivered Late"]) & df["Delivery_Days"].notna(),
    df["Delivery_Days"] <= df["Promised_Days"], np.nan
)
df["Order_Month"] = df["Order_Date"].dt.to_period("M").astype(str)

st.sidebar.header("Filters")
regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
region = st.sidebar.selectbox("Region", regions)
warehouses = ["All"] + sorted(df["Warehouse"].dropna().unique().tolist())
warehouse = st.sidebar.selectbox("Warehouse", warehouses)
couriers = ["All"] + sorted(df["Courier"].dropna().unique().tolist())
courier = st.sidebar.selectbox("Courier", couriers)
statuses = ["All"] + sorted(df["Delivery_Status"].dropna().unique().tolist())
status = st.sidebar.selectbox("Delivery status", statuses)

f = df.copy()
if region != "All": f = f[f["Region"] == region]
if warehouse != "All": f = f[f["Warehouse"] == warehouse]
if courier != "All": f = f[f["Courier"] == courier]
if status != "All": f = f[f["Delivery_Status"] == status]

delivered = f[f["Delivery_Status"].isin(["Delivered On Time", "Delivered Late"])]
sla = delivered["SLA_Met"].dropna()
on_time_rate = (sla.mean() * 100) if len(sla) else 0
avg_days = delivered["Delivery_Days"].mean()
total_orders = len(f)
late_count = int((f["Delivery_Status"] == "Delivered Late").sum())
cancel_rate = (f["Delivery_Status"].eq("Cancelled").mean() * 100) if len(f) else 0
avg_rating = delivered["Customer_Rating"].mean()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Orders", f"{total_orders:,}")
c2.metric("On-time SLA", f"{on_time_rate:.1f}%")
c3.metric("Avg. delivery days", f"{avg_days:.2f}" if pd.notna(avg_days) else "N/A")
c4.metric("Cancellation rate", f"{cancel_rate:.1f}%")

st.subheader("Delivery performance")
left, right = st.columns(2)
with left:
    status_counts = f["Delivery_Status"].value_counts().rename_axis("Delivery_Status").reset_index(name="Orders")
    fig = px.bar(status_counts, x="Delivery_Status", y="Orders", color="Delivery_Status",
                 title="Orders by delivery status")
    st.plotly_chart(fig, use_container_width=True)
with right:
    region_perf = delivered.groupby("Region", as_index=False).agg(
        Avg_Delivery_Days=("Delivery_Days","mean"),
        Orders=("Order_ID","count"),
        On_Time_Rate=("SLA_Met","mean")
    )
    region_perf["On_Time_Rate"] *= 100
    fig = px.bar(region_perf, x="Region", y="On_Time_Rate", color="Region",
                 title="On-time SLA by region", range_y=[0,100])
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Courier comparison")
courier_perf = delivered.groupby("Courier", as_index=False).agg(
    Orders=("Order_ID","count"),
    Avg_Delivery_Days=("Delivery_Days","mean"),
    Avg_Shipping_Cost_EGP=("Shipping_Cost_EGP","mean"),
    Avg_Rating=("Customer_Rating","mean"),
    SLA_Rate=("SLA_Met","mean")
)
courier_perf["SLA_Rate"] *= 100
st.dataframe(courier_perf.round(2), use_container_width=True)
fig = px.scatter(courier_perf, x="Avg_Shipping_Cost_EGP", y="SLA_Rate", size="Orders",
                 color="Courier", hover_name="Courier",
                 title="Courier trade-off: average shipping cost vs on-time SLA")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Monthly order trend")
monthly = f.groupby("Order_Month", as_index=False).agg(Orders=("Order_ID","count"))
fig = px.line(monthly, x="Order_Month", y="Orders", markers=True, title="Orders by month")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Data quality checks"):
    st.write(f"Rows before removing exact duplicates: {rows_before:,}")
    st.write(f"Rows after removing exact duplicates: {len(df):,}")
    st.write("Missing values remaining after the app's basic cleaning:")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame())
    st.caption("Teaching note: missing delivery dates are not imputed; those orders are excluded from delivery-time and SLA calculations.")

with st.expander("Explore filtered data"):
    st.dataframe(f, use_container_width=True)
    st.download_button("Download filtered data", f.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_warehouse_delivery.csv", mime="text/csv")

st.subheader("Suggested business actions")
st.markdown("""
- Investigate regions with lower on-time SLA; separate distance effects from courier performance.
- Compare courier SLA alongside cost and order volume before reallocating shipments.
- Audit extreme shipping-cost records against invoices before setting a permanent cap.
- Keep in-transit and cancelled orders out of delivery-time averages; report them separately.
- Review customer ratings with sample size in mind before making vendor decisions.
""")
