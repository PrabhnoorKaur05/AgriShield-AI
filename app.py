import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

df = pd.read_csv("agrishield_deploy.csv")
district_df = pd.read_csv("district_intelligence.csv")
commodity_df = pd.read_csv("commodity_intelligence.csv")

df["Date"] = pd.to_datetime(df["Date"])

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "District Intelligence",
        "Commodity Intelligence",
        "Strategic Risk Lab",
        "Investigation Panel"
    ]
)

st.title("AgriShield AI: Agricultural Risk Intelligence Platform")
st.caption("AI-powered early warning system for agricultural price instability")

# Overview

if page == "Overview":

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("ML Anomalies", (df["ML_Anomaly_Label"] == -1).sum())
    col3.metric("High Risk Cases", df["High_Risk_Flag"].sum())

    st.markdown("""
    ### Why This Matters

    This platform identifies early warning signals in agricultural markets by combining:
    - Price volatility
    - Supply shocks
    - Regional deviations
    - ML-based anomaly detection

    It helps:
    - Detect sudden price spikes
    - Identify unstable districts
    - Support supply chain decisions
    - Enable proactive interventions
    """)

    st.markdown("""Real-World Use Case:
A sudden spike in onion prices in a specific district is detected as an anomaly.\n
The system flags it as high-risk, helping authorities or traders take early action.\n
This can prevent:\n
- Supply shortages\n
- Market instability\n
- Price inflation shocks""")

    fig = px.histogram(
        df,
        x="Composite_Risk_Score",
        nbins=70,
        title="Composite Risk Score Distribution"
    )
    st.plotly_chart(fig, width="stretch")

    national_trend = df.groupby("Date")["Composite_Risk_Score"].mean().reset_index()

    fig2 = px.line(
        national_trend,
        x="Date",
        y="Composite_Risk_Score",
        title="National Risk Trend"
    )
    st.plotly_chart(fig2, width="stretch")


# District Intelligence

if page == "District Intelligence":

    district_df_sorted = district_df.sort_values("Avg_Risk", ascending=False)

    fig = px.bar(
        district_df_sorted.head(20),
        x="District",
        y="Avg_Risk",
        color="Risk_Tier",
        title="Top 20 Districts by Systemic Risk"
    )
    st.plotly_chart(fig, width="stretch")

    selected_district = st.selectbox(
        "Select District",
        sorted(district_df["District"].unique())
    )

    district_data = district_df[district_df["District"] == selected_district].iloc[0]
    district_market_df = df[df["District"] == selected_district]

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Tier", district_data["Risk_Tier"])
    col2.metric("National Rank", int(district_data["National_Rank"]))
    col3.metric("Risk Momentum", district_data["Risk_Momentum"])

    anomaly_rate = (district_market_df["ML_Anomaly_Label"] == -1).mean() * 100
    st.metric("ML Anomaly Rate", f"{anomaly_rate:.2f}%")

    trend = (
        district_market_df.groupby("Date")["Composite_Risk_Score"]
        .mean()
        .reset_index()
    )

    national_trend = df.groupby("Date")["Composite_Risk_Score"].mean().reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=trend["Date"], y=trend["Composite_Risk_Score"], name="District"))
    fig2.add_trace(go.Scatter(x=national_trend["Date"], y=national_trend["Composite_Risk_Score"], name="National"))
    fig2.update_layout(title="District vs National Risk Trend")
    st.plotly_chart(fig2, width="stretch")

    price = district_market_df["Price_Risk"].mean()
    supply = district_market_df["Supply_Risk"].mean()
    regional = district_market_df["Regional_Risk"].mean()
    anomaly = anomaly_rate

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[price, supply, regional, anomaly],
        theta=["Price Risk", "Supply Risk", "Regional Risk", "Anomaly Rate"],
        fill="toself"
    ))
    radar.update_layout(title="District Risk Radar")
    st.plotly_chart(radar, width="stretch")

    if district_data["Risk_Tier"] == "Critical":
        summary = f"{selected_district} shows systemic instability driven by structural price shocks and abnormal transaction patterns."
    elif district_data["Risk_Tier"] == "High":
        summary = f"{selected_district} exhibits elevated risk with localized stress clusters."
    else:
        summary = f"{selected_district} maintains moderate systemic conditions."

    st.subheader("Executive Risk Brief")
    st.info(summary)


# Commodity Intelligence

if page == "Commodity Intelligence":

    fig = px.bar(
        commodity_df.head(20),
        x="Commodity",
        y="Commodity_Stress_Index",
        title="Top Commodities by Stress Index"
    )
    st.plotly_chart(fig, width="stretch")

    selected_commodity = st.selectbox(
        "Select Commodity",
        sorted(df["Commodity"].unique())
    )

    # normalize selection
    selected_commodity = selected_commodity.strip().lower()

    commodity_df = df[df["Commodity"] == selected_commodity]
    st.write("Rows in commodity:", len(commodity_df))

    top_alerts = commodity_df[commodity_df["ML_Anomaly_Label"] == -1] \
        .sort_values("Composite_Risk_Score", ascending=False) \
        .head(10)

    st.subheader("Top Risk Alerts")

    if len(top_alerts) > 0:
        st.dataframe(top_alerts[[
            "Date", "Commodity", "District",
            "Modal_Price", "Composite_Risk_Score"
        ]])
    else:
        st.write("No high-risk anomalies detected")

    trend = (
        commodity_df.groupby("Date")["Composite_Risk_Score"]
        .mean()
        .reset_index()
    )

    fig2 = px.line(

        trend,
        x="Date",
        y="Composite_Risk_Score",
        title=f"Risk Trend for {selected_commodity}"
    )
    st.plotly_chart(fig2, width="stretch")

    # --- Forecast (Simple Moving Average) ---
    commodity_df["Forecast"] = commodity_df["Modal_Price"].rolling(7).mean()

    st.subheader("Price Forecast (7-day moving avg)")
    st.line_chart(
        commodity_df.set_index("Date")[["Modal_Price", "Forecast"]]
    )

    avg_risk = commodity_df["Composite_Risk_Score"].mean()
    volatility = commodity_df["Price_Volatility"].mean()
    total = len(commodity_df)
    anomalies = (commodity_df["ML_Anomaly_Label"] == -1).sum()

    anomaly_rate = (anomalies / total) * 100 if total > 0 else 0
    high_risk_rate = commodity_df["High_Risk_Flag"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Risk Score", round(avg_risk, 2))
    col2.metric("Price Volatility", round(volatility, 4))
    col3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    col4.metric("High Risk Frequency", f"{high_risk_rate*100:.2f}%")

    st.subheader("Anomaly Breakdown")

    anomaly_view = commodity_df[commodity_df["ML_Anomaly_Label"] == -1]

    if len(anomaly_view) > 0:
        cols = ["Commodity", "District", "Composite_Risk_Score"]
        if "Anomaly_Reason" in anomaly_view.columns:
            cols.insert(2, "Anomaly_Reason")
        st.dataframe(anomaly_view[cols].head(10))
    else:
        st.write("No anomalies detected")

    st.subheader("Risk Insight")

    if avg_risk > 40:
        st.error("High systemic risk — sustained instability detected")
    elif anomaly_rate > 1:
        st.warning("Moderate risk — anomaly-driven fluctuations detected")
    else:
        st.success("Stable market conditions")

    st.caption("Insight combines anomaly detection, volatility, and systemic risk scoring")

    national_avg_risk = df["Composite_Risk_Score"].mean()
    national_volatility = df["Price_Volatility"].mean()
    national_anomaly = (df["ML_Anomaly_Label"] == -1).mean() * 100

    st.subheader("Commodity Risk Interpretation")

    explanation = ""

    if avg_risk > national_avg_risk:
        explanation += "• This commodity has above-average systemic risk compared to national baseline.\n\n"

    if volatility > national_volatility:
        explanation += "• Price volatility is higher than typical agricultural commodities.\n\n"

    if anomaly_rate > national_anomaly:
        explanation += "• AI anomaly detection flags this commodity more frequently than average.\n\n"

    if high_risk_rate > 0.5:
        explanation += "• A significant proportion of transactions fall into high-risk classification.\n\n"

    if explanation == "":
        explanation = "• This commodity currently exhibits relatively stable systemic behavior."

    st.info(explanation)

# Strategic Risk Lab

if page == "Strategic Risk Lab":

    cluster_data = district_df[["Avg_Risk", "Anomaly_Rate", "High_Risk_Rate"]].fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    district_df["Cluster"] = kmeans.fit_predict(scaled)

    fig = px.scatter(
        district_df,
        x="Avg_Risk",
        y="Anomaly_Rate",
        color="Cluster",
        hover_data=["District"],
        title="District Risk Clusters"
    )
    st.plotly_chart(fig, width="stretch")

    heatmap_data = df.pivot_table(
        values="Composite_Risk_Score",
        index="District",
        columns="Commodity",
        aggfunc="mean"
    ).fillna(0)

    fig2 = px.imshow(
        heatmap_data.head(20),
        title="Risk Intensity Matrix (Top Districts)"
    )
    st.plotly_chart(fig2, width="stretch")


# Investigation Panel

if page == "Investigation Panel":

    show_high = st.checkbox("Show Only High Risk")
    show_anomaly = st.checkbox("Show Only ML Anomalies")

    filtered = df.copy()

    if show_high:
        filtered = filtered[filtered["High_Risk_Flag"] == True]

    if show_anomaly:
        filtered = filtered[filtered["ML_Anomaly_Label"] == -1]

    filtered = filtered.sort_values("Composite_Risk_Score", ascending=False)

    st.dataframe(filtered, use_container_width=True)

    st.download_button(
        "Download Filtered Data",
        filtered.to_csv(index=False),
        "filtered_risk_data.csv"
    )