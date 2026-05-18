import pandas as pd
import numpy as np

df = pd.read_csv("agrishield_ai_output.csv")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=["Composite_Risk_Score"], inplace=True)

df["Date"] = pd.to_datetime(df["Date"])

# ---------------- DISTRICT INTELLIGENCE ----------------

district_group = df.groupby("District")

district_df = district_group.agg(
    Avg_Risk=("Composite_Risk_Score", "mean"),
    Anomaly_Rate=("ML_Anomaly_Label", lambda x: (x == -1).mean()),
    Avg_Price_Risk=("Price_Risk", "mean"),
    Avg_Supply_Risk=("Supply_Risk", "mean"),
    Avg_Regional_Risk=("Regional_Risk", "mean"),
    High_Risk_Rate=("High_Risk_Flag", "mean"),
    Total_Records=("Composite_Risk_Score", "count")
).reset_index()

district_df["National_Rank"] = district_df["Avg_Risk"].rank(ascending=False)

latest_date = df["Date"].max()
cutoff_recent = latest_date - pd.Timedelta(days=90)
cutoff_previous = latest_date - pd.Timedelta(days=180)

recent = df[df["Date"] >= cutoff_recent]
previous = df[(df["Date"] >= cutoff_previous) & (df["Date"] < cutoff_recent)]

recent_avg = recent.groupby("District")["Composite_Risk_Score"].mean()
previous_avg = previous.groupby("District")["Composite_Risk_Score"].mean()

district_df["Recent_Risk"] = district_df["District"].map(recent_avg)
district_df["Previous_Risk"] = district_df["District"].map(previous_avg)

district_df["Risk_Acceleration"] = (
    district_df["Recent_Risk"] - district_df["Previous_Risk"]
)

district_df["Risk_Momentum"] = district_df["Risk_Acceleration"].apply(
    lambda x: "Increasing" if x > 2 else ("Decreasing" if x < -2 else "Stable")
)

national_anomaly_rate = (df["ML_Anomaly_Label"] == -1).mean()

district_df["Systemic_Fragility"] = np.where(
    (district_df["High_Risk_Rate"] > 0.5) &
    (district_df["Anomaly_Rate"] > national_anomaly_rate),
    True,
    False
)

def classify_tier(score):
    if score >= district_df["Avg_Risk"].quantile(0.9):
        return "Critical"
    elif score >= district_df["Avg_Risk"].quantile(0.75):
        return "High"
    elif score >= district_df["Avg_Risk"].quantile(0.5):
        return "Moderate"
    else:
        return "Low"

district_df["Risk_Tier"] = district_df["Avg_Risk"].apply(classify_tier)

def generate_report(row):
    text = f"{row['District']} ranks {int(row['National_Rank'])} nationally in systemic agricultural risk. "
    
    if row["Risk_Momentum"] == "Increasing":
        text += "Risk momentum is accelerating. "
    elif row["Risk_Momentum"] == "Decreasing":
        text += "Risk levels are easing. "
    else:
        text += "Risk levels are stable. "

    if row["Avg_Price_Risk"] > 0.6:
        text += "Price volatility is a major contributing factor. "
    if row["Avg_Supply_Risk"] > 0.6:
        text += "Supply irregularities are contributing to instability. "
    if row["Avg_Regional_Risk"] > 0.6:
        text += "Prices deviate significantly from regional norms. "
    if row["Systemic_Fragility"]:
        text += "Market structure indicates systemic fragility. "

    return text

district_df["Executive_Report"] = district_df.apply(generate_report, axis=1)

# ---------------- COMMODITY INTELLIGENCE ----------------

commodity_df = df.groupby("Commodity").agg(
    Avg_Risk=("Composite_Risk_Score", "mean"),
    Volatility=("Composite_Risk_Score", "std"),
    Anomaly_Rate=("ML_Anomaly_Label", lambda x: (x == -1).mean())
).reset_index()

commodity_df["Commodity_Stress_Index"] = (
    commodity_df["Avg_Risk"] * 0.5 +
    commodity_df["Volatility"] * 0.3 +
    commodity_df["Anomaly_Rate"] * 100 * 0.2
)

commodity_df.sort_values("Commodity_Stress_Index", ascending=False, inplace=True)

# ---------------- SAVE OUTPUT ----------------

district_df.to_csv("district_intelligence.csv", index=False)
commodity_df.to_csv("commodity_intelligence.csv", index=False)

print("Risk engine execution complete.")