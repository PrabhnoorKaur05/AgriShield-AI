import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

Z_THRESHOLD = 3.5

df = pd.read_csv("final_agri_dataset.csv")

df["Commodity"] = df["Commodity"].astype(str).str.strip().str.lower()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Commodity", "Date"])

df["Price_Change"] = df.groupby("Commodity")["Modal_Price"].diff()

rolling_mean = (
    df.groupby("Commodity")["Modal_Price"]
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)

rolling_std = (
    df.groupby("Commodity")["Modal_Price"]
    .rolling(7)
    .std()
    .reset_index(level=0, drop=True)
)

epsilon = 1e-5
df["Price_Zscore"] = (df["Modal_Price"] - rolling_mean) / (rolling_std + epsilon)
df["Price_Volatility"] = rolling_std

df["Arrival_Change"] = df.groupby("Commodity")["Arrival_Quantity"].diff()
df["Supply_Shock"] = df["Arrival_Change"] / (df["Arrival_Quantity"] + 1)

district_avg = (
    df.groupby(["District", "Commodity"])["Modal_Price"]
    .transform("mean")
)

df["District_Price_Deviation"] = df["Modal_Price"] - district_avg

features = df[
    [
        "Modal_Price",
        "Arrival_Quantity",
        "Price_Zscore",
        "Price_Volatility",
        "Supply_Shock",
        "District_Price_Deviation",
    ]
]

features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

for col in ["Price_Zscore", "Supply_Shock", "District_Price_Deviation"]:
    df[col] = df[col].clip(-5, 5)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- Z SCORE FIRST ---
df["z_score"] = df.groupby("Commodity")["Modal_Price"].transform(
    lambda x: np.abs((x - x.mean()) / x.std())
)

# --- Z LABEL ---
df["z_label"] = df["z_score"] > Z_THRESHOLD

# --- ISOLATION FOREST ---
iso = IsolationForest(contamination=0.01, random_state=42)
df["iso_label"] = iso.fit_predict(scaled_features)

# --- FILTER ISO USING Z-SCORE (IMPORTANT) ---
df["iso_label"] = np.where(df["z_score"] > 2.5, df["iso_label"], 1)

df["ML_Anomaly_Label"] = np.where(
    (df["iso_label"] == -1) & (df["z_label"]),
    -1,
    1
)

df["Anomaly_Reason"] = np.where(
    df["z_score"] > Z_THRESHOLD,
    "Extreme Price Spike",
    np.where(df["iso_label"] == -1, "Pattern Anomaly", "Normal")
)

print("\nML Anomaly Distribution:")
print(df["ML_Anomaly_Label"].value_counts())

def normalize(series):
    series = series.replace([np.inf, -np.inf], np.nan).fillna(0)
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return pd.Series(0, index=series.index)
    return (series - min_val) / (max_val - min_val)

df["Price_Risk"] = normalize(abs(df["Price_Zscore"]))
df["Supply_Risk"] = normalize(abs(df["Supply_Shock"]))
df["Regional_Risk"] = normalize(abs(df["District_Price_Deviation"]))

df["Composite_Risk_Score"] = (
    0.4 * df["Price_Risk"]
    + 0.3 * df["Supply_Risk"]
    + 0.2 * df["Regional_Risk"]
    + 0.05 * (df["ML_Anomaly_Label"] == -1).astype(int)
) * 100

threshold = df["Composite_Risk_Score"].quantile(0.95)
df["High_Risk_Flag"] = df["Composite_Risk_Score"] > threshold

district_stats = df.groupby("District").agg(
    Avg_Risk=("Composite_Risk_Score", "mean"),
    High_Risk_Count=("High_Risk_Flag", "sum"),
    Total_Records=("Composite_Risk_Score", "count"),
    ML_Anomaly_Count=("ML_Anomaly_Label", lambda x: (x == -1).sum())
)

district_stats["High_Risk_Percentage"] = (
    district_stats["High_Risk_Count"] / district_stats["Total_Records"]
)

district_stats["ML_Anomaly_Percentage"] = (
    district_stats["ML_Anomaly_Count"] / district_stats["Total_Records"]
)

district_stats["Volume_Weight"] = np.log1p(district_stats["Total_Records"])

district_stats["Structural_Risk_Index"] = (
    0.4 * normalize(district_stats["Avg_Risk"]) +
    0.3 * normalize(district_stats["High_Risk_Percentage"]) +
    0.2 * normalize(district_stats["ML_Anomaly_Percentage"]) +
    0.1 * normalize(district_stats["Volume_Weight"])
) * 100

market_volume = df.groupby(["District", "Market"])["Arrival_Quantity"].sum().reset_index()

district_total_volume = market_volume.groupby("District")["Arrival_Quantity"].transform("sum")

market_volume["Market_Share"] = market_volume["Arrival_Quantity"] / district_total_volume

max_share = market_volume.groupby("District")["Market_Share"].max()
total_volume = market_volume.groupby("District")["Arrival_Quantity"].sum()

adjusted_concentration = max_share * np.log1p(total_volume)

district_stats["Market_Concentration_Index"] = normalize(adjusted_concentration)

district_stats["Final_Systemic_Risk"] = (
    0.85 * normalize(district_stats["Structural_Risk_Index"]) +
    0.15 * district_stats["Market_Concentration_Index"]
) * 100

district_stats = district_stats.sort_values("Final_Systemic_Risk", ascending=False)

print("Total ML Anomalies:", (df["ML_Anomaly_Label"] == -1).sum())
print("Total High Risk Cases:", df["High_Risk_Flag"].sum())

print("\nTop 10 Final Systemic Risk Districts:")
print(district_stats.head(10)[
    ["Structural_Risk_Index", "Market_Concentration_Index", "Final_Systemic_Risk"]
])

df.sample(20000, random_state=42).to_csv(
    "agrishield_deploy.csv",
    index=False
)
district_stats.to_csv("district_final_systemic_risk.csv")

print("\nAnalysis completed and results saved!")