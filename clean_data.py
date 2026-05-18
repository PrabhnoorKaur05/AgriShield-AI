import pandas as pd
import os

data_folder = "csvs"
all_dfs = []

for file in os.listdir(data_folder):

    if file.endswith(".csv"):

        file_path = os.path.join(data_folder, file)

        try:
            # Skip first row (garbage header)
            df = pd.read_csv(file_path, skiprows=1)

            # Remove unwanted unnamed columns
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

            # Standardize column names
            df.columns = [
                "State",
                "District",
                "Market",
                "Commodity_Group",
                "Commodity",
                "Date",
                "Arrival_Quantity",
                "Arrival_Unit",
                "Modal_Price",
                "Price_Unit"
            ]

            # Convert date
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Convert numeric columns
            df["Arrival_Quantity"] = pd.to_numeric(df["Arrival_Quantity"], errors="coerce")
            df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")

            # Extract category from filename
            name = file.replace(".csv", "")
            parts = name.split(",")

            if len(parts) == 2:
                category = parts[0].strip()
            else:
                category = "Unknown"

            df["Category"] = category
            df["Source_File"] = file

            # Remove rows with missing important values
            df = df.dropna(subset=["Date", "Modal_Price"])

            all_dfs.append(df)

            print("Loaded:", file)

        except Exception as e:
            print("Skipped:", file, e)

# Combine all files
final_df = pd.concat(all_dfs, ignore_index=True)

# Remove duplicates
final_df = final_df.drop_duplicates()

# Create Month and Year columns
final_df["Month"] = final_df["Date"].dt.month
final_df["Year"] = final_df["Date"].dt.year

# Save clean dataset
final_df.to_csv("final_agri_dataset.csv", index=False)

print("\n✅ Clean dataset created successfully!")
print("Total rows:", len(final_df))