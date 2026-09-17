
import pandas as pd

file_path = "building_consumption.csv"

# Read only a small sample
df = pd.read_csv(file_path, nrows=1000)

print("First 10 rows:")
print(df.head(10).to_string(index=False))

print("\nUnique campuses in this sample:")
print(df["campus_id"].unique())

print("\nUnique meters in this sample:")
print(df["meter_id"].unique())

print("\nTimestamp range in this sample:")
print(df["timestamp"].min())
print(df["timestamp"].max())

print("\nConsumption statistics:")
print(df["consumption"].describe())