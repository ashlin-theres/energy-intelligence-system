
import pandas as pd

file_path = "building_consumption.csv"
output_path = "sample_energy.csv"

# Read the CSV in chunks to avoid using too much memory
chunks = pd.read_csv(file_path, chunksize=200_000)

samples = []
rows_to_keep = 0

for chunk in chunks:
    # Keep only the first campus and meter for now
    filtered = chunk[
        (chunk["campus_id"] == 1) &
        (chunk["meter_id"] == 1)
    ]

    samples.append(filtered)
    rows_to_keep += len(filtered)

    # Stop after collecting enough data
    if rows_to_keep >= 20_000:
        break

df = pd.concat(samples, ignore_index=True)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Rows:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print(df.head())