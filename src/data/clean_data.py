import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "processed" / "interactions_raw.parquet"

df = pd.read_parquet(IN_PATH)

print("Missing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique users:", df["user_id"].nunique())
print("Unique tracks:", df["track_id"].nunique())

df = df.dropna(subset=["track_id"])
df = df.drop_duplicates()
df["timestamp"] = pd.to_datetime(df["timestamp"])

user_counts = df["user_id"].value_counts()
df = df[df["user_id"].isin(user_counts[user_counts >= 20].index)]

track_counts = df["track_id"].value_counts()
df = df[df["track_id"].isin(track_counts[track_counts >= 10].index)]

print("Final rows:", len(df))
print("Final users:", df["user_id"].nunique())
print("Final tracks:", df["track_id"].nunique())

OUT_PATH = ROOT / "data" / "processed" / "interactions_clean.parquet"
df.to_parquet(OUT_PATH, index=False)

print("Saved clean dataset.")
