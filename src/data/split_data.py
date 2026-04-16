import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data" / "processed" / "interactions_clean.parquet"
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "val.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"

df = pd.read_parquet(IN_PATH)

# sort so each user's history is chronological
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# position of each event within a user's timeline
df["event_rank"] = df.groupby("user_id").cumcount()
df["user_len"] = df.groupby("user_id")["user_id"].transform("size")

train = df[df["event_rank"] < df["user_len"] - 2].copy()
val = df[df["event_rank"] == df["user_len"] - 2].copy()
test = df[df["event_rank"] == df["user_len"] - 1].copy()

# drop helper cols
for part in (train, val, test):
    part.drop(columns=["event_rank", "user_len"], inplace=True)

train.to_parquet(TRAIN_PATH, index=False)
val.to_parquet(VAL_PATH, index=False)
test.to_parquet(TEST_PATH, index=False)

print("Train rows:", len(train))
print("Val rows:", len(val))
print("Test rows:", len(test))

print("Train users:", train["user_id"].nunique())
print("Val users:", val["user_id"].nunique())
print("Test users:", test["user_id"].nunique())
