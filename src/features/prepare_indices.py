import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

train_path = ROOT / "data/processed/train.parquet"
out_dir = ROOT / "data/processed"

train = pd.read_parquet(train_path)

# Create mappings
user_ids = train["user_id"].unique()
track_ids = train["track_id"].unique()

user_to_idx = {u: i for i, u in enumerate(user_ids)}
track_to_idx = {t: i for i, t in enumerate(track_ids)}

print("Users:", len(user_to_idx))
print("Tracks:", len(track_to_idx))

# Save mappings
pd.Series(user_to_idx).to_pickle(out_dir / "user_to_idx.pkl")
pd.Series(track_to_idx).to_pickle(out_dir / "track_to_idx.pkl")

print("Mappings saved.")
