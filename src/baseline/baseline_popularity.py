import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"

train = pd.read_parquet(TRAIN_PATH)
test = pd.read_parquet(TEST_PATH)

# global popularity ranking from train only
popular_tracks = train["track_id"].value_counts().index.tolist()

# tracks each user has already seen in train
user_seen = train.groupby("user_id")["track_id"].agg(set).to_dict()

hits = 0
total = 0

for _, row in test.iterrows():
    user = row["user_id"]
    true_track = row["track_id"]
    seen_tracks = user_seen[user]

    recs = []
    for track in popular_tracks:
        if track not in seen_tracks:
            recs.append(track)
        if len(recs) == 10:
            break

    if true_track in recs:
        hits += 1
    total += 1

recall_at_10 = hits / total

print("Hits@10:", hits)
print("Users evaluated:", total)
print("Recall@10:", recall_at_10)

print(test.head())

train_tracks = set(train["track_id"].unique())
test_tracks = set(test["track_id"].unique())

covered = sum(track in train_tracks for track in test["track_id"])
print("Test tracks appearing in train:", covered, "out of", len(test))
print("Coverage rate:", covered / len(test))

top10_global = popular_tracks[:10]

hits = 0
for true_track in test["track_id"]:
    if true_track in top10_global:
        hits += 1

print("Raw global hits:", hits)
print("Raw global Recall@10:", hits / len(test))

print(train["track_id"].value_counts().head(10))
