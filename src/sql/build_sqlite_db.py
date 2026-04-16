import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "results" / "recsys.db"

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "val.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"

def main():
    conn = sqlite3.connect(DB_PATH)

    train = pd.read_parquet(TRAIN_PATH)[["user_id", "track_id", "timestamp"]].copy()
    val = pd.read_parquet(VAL_PATH)[["user_id", "track_id", "timestamp"]].copy()
    test = pd.read_parquet(TEST_PATH)[["user_id", "track_id", "timestamp"]].copy()

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    interactions = pd.concat([train, val, test], ignore_index=True)

    interactions.to_sql("interactions", conn, if_exists="replace", index=False)

    users = pd.DataFrame({"user_id": interactions["user_id"].drop_duplicates()})
    users.to_sql("users", conn, if_exists="replace", index=False)

    tracks = interactions[["track_id"]].drop_duplicates().copy()
    tracks.to_sql("tracks", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_track ON interactions(track_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_split ON interactions(split)")

    conn.commit()
    conn.close()

    print("Built SQLite DB at:", DB_PATH)

if __name__ == "__main__":
    main()
