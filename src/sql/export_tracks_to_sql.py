import pandas as pd
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "interactions_raw.parquet"
DB_PATH = ROOT / "results" / "recsys.db"


def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)

    print("Creating track metadata...")
    tracks = df[["track_id", "track_name", "artist_name"]].drop_duplicates()

    print("Saving to SQLite...")
    conn = sqlite3.connect(DB_PATH)
    tracks.to_sql("tracks", conn, if_exists="replace", index=False)

    conn.close()
    print("Done!")

if __name__ == "__main__":
    main()
