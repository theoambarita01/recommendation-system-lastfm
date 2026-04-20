import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CUSTOM_PATH = ROOT / "data" / "raw" / "theo_lastfm_scrobbles.csv"
MAIN_PATH = ROOT / "data" / "processed" / "interactions_clean.parquet"
OUT_PATH = ROOT / "data" / "processed" / "interactions_with_custom.parquet"


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
    # ----------------------------
    # Load custom Last.fm scrobbles
    # ----------------------------
    df = pd.read_csv(CUSTOM_PATH)

    df = df.rename(columns={
        "artist": "artist_name",
        "track": "track_name",
        "track_mbid": "track_id",
    })

    df["timestamp"] = pd.to_datetime(df["uts"], unit="s", errors="coerce")
    df["user_id"] = "user_theo"

    df = df[["user_id", "timestamp", "track_id", "track_name", "artist_name"]].copy()

    print("Custom rows loaded:", len(df))
    print("Missing track_id in custom rows:", df["track_id"].isna().sum())

    # ----------------------------
    # Load main dataset
    # ----------------------------
    main_df = pd.read_parquet(MAIN_PATH).copy()

    # normalized fallback key
    main_df["key"] = (
        normalize_text(main_df["artist_name"]) + "___" +
        normalize_text(main_df["track_name"])
    )

    df["key"] = (
        normalize_text(df["artist_name"]) + "___" +
        normalize_text(df["track_name"])
    )

    # ----------------------------
    # Match custom rows
    # ----------------------------
    with_id = df[df["track_id"].notna()].copy()
    without_id = df[df["track_id"].isna()].copy()

    # 1) direct match by track_id
    valid_track_ids = main_df[["track_id"]].dropna().drop_duplicates()
    matched_by_id = with_id.merge(valid_track_ids, on="track_id", how="inner")

    # 2) fallback match by artist_name + track_name
    fallback_lookup = main_df[["key", "track_id"]].drop_duplicates()
    matched_by_text = without_id.drop(columns=["track_id"], errors="ignore").merge(
        fallback_lookup,
        on="key",
        how="inner"
    )

    matched_df = pd.concat([matched_by_id, matched_by_text], ignore_index=True)
    matched_df = matched_df[["user_id", "timestamp", "track_id", "track_name", "artist_name"]].copy()

    print("Matched rows:", len(matched_df))
    print("Coverage:", len(matched_df) / len(df) if len(df) > 0 else 0.0)

    # ----------------------------
    # Append to main dataset
    # ----------------------------
    combined = pd.concat([main_df.drop(columns=["key"]), matched_df], ignore_index=True)

    combined.to_parquet(OUT_PATH, index=False)

    print("Saved combined dataset to:", OUT_PATH)
    print("Combined rows:", len(combined))


if __name__ == "__main__":
    main()
