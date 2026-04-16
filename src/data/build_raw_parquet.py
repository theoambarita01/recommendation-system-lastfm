import pandas as pd
from pathlib import Path

import sys
print("PYTHON EXE:", sys.executable)

# repo root = parent of src/
ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "userid-timestamp-artid-artname-traid-traname.tsv"
OUT_PATH = ROOT / "data" / "processed" / "interactions_raw.parquet"


def main():
    cols = ["user_id","timestamp","artist_id","artist_name","track_id","track_name"]

    print("Looking for:", RAW_PATH)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Not found: {RAW_PATH}\nCheck filename + data/raw folder.")

    df = pd.read_csv(
        RAW_PATH,
        sep="\t",
        header=None,
        names=cols,
        engine="python",
        on_bad_lines="skip",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")

    print("Rows:", len(df))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
