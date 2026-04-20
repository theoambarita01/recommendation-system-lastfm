import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "val.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"

OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 10
TOP_K_TRACKS = 20000


def build_examples(df_hist, df_target, track_to_idx, seq_len=10):
    examples = []

    hist_grouped = df_hist.groupby("user_id")["track_id"].agg(list).to_dict()

    for _, row in df_target.iterrows():
        user = row["user_id"]
        target_track = row["track_id"]

        if user not in hist_grouped:
            continue
        if target_track not in track_to_idx:
            continue

        hist = hist_grouped[user]
        hist = [t for t in hist if t in track_to_idx]

        if len(hist) == 0:
            continue

        seq = hist[-seq_len:]
        seq_idx = [track_to_idx[t] for t in seq]

        if len(seq_idx) < seq_len:
            seq_idx = [0] * (seq_len - len(seq_idx)) + seq_idx

        target_idx = track_to_idx[target_track]
        examples.append((user, seq_idx, target_idx))

    return pd.DataFrame(examples, columns=["user_id", "sequence", "target_idx"])


def main():
    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = train.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    top_tracks = train["track_id"].value_counts().head(TOP_K_TRACKS).index.tolist()

    # reserve 0 for padding
    track_to_idx = {track: i + 1 for i, track in enumerate(top_tracks)}

    print("Top-K tracks kept:", len(track_to_idx))

    val_examples = build_examples(train, val, track_to_idx, seq_len=SEQ_LEN)
    test_examples = build_examples(train, test, track_to_idx, seq_len=SEQ_LEN)

    # training examples: sliding windows from train itself
    train_examples = []

    for user, grp in train.groupby("user_id"):
        seq_tracks = [t for t in grp["track_id"].tolist() if t in track_to_idx]

        if len(seq_tracks) < 2:
            continue

        seq_idx_all = [track_to_idx[t] for t in seq_tracks]

        for i in range(1, len(seq_idx_all)):
            input_seq = seq_idx_all[max(0, i - SEQ_LEN):i]
            target_idx = seq_idx_all[i]

            if len(input_seq) < SEQ_LEN:
                input_seq = [0] * (SEQ_LEN - len(input_seq)) + input_seq

            train_examples.append((user, input_seq, target_idx))

    train_examples = pd.DataFrame(
        train_examples, columns=["user_id", "sequence", "target_idx"]
    )

    train_examples.to_pickle(OUT_DIR / "gru_train.pkl")
    val_examples.to_pickle(OUT_DIR / "gru_val.pkl")
    test_examples.to_pickle(OUT_DIR / "gru_test.pkl")
    pd.Series(track_to_idx).to_pickle(OUT_DIR / "gru_track_to_idx.pkl")

    print("Train examples:", len(train_examples))
    print("Val examples:", len(val_examples))
    print("Test examples:", len(test_examples))


if __name__ == "__main__":
    main()
