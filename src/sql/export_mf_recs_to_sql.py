import sqlite3
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "recsys.db"

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"

USER_MAP_PATH = ROOT / "data" / "processed" / "user_to_idx.pkl"
TRACK_MAP_PATH = ROOT / "data" / "processed" / "track_to_idx.pkl"

MODEL_PATH = ROOT / "results" / "mf_model.pt"  # optional if you save model


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_idxs, item_idxs):
        u = self.user_emb(user_idxs)
        i = self.item_emb(item_idxs)
        dot = (u * i).sum(dim=1)
        ub = self.user_bias(user_idxs).squeeze(1)
        ib = self.item_bias(item_idxs).squeeze(1)
        return dot + ub + ib


def main():
    print("Loading data...")

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    user_to_idx = pd.read_pickle(USER_MAP_PATH)
    track_to_idx = pd.read_pickle(TRACK_MAP_PATH)

    idx_to_track = {v: k for k, v in track_to_idx.items()}

    # Load trained model
    print("Loading model...")

    EMBED_DIM = 64

    model = MatrixFactorization(
        num_users=len(user_to_idx),
        num_items=len(track_to_idx),
        embed_dim=EMBED_DIM,
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    num_items = len(track_to_idx)

    # user → seen tracks
    user_seen = (
        train.groupby("user_id")["track_id"]
        .agg(set)
        .to_dict()
    )

    rec_rows = []

    print("Generating recommendations...")

    with torch.no_grad():
        item_emb = model.item_emb.weight.data
        item_bias = model.item_bias.weight.data.squeeze(1)

        for user_id in test["user_id"].unique():
            if user_id not in user_to_idx:
                continue

            user_idx = user_to_idx[user_id]
            seen_tracks = user_seen.get(user_id, set())

            user_vec = model.user_emb.weight[user_idx]
            user_b = model.user_bias.weight[user_idx].squeeze()

            scores = torch.matmul(item_emb, user_vec) + item_bias + user_b

            # exclude seen items
            seen_indices = [
                track_to_idx[t]
                for t in seen_tracks
                if t in track_to_idx
            ]

            if len(seen_indices) > 0:
                scores[seen_indices] = -1e9

            topk = torch.topk(scores, k=10).indices.tolist()

            for rank, item_idx in enumerate(topk, start=1):
                rec_rows.append({
                    "user_id": user_id,
                    "model_name": "mf",
                    "rank": rank,
                    "track_id": idx_to_track[item_idx],
                    "score": float(scores[item_idx].item())
                })

    recs_df = pd.DataFrame(rec_rows)

    print("Saving to SQLite...")

    conn = sqlite3.connect(DB_PATH)

    recs_df.to_sql("recommendations", conn, if_exists="replace", index=False)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_recs_user
        ON recommendations(user_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_recs_model
        ON recommendations(model_name)
    """)

    conn.commit()
    conn.close()

    print("Saved recommendations to:", DB_PATH)
    print(recs_df.head())


if __name__ == "__main__":
    main()
