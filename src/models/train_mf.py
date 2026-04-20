import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"
USER_MAP_PATH = ROOT / "data" / "processed" / "user_to_idx.pkl"
TRACK_MAP_PATH = ROOT / "data" / "processed" / "track_to_idx.pkl"


# ----------------------------
# Config
# ----------------------------
EMBED_DIM = 64
BATCH_SIZE = 4096
EPOCHS = 3
NEGATIVES_PER_POS = 3
LEARNING_RATE = 1e-3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# Reproducibility
# ----------------------------
random.seed(SEED)
torch.manual_seed(SEED)


# ----------------------------
# Dataset
# ----------------------------
class ImplicitDataset(Dataset):
    def __init__(self, positives, user_seen, num_items, negatives_per_pos=3):
        self.positives = positives
        self.user_seen = user_seen
        self.num_items = num_items
        self.negatives_per_pos = negatives_per_pos

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        user_idx, pos_item_idx = self.positives[idx]

        item_indices = [pos_item_idx]
        labels = [1.0]

        seen_items = self.user_seen[user_idx]

        neg_count = 0
        while neg_count < self.negatives_per_pos:
            neg_item_idx = random.randint(0, self.num_items - 1)
            if neg_item_idx not in seen_items:
                item_indices.append(neg_item_idx)
                labels.append(0.0)
                neg_count += 1

        user_indices = [user_idx] * (1 + self.negatives_per_pos)

        return (
            torch.tensor(user_indices, dtype=torch.long),
            torch.tensor(item_indices, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
        )


def collate_fn(batch):
    user_idxs = torch.cat([x[0] for x in batch], dim=0)
    item_idxs = torch.cat([x[1] for x in batch], dim=0)
    labels = torch.cat([x[2] for x in batch], dim=0)
    return user_idxs, item_idxs, labels


# ----------------------------
# Model
# ----------------------------
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embed_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idxs, item_idxs):
        u = self.user_emb(user_idxs)
        i = self.item_emb(item_idxs)

        dot = (u * i).sum(dim=1)
        ub = self.user_bias(user_idxs).squeeze(1)
        ib = self.item_bias(item_idxs).squeeze(1)

        return dot + ub + ib


# ----------------------------
# Main
# ----------------------------
def main():
    print("Device:", DEVICE)

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    train = train.sample(n=3_000_000, random_state=42).copy()

    user_to_idx = pd.read_pickle(USER_MAP_PATH)
    track_to_idx = pd.read_pickle(TRACK_MAP_PATH)

    num_users = len(user_to_idx)
    num_items = len(track_to_idx)

    print("Users:", num_users)
    print("Tracks:", num_items)

    # Keep only rows whose IDs are in the mappings
    train = train[
        train["user_id"].isin(user_to_idx.keys()) &
        train["track_id"].isin(track_to_idx.keys())
    ].copy()

    test = test[
        test["user_id"].isin(user_to_idx.keys()) &
        test["track_id"].isin(track_to_idx.keys())
    ].copy()

    print("Train rows after mapping filter:", len(train))
    print("Test rows after mapping filter:", len(test))

    train["user_idx"] = train["user_id"].map(user_to_idx)
    train["item_idx"] = train["track_id"].map(track_to_idx)

    test["user_idx"] = test["user_id"].map(user_to_idx)
    test["item_idx"] = test["track_id"].map(track_to_idx)

    positives = list(zip(train["user_idx"].tolist(), train["item_idx"].tolist()))

    # user -> set(items seen in train)
    user_seen = (
        train.groupby("user_idx")["item_idx"]
        .agg(lambda x: set(x.tolist()))
        .to_dict()
    )

    dataset = ImplicitDataset(
        positives=positives,
        user_seen=user_seen,
        num_items=num_items,
        negatives_per_pos=NEGATIVES_PER_POS,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        embed_dim=EMBED_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # ----------------------------
    # Training
    # ----------------------------
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_examples = 0

        for user_idxs, item_idxs, labels in loader:
            user_idxs = user_idxs.to(DEVICE)
            item_idxs = item_idxs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(user_idxs, item_idxs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        avg_loss = total_loss / total_examples
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.6f}")

    # ----------------------------
    # Evaluation: Recall@10
    # ----------------------------
    model.eval()

    test = test[test["user_idx"].isin(user_seen.keys())]

    with torch.no_grad():
        item_emb = model.item_emb.weight.data.to(DEVICE)          # [num_items, d]
        item_bias = model.item_bias.weight.data.squeeze(1).to(DEVICE)  # [num_items]

        hits = 0
        total = 0

        for _, row in test.iterrows():
            user_idx = int(row["user_idx"])
            true_item = int(row["item_idx"])

            seen_items = user_seen.get(user_idx, set())

            user_vec = model.user_emb.weight[user_idx]   # [d]
            user_b = model.user_bias.weight[user_idx].squeeze()  # scalar

            scores = torch.matmul(item_emb, user_vec) + item_bias + user_b

            # exclude seen training items
            seen_tensor = torch.tensor(list(seen_items), dtype=torch.long, device=DEVICE)
            scores[seen_tensor] = -1e9

            topk = torch.topk(scores, k=10).indices.tolist()

            if true_item in topk:
                hits += 1
            total += 1

        recall_at_10 = hits / total if total > 0 else 0.0

    print("MF Hits@10:", hits)
    print("MF Users evaluated:", total)
    print("MF Recall@10:", recall_at_10)

    torch.save(model.state_dict(), ROOT / "results" / "mf_model.pt")


if __name__ == "__main__":
    main()
