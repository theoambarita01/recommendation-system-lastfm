from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]

TRAIN_EX_PATH = ROOT / "data" / "processed" / "gru_train.pkl"
TEST_EX_PATH = ROOT / "data" / "processed" / "gru_test.pkl"

SEQ_LEN = 10
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 512
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, df):
        self.sequences = df["sequence"].tolist()
        self.targets = df["target_idx"].tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


class GRURec(nn.Module):
    def __init__(self, num_items, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, num_items)

    def forward(self, x):
        emb = self.embedding(x)
        _, h = self.gru(emb)
        h = h.squeeze(0)
        logits = self.output(h)
        return logits


def recall_at_10(model, loader, device):
    model.eval()
    hits = 0
    total = 0

    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            targets = targets.to(device)

            logits = model(seqs)
            topk = torch.topk(logits, k=10, dim=1).indices

            hits += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
            total += targets.size(0)

    return hits, total, hits / total if total > 0 else 0.0


def main():
    train_df = pd.read_pickle(TRAIN_EX_PATH)
    test_df = pd.read_pickle(TEST_EX_PATH)
    track_to_idx = pd.read_pickle(ROOT / "data" / "processed" / "gru_track_to_idx.pkl")

    # optional dry run if training is too slow
    train_df = train_df.sample(n=min(500_000, len(train_df)), random_state=42).copy()

    num_items = len(track_to_idx) + 1  # +1 for padding idx 0

    print("Device:", DEVICE)
    print("Train examples:", len(train_df))
    print("Test examples:", len(test_df))
    print("Num items:", num_items)

    train_ds = SequenceDataset(train_df)
    test_ds = SequenceDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = GRURec(num_items=num_items, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for seqs, targets in train_loader:
            seqs = seqs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

        avg_loss = total_loss / total_examples
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.6f}")

    hits, total, rec10 = recall_at_10(model, test_loader, DEVICE)
    print("GRU Hits@10:", hits)
    print("GRU Users evaluated:", total)
    print("GRU Recall@10:", rec10)


if __name__ == "__main__":
    main()
