import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = ROOT / "results" / "recsys.db"


def main():
    metrics_df = pd.DataFrame([
        {
            "model_name": "popularity",
            "metric_name": "Recall@10",
            "metric_value": 0.0000,
            "users_evaluated": 983,
        },
        {
            "model_name": "mf",
            "metric_name": "Recall@10",
            "metric_value": 0.004069175991861648,
            "users_evaluated": 983,
        },
        {
            "model_name": "gru",
            "metric_name": "Recall@10",
            "metric_value": 0.09923664122137404,
            "users_evaluated": 393,
        },
    ])

    conn = sqlite3.connect(DB_PATH)

    metrics_df.to_sql("metrics", conn, if_exists="replace", index=False)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_model
        ON metrics(model_name)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_metrics_metric
        ON metrics(metric_name)
    """)

    conn.commit()
    conn.close()

    print("Saved metrics table to:", DB_PATH)
    print(metrics_df)


if __name__ == "__main__":
    main()
