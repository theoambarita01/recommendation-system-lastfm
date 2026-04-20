import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "results" / "recsys.db"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT 
    t.track_name,
    t.artist_name,
    r.score
FROM recommendations r
JOIN tracks t ON r.track_id = t.track_id
WHERE r.user_id = 'user_000001'
ORDER BY r.rank
LIMIT 10;
"""

df = pd.read_sql_query(query, conn)

print(df)

conn.close()
