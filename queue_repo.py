import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_PROD_PATH = Path("queue/queue.db")
DB_INJECTED_PATH = Path("queue/queue_injected.db")


def get_connection(injected: bool = False) -> sqlite3.Connection:
    db_path = DB_INJECTED_PATH if injected else DB_PROD_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def initialize_db(injected: bool = False):
    conn = get_connection(injected)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queue_items (
            job_id TEXT PRIMARY KEY,
            lifecycle TEXT,
            status TEXT,
            decision TEXT,
            source TEXT,
            captured_at TEXT,
            updated_at TEXT,
            published_at TEXT,
            payload_json TEXT NOT NULL
        );
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lifecycle ON queue_items(lifecycle);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON queue_items(status);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_decision ON queue_items(decision);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON queue_items(updated_at);")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    initialize_db(injected=False)
    initialize_db(injected=True)
    print("SQLite databases initialized.")
