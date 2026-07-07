from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import DB_PATH


def get_conn(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn
