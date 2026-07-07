from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "daily"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "mlb_moneyline.db"))
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

for path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
