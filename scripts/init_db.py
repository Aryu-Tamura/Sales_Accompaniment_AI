from pathlib import Path
import sqlite3

BASE = Path(__file__).resolve().parents[1]
DB = BASE / "data" / "app.db"
SCHEMA = BASE / "db" / "schema.sql"
SEED = BASE / "db" / "seed.sql"
DB.parent.mkdir(exist_ok=True)

with sqlite3.connect(DB) as con:
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    con.executescript(SEED.read_text(encoding="utf-8"))
print("DB initialized:", DB)
