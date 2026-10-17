"""
Interaction log (SQLite)
-------------------------
Stores every question and answer, plus the retrieval details and the
user's thumbs up/down. SQLite needs no server and no install: it is built
into Python (the `sqlite3` module) and lives in a single file.

Everything is behind these few functions, so if you later move to a hosted
database (e.g. Postgres on Supabase), only THIS file changes.

PRIVACY: this file contains users' legal questions. It is listed in
.gitignore so it never gets pushed to GitHub. Do not commit it.
"""

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
import uuid

DB_PATH = os.environ.get("LAWBOT_DB_PATH", "lawbot_logs.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id                   TEXT PRIMARY KEY,
    session_id           TEXT,
    created_at           TEXT,
    question             TEXT,
    answer               TEXT,
    sections_json        TEXT,
    best_distance        REAL,
    in_scope             INTEGER,
    model                TEXT,
    latency_ms           INTEGER,
    unverified_citations TEXT,
    emergency_flags      TEXT,
    feedback             INTEGER
);
"""


def _connect():
    return sqlite3.connect(DB_PATH)


def init_db():
    with closing(_connect()) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def log_interaction(session_id, question, answer, sections, best_distance,
                    in_scope, model, latency_ms, unverified, emergency_keys) -> str:
    interaction_id = uuid.uuid4().hex
    slim_sections = [
        {"act": s.get("act"), "section": s.get("section"),
         "distance": s.get("distance")} for s in sections
    ]
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
            (interaction_id, session_id,
             datetime.now(timezone.utc).isoformat(timespec="seconds"),
             question, answer, json.dumps(slim_sections), best_distance,
             1 if in_scope else 0, model, latency_ms,
             json.dumps(unverified), json.dumps(emergency_keys)),
        )
        conn.commit()
    return interaction_id


def set_feedback(interaction_id: str, value: int):
    """value: 1 = thumbs up, 0 = thumbs down."""
    with closing(_connect()) as conn:
        conn.execute("UPDATE interactions SET feedback=? WHERE id=?",
                     (value, interaction_id))
        conn.commit()


def fetch_recent(limit: int = 50) -> list:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM interactions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
