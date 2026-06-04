"""Database utilities — SQLite-backed opportunity store."""
import sqlite3
import json
import os
from pathlib import Path

_HERE = Path(__file__).parent
DB_PATH = _HERE.parent / "data" / "engageiq.db"
CSV_PATH = _HERE.parent / "data" / "opportunities.csv"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id           TEXT PRIMARY KEY,
            source       TEXT NOT NULL,
            record_type  TEXT DEFAULT 'repo',
            data_source  TEXT DEFAULT 'offline',
            title        TEXT NOT NULL,
            description  TEXT,
            url          TEXT,
            domain       TEXT NOT NULL,
            language     TEXT,
            tags         TEXT,
            stars        INTEGER DEFAULT 0,
            forks        INTEGER DEFAULT 0,
            contributors INTEGER DEFAULT 0,
            open_issues  INTEGER DEFAULT 0,
            good_first_issues INTEGER DEFAULT 0,
            comments     INTEGER DEFAULT 0,
            upvotes      INTEGER DEFAULT 0,
            activity_score REAL DEFAULT 0,
            growth_rate  REAL DEFAULT 0,
            created_at   TEXT,
            updated_at   TEXT,
            ingested_at  TEXT DEFAULT (datetime('now'))
        );
        -- Add columns to existing tables if upgrading
        CREATE INDEX IF NOT EXISTS idx_opp_record_type ON opportunities(record_type);
        CREATE INDEX IF NOT EXISTS idx_opp_data_source ON opportunities(data_source);

        CREATE TABLE IF NOT EXISTS user_feedback (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            opportunity_id  TEXT,
            feedback        TEXT,
            session_id      TEXT DEFAULT 'default',
            timestamp       TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS user_profiles (
            session_id   TEXT PRIMARY KEY,
            name         TEXT,
            role         TEXT,
            interests    TEXT,
            time_budget  INTEGER DEFAULT 5,
            updated_at   TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_opp_domain  ON opportunities(domain);
        CREATE INDEX IF NOT EXISTS idx_opp_source  ON opportunities(source);
        CREATE INDEX IF NOT EXISTS idx_opp_stars   ON opportunities(stars DESC);
    """)
    conn.commit()
    conn.close()


def bulk_insert(records: list[dict]):
    if not records:
        return
    conn = get_connection()
    conn.executemany("""
        INSERT OR IGNORE INTO opportunities
            (id, source, record_type, data_source, title, description, url, domain,
             language, tags, stars, forks, contributors, open_issues, good_first_issues,
             comments, upvotes, activity_score, growth_rate, created_at, updated_at)
        VALUES
            (:id, :source, :record_type, :data_source, :title, :description, :url, :domain,
             :language, :tags, :stars, :forks, :contributors, :open_issues, :good_first_issues,
             :comments, :upvotes, :activity_score, :growth_rate, :created_at, :updated_at)
    """, [_normalize_record(r) for r in records])
    conn.commit()
    conn.close()


def _normalize_record(r: dict) -> dict:
    """Ensure all required fields exist with sensible defaults."""
    return {
        "id": r.get("id", ""),
        "source": r.get("source", ""),
        "record_type": r.get("record_type", "repo"),
        "data_source": r.get("_data_source", r.get("data_source", "offline")),
        "title": r.get("title", ""),
        "description": r.get("description", ""),
        "url": r.get("url", ""),
        "domain": r.get("domain", ""),
        "language": r.get("language", ""),
        "tags": r.get("tags", "[]"),
        "stars": int(r.get("stars", 0) or 0),
        "forks": int(r.get("forks", 0) or 0),
        "contributors": int(r.get("contributors", 0) or 0),
        "open_issues": int(r.get("open_issues", 0) or 0),
        "good_first_issues": int(r.get("good_first_issues", 0) or 0),
        "comments": int(r.get("comments", 0) or 0),
        "upvotes": int(r.get("upvotes", 0) or 0),
        "activity_score": float(r.get("activity_score", 0) or 0),
        "growth_rate": float(r.get("growth_rate", 0) or 0),
        "created_at": r.get("created_at", ""),
        "updated_at": r.get("updated_at", ""),
    }


def get_all_as_df():
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM opportunities", conn)
    conn.close()
    return df


def get_count() -> int:
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM opportunities").fetchone()[0]
    conn.close()
    return n


def save_feedback(opportunity_id: str, feedback: str, session_id: str = "default"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO user_feedback (opportunity_id, feedback, session_id) VALUES (?,?,?)",
        (opportunity_id, feedback, session_id),
    )
    conn.commit()
    conn.close()


def get_feedback(session_id: str = "default") -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM user_feedback WHERE session_id=? ORDER BY timestamp DESC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_profile(session_id: str, name: str, role: str, interests: list, time_budget: int):
    conn = get_connection()
    conn.execute(
        """INSERT OR REPLACE INTO user_profiles
           (session_id, name, role, interests, time_budget, updated_at)
           VALUES (?,?,?,?,?,datetime('now'))""",
        (session_id, name, role, json.dumps(interests), time_budget),
    )
    conn.commit()
    conn.close()


def get_profile(session_id: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE session_id=?", (session_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["interests"] = json.loads(d["interests"] or "[]")
        return d
    return None
