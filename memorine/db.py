"""
Memorine database layer.
SQLite + FTS5 — the entire brain in one file.
"""

import os
import sqlite3
import threading

_LOCAL = threading.local()

DEFAULT_DB_PATH = os.path.expanduser("~/.memorine/memorine.db")


def get_connection(db_path=None):
    path = db_path or DEFAULT_DB_PATH
    key = f"conn_{path}"
    if not hasattr(_LOCAL, key) or getattr(_LOCAL, key) is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        setattr(_LOCAL, key, conn)
    return getattr(_LOCAL, key)


def init_schema(conn):
    c = conn.cursor()

    # -- CORTEX: facts + associations --
    c.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            confidence REAL DEFAULT 1.0,
            weight REAL DEFAULT 1.0,
            source TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            superseded_by INTEGER REFERENCES facts(id),
            active INTEGER DEFAULT 1
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS fact_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_a INTEGER NOT NULL REFERENCES facts(id),
            fact_b INTEGER NOT NULL REFERENCES facts(id),
            relation TEXT DEFAULT 'related',
            strength REAL DEFAULT 1.0,
            created_at REAL NOT NULL
        )
    """)

    # -- HIPPOCAMPUS: events + causal chains --
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            event TEXT NOT NULL,
            context TEXT,
            tags TEXT,
            timestamp REAL NOT NULL,
            causal_parent INTEGER REFERENCES events(id)
        )
    """)

    # -- CEREBELLUM: procedures + learning --
    c.execute("""
        CREATE TABLE IF NOT EXISTS procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            version INTEGER DEFAULT 1,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            total_runs INTEGER DEFAULT 0,
            successes INTEGER DEFAULT 0,
            failures INTEGER DEFAULT 0,
            avg_duration_ms REAL DEFAULT 0,
            active INTEGER DEFAULT 1
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS procedure_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            procedure_id INTEGER NOT NULL REFERENCES procedures(id),
            step_order INTEGER NOT NULL,
            description TEXT NOT NULL,
            total_runs INTEGER DEFAULT 0,
            successes INTEGER DEFAULT 0,
            failures INTEGER DEFAULT 0,
            last_error TEXT,
            skip_recommended INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS procedure_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            procedure_id INTEGER NOT NULL REFERENCES procedures(id),
            started_at REAL NOT NULL,
            completed_at REAL,
            success INTEGER,
            error TEXT,
            duration_ms REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS run_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES procedure_runs(id),
            step_order INTEGER NOT NULL,
            description TEXT NOT NULL,
            success INTEGER,
            error TEXT,
            duration_ms REAL,
            timestamp REAL NOT NULL
        )
    """)

    # -- SYNAPSES: cross-agent sharing --
    c.execute("""
        CREATE TABLE IF NOT EXISTS shared (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_id INTEGER NOT NULL REFERENCES facts(id),
            from_agent TEXT NOT NULL,
            to_agent TEXT,
            shared_at REAL NOT NULL
        )
    """)

    # -- FTS5 indexes for fast search --
    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
        USING fts5(fact, category, content=facts, content_rowid=id)
    """)

    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS events_fts
        USING fts5(event, context, tags, content=events, content_rowid=id)
    """)

    c.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS procedures_fts
        USING fts5(name, description, content=procedures, content_rowid=id)
    """)

    # -- Triggers to keep FTS in sync --
    for table, fts, cols in [
        ("facts", "facts_fts", "fact, category"),
        ("events", "events_fts", "event, context, tags"),
        ("procedures", "procedures_fts", "name, description"),
    ]:
        c.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON {table} BEGIN
                INSERT INTO {fts}(rowid, {cols})
                VALUES (new.id, {', '.join(f'new.{c.strip()}' for c in cols.split(','))});
            END
        """)
        c.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE ON {table} BEGIN
                INSERT INTO {fts}({fts}, rowid, {cols})
                VALUES ('delete', old.id, {', '.join(f'old.{c.strip()}' for c in cols.split(','))});
                INSERT INTO {fts}(rowid, {cols})
                VALUES (new.id, {', '.join(f'new.{c.strip()}' for c in cols.split(','))});
            END
        """)
        c.execute(f"""
            CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON {table} BEGIN
                INSERT INTO {fts}({fts}, rowid, {cols})
                VALUES ('delete', old.id, {', '.join(f'old.{c.strip()}' for c in cols.split(','))});
            END
        """)

    # Indexes
    c.execute("CREATE INDEX IF NOT EXISTS idx_facts_agent ON facts(agent_id, active)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(agent_id, category, active)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_facts_updated ON facts(updated_at DESC)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent_id, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_causal ON events(causal_parent)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_procedures_agent ON procedures(agent_id, name)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_run_steps_run ON run_steps(run_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_shared_to ON shared(to_agent)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_shared_fact ON shared(fact_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fact_links_a ON fact_links(fact_a)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fact_links_b ON fact_links(fact_b)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_facts_superseded ON facts(superseded_by)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
    # Index for cleanup_faded() — covers WHERE active = 1 AND (last_accessed, access_count)
    c.execute("CREATE INDEX IF NOT EXISTS idx_facts_decay ON facts(active, last_accessed, access_count)")

    conn.commit()

    # Optional: initialize embedding storage if available
    try:
        from . import embeddings
        if embeddings.is_available():
            embeddings.init_vec_schema(conn)
    except Exception:
        pass


def get_default_agent_id(conn):
    """Return the agent_id with the most active facts, or 'default' if DB is empty."""
    row = conn.execute("""
        SELECT agent_id FROM facts WHERE active = 1
        GROUP BY agent_id ORDER BY COUNT(*) DESC LIMIT 1
    """).fetchone()
    return row[0] if row else "default"


def get_stats(conn, agent_id):
    """Database statistics for an agent."""
    stats = {}
    stats["facts_active"] = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE agent_id = ? AND active = 1",
        (agent_id,)
    ).fetchone()[0]
    stats["facts_inactive"] = conn.execute(
        "SELECT COUNT(*) FROM facts WHERE agent_id = ? AND active = 0",
        (agent_id,)
    ).fetchone()[0]
    stats["events_total"] = conn.execute(
        "SELECT COUNT(*) FROM events WHERE agent_id = ?", (agent_id,)
    ).fetchone()[0]
    stats["procedures_total"] = conn.execute(
        "SELECT COUNT(*) FROM procedures WHERE agent_id = ? AND active = 1",
        (agent_id,)
    ).fetchone()[0]
    stats["shared_total"] = conn.execute(
        "SELECT COUNT(*) FROM shared WHERE from_agent = ?", (agent_id,)
    ).fetchone()[0]

    # Check if embeddings are active
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM fact_embeddings"
        ).fetchone()[0]
        stats["embeddings"] = count
    except Exception:
        stats["embeddings"] = 0

    # DB file size
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    try:
        stats["db_size_bytes"] = os.path.getsize(path)
    except Exception:
        stats["db_size_bytes"] = 0

    return stats
