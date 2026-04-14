"""
Memorine hippocampus — episodes, temporal memory, and causal chains.
What happened, when, and why.
"""

import json
import re
import time

# FTS5 special operator characters stripped from each token before quoting.
FTS5_SPECIAL_CHARS = frozenset('*"(),')


def _sanitize_token(token: str) -> str:
    """Strip FTS5 special chars from token, then wrap in double quotes."""
    cleaned = "".join(c for c in token if c not in FTS5_SPECIAL_CHARS)
    return f'"{cleaned}"'


def log_event(conn, agent_id, event, context=None, tags=None,
              caused_by=None):
    """Record an event with optional context and causal link."""
    now = time.time()
    ctx_json = json.dumps(context) if context else None
    tags_str = ",".join(t for t in tags if t and t.strip()) if tags else None

    cur = conn.execute(
        "INSERT INTO events (agent_id, event, context, tags, timestamp, "
        "causal_parent) VALUES (?, ?, ?, ?, ?, ?)",
        (agent_id, event, ctx_json, tags_str, now, caused_by)
    )
    conn.commit()
    return cur.lastrowid


def recall_events(conn, agent_id, query=None, since=None, until=None,
                  tags=None, limit=20, offset=0):
    """Search events by text, time range, or tags."""
    if query:
        tokens = re.findall(r"\w{3,}", query.lower())
        if tokens:
            fts_terms = [_sanitize_token(t) for t in tokens]
            fts_query = " OR ".join(fts_terms)
            sql = """
                SELECT * FROM events
                WHERE id IN (SELECT rowid FROM events_fts WHERE events_fts MATCH ?)
                AND agent_id = ?
            """
            params = [fts_query, agent_id]
        else:
            return []
    else:
        sql = "SELECT * FROM events WHERE agent_id = ?"
        params = [agent_id]

    if since:
        sql += " AND timestamp >= ?"
        params.append(since)
    if until:
        sql += " AND timestamp <= ?"
        params.append(until)
    if tags:
        for tag in tags:
            sql += " AND tags LIKE ?"
            params.append(f"%{tag}%")

    sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        if r.get("context"):
            try:
                r["context"] = json.loads(r["context"])
            except (json.JSONDecodeError, TypeError):
                pass
        if r.get("tags") and isinstance(r["tags"], str):
            r["tags"] = r["tags"].split(",")
        results.append(r)
    return results


def causal_chain(conn, event_id, direction="up", max_depth=10):
    """Trace the causal chain from an event.

    direction='up': find what caused this event (ancestors)
    direction='down': find what this event caused (descendants)

    Circular reference protection:
      - "up": breaks immediately when causal_parent is already in visited
      - "down": breaks immediately when encountering already-visited node,
                preventing re-enqueue of circular chain descendants
    """
    chain = []
    visited = set()

    if direction == "up":
        current_id = event_id
        for _ in range(max_depth):
            if current_id is None or current_id in visited:
                break
            visited.add(current_id)
            row = conn.execute(
                "SELECT * FROM events WHERE id = ?", (current_id,)
            ).fetchone()
            if not row:
                break
            chain.append(dict(row))
            current_id = row["causal_parent"]
        chain.reverse()
    else:
        queue = [event_id]
        for _ in range(max_depth):
            if not queue:
                break
            current_id = queue.pop(0)
            if current_id is None or current_id in visited:
                break
            visited.add(current_id)
            children = conn.execute(
                "SELECT * FROM events WHERE causal_parent = ?",
                (current_id,)
            ).fetchall()
            for child in children:
                chain.append(dict(child))
                queue.append(child["id"])

    return chain


def timeline(conn, agent_id, since=None, until=None, limit=50, offset=0):
    """Get a chronological timeline of events."""
    sql = "SELECT * FROM events WHERE agent_id = ?"
    params = [agent_id]
    if since:
        sql += " AND timestamp >= ?"
        params.append(since)
    if until:
        sql += " AND timestamp <= ?"
        params.append(until)
    sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
