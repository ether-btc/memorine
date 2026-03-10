"""
Memorine hippocampus — episodes, temporal memory, and causal chains.
What happened, when, and why.
"""

import json
import re
import time


def log_event(conn, agent_id, event, context=None, tags=None, caused_by=None):
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
        fts_query = " OR ".join(re.findall(r"\w{3,}", query.lower()))
        if fts_query:
            sql = """
                SELECT * FROM events
                WHERE id IN (SELECT rowid FROM events_fts WHERE events_fts MATCH ?)
                AND agent_id = ?
            """
            params = [fts_query, agent_id]
        else:
            sql = "SELECT * FROM events WHERE agent_id = ?"
            params = [agent_id]
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
        event_dict = dict(row)
        if event_dict.get("context"):
            try:
                event_dict["context"] = json.loads(event_dict["context"])
            except (json.JSONDecodeError, TypeError):
                pass
        if event_dict.get("tags") and isinstance(event_dict["tags"], str):
            event_dict["tags"] = event_dict["tags"].split(",")
        results.append(event_dict)
    return results


def causal_chain(conn, event_id, direction="up", max_depth=10):
    """Trace the causal chain from an event.

    direction='up': find what caused this event (ancestors)
    direction='down': find what this event caused (descendants)
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
            if current_id in visited:
                continue
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
