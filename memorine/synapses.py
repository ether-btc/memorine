"""
Memorine synapses — cross-agent intelligence.
One agent learns, everyone benefits.
"""

import time

from . import cortex


def share(conn, fact_id, from_agent, to_agent=None):
    """Share a fact with another agent (or all agents if to_agent is None)."""
    now = time.time()
    existing = conn.execute(
        "SELECT id FROM shared WHERE fact_id = ? AND from_agent = ? "
        "AND to_agent IS ?",
        (fact_id, from_agent, to_agent)
    ).fetchone()
    if existing:
        return existing["id"]

    cur = conn.execute(
        "INSERT INTO shared (fact_id, from_agent, to_agent, shared_at) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, from_agent, to_agent, now)
    )
    conn.commit()
    return cur.lastrowid


def share_fact(conn, from_agent, fact_text, to_agent=None, category="shared"):
    """Learn a fact and immediately share it."""
    fact_id, _ = cortex.learn(
        conn, from_agent, fact_text, category=category, source="shared"
    )
    share(conn, fact_id, from_agent, to_agent)
    return fact_id


def shared_with_me(conn, agent_id, limit=20):
    """Get all facts shared with this agent."""
    rows = conn.execute("""
        SELECT f.*, s.from_agent, s.shared_at
        FROM shared s
        JOIN facts f ON f.id = s.fact_id
        WHERE (s.to_agent = ? OR s.to_agent IS NULL)
          AND s.from_agent != ?
          AND f.active = 1
        ORDER BY s.shared_at DESC
        LIMIT ?
    """, (agent_id, agent_id, limit)).fetchall()
    return [dict(r) for r in rows]


def team_knowledge(conn, category=None, limit=50):
    """Get collective knowledge shared across all agents."""
    sql = """
        SELECT f.*, s.from_agent, s.shared_at
        FROM shared s
        JOIN facts f ON f.id = s.fact_id
        WHERE s.to_agent IS NULL AND f.active = 1
    """
    params = []
    if category:
        sql += " AND f.category = ?"
        params.append(category)
    sql += " ORDER BY s.shared_at DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params).fetchall()]
