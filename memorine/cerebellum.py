"""
Memorine cerebellum — procedures that learn, optimize, and evolve.
Track what works, what fails, and why. Auto-optimize over time.
"""

import re
import time

from .constants import (
    AUTO_OPTIMIZE_MIN_RUNS,
    STEP_OPTIMIZE_MIN_RUNS,
    STEP_SKIP_FAILURE_RATE,
)


def create_procedure(conn, agent_id, name, description=None, steps=None):
    """Create a new procedure with optional predefined steps."""
    now = time.time()
    cur = conn.execute(
        "INSERT INTO procedures (agent_id, name, description, created_at, "
        "updated_at) VALUES (?, ?, ?, ?, ?)",
        (agent_id, name, description, now, now)
    )
    proc_id = cur.lastrowid

    if steps:
        for i, step_desc in enumerate(steps, 1):
            conn.execute(
                "INSERT INTO procedure_steps (procedure_id, step_order, "
                "description) VALUES (?, ?, ?)",
                (proc_id, i, step_desc)
            )

    conn.commit()
    return proc_id


def start_run(conn, procedure_id):
    """Start a new run of a procedure. Returns run_id."""
    now = time.time()
    cur = conn.execute(
        "INSERT INTO procedure_runs (procedure_id, started_at) "
        "VALUES (?, ?)",
        (procedure_id, now)
    )
    conn.commit()
    return cur.lastrowid


def log_step(conn, run_id, step_order, description, success=True,
             error=None, duration_ms=None, agent_id=None):
    """Log the result of a step in a procedure run."""
    if agent_id:
        check = conn.execute("""
            SELECT pr.id FROM procedure_runs pr
            JOIN procedures p ON p.id = pr.procedure_id
            WHERE pr.id = ? AND p.agent_id = ?
        """, (run_id, agent_id)).fetchone()
        if not check:
            return

    now = time.time()
    conn.execute(
        "INSERT INTO run_steps (run_id, step_order, description, success, "
        "error, duration_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, step_order, description, int(success), error,
         duration_ms, now)
    )

    row = conn.execute(
        "SELECT procedure_id FROM procedure_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row:
        proc_id = row["procedure_id"]
        existing = conn.execute(
            "SELECT id FROM procedure_steps "
            "WHERE procedure_id = ? AND step_order = ?",
            (proc_id, step_order)
        ).fetchone()

        if existing:
            if success:
                conn.execute(
                    "UPDATE procedure_steps SET total_runs = total_runs + 1, "
                    "successes = successes + 1 WHERE id = ?",
                    (existing["id"],)
                )
            else:
                conn.execute(
                    "UPDATE procedure_steps SET total_runs = total_runs + 1, "
                    "failures = failures + 1, last_error = ? WHERE id = ?",
                    (error, existing["id"])
                )
        else:
            conn.execute(
                "INSERT INTO procedure_steps (procedure_id, step_order, "
                "description, total_runs, successes, failures, last_error) "
                "VALUES (?, ?, ?, 1, ?, ?, ?)",
                (proc_id, step_order, description,
                 1 if success else 0, 0 if success else 1,
                 error if not success else None)
            )

    conn.commit()


def complete_run(conn, run_id, success=True, error=None, agent_id=None):
    """Mark a procedure run as complete."""
    if agent_id:
        check = conn.execute("""
            SELECT pr.id FROM procedure_runs pr
            JOIN procedures p ON p.id = pr.procedure_id
            WHERE pr.id = ? AND p.agent_id = ?
        """, (run_id, agent_id)).fetchone()
        if not check:
            return

    now = time.time()
    started = conn.execute(
        "SELECT started_at, procedure_id FROM procedure_runs WHERE id = ?",
        (run_id,)
    ).fetchone()

    if not started:
        return

    duration = (now - started["started_at"]) * 1000
    conn.execute(
        "UPDATE procedure_runs SET completed_at = ?, success = ?, "
        "error = ?, duration_ms = ? WHERE id = ?",
        (now, int(success), error, duration, run_id)
    )

    proc_id = started["procedure_id"]
    if success:
        conn.execute(
            "UPDATE procedures SET total_runs = total_runs + 1, "
            "successes = successes + 1, updated_at = ?, "
            "avg_duration_ms = (avg_duration_ms * total_runs + ?) / (total_runs + 1) "
            "WHERE id = ?",
            (now, duration, proc_id)
        )
    else:
        conn.execute(
            "UPDATE procedures SET total_runs = total_runs + 1, "
            "failures = failures + 1, updated_at = ? WHERE id = ?",
            (now, proc_id)
        )

    conn.commit()

    # Auto-optimize after enough runs
    proc = conn.execute(
        "SELECT total_runs FROM procedures WHERE id = ?", (proc_id,)
    ).fetchone()
    if proc and proc["total_runs"] >= AUTO_OPTIMIZE_MIN_RUNS:
        optimize(conn, proc_id)


def optimize(conn, procedure_id):
    """Auto-optimize: flag steps that fail too often."""
    steps = conn.execute(
        "SELECT * FROM procedure_steps WHERE procedure_id = ? "
        "ORDER BY step_order",
        (procedure_id,)
    ).fetchall()

    for step in steps:
        if step["total_runs"] >= STEP_OPTIMIZE_MIN_RUNS:
            fail_rate = step["failures"] / step["total_runs"]
            if fail_rate > STEP_SKIP_FAILURE_RATE:
                conn.execute(
                    "UPDATE procedure_steps SET skip_recommended = 1 "
                    "WHERE id = ?",
                    (step["id"],)
                )

    conn.commit()


def get_procedure(conn, agent_id, name):
    """Get a procedure with all its steps and stats."""
    proc = conn.execute(
        "SELECT * FROM procedures WHERE agent_id = ? AND name = ? "
        "AND active = 1 ORDER BY version DESC LIMIT 1",
        (agent_id, name)
    ).fetchone()

    if not proc:
        return None

    result = dict(proc)
    steps = conn.execute(
        "SELECT * FROM procedure_steps WHERE procedure_id = ? "
        "ORDER BY step_order",
        (proc["id"],)
    ).fetchall()

    result["steps"] = []
    for step_row in steps:
        step = dict(step_row)
        if step_row["total_runs"] > 0:
            step["success_rate"] = round(step_row["successes"] / step_row["total_runs"], 3)
        else:
            step["success_rate"] = None
        result["steps"].append(step)

    if result["total_runs"] > 0:
        result["success_rate"] = round(
            result["successes"] / result["total_runs"], 3
        )
    else:
        result["success_rate"] = None

    runs = conn.execute(
        "SELECT * FROM procedure_runs WHERE procedure_id = ? "
        "ORDER BY started_at DESC LIMIT 5",
        (proc["id"],)
    ).fetchall()
    result["recent_runs"] = [dict(r) for r in runs]

    return result


def find_procedure(conn, agent_id, task_description, limit=3):
    """Find procedures matching a task description (FTS5 search)."""
    fts_query = " OR ".join(re.findall(r"\w{3,}", task_description.lower()))
    if not fts_query:
        return []

    rows = conn.execute("""
        SELECT p.* FROM procedures_fts
        JOIN procedures p ON p.id = procedures_fts.rowid
        WHERE procedures_fts MATCH ? AND p.agent_id = ? AND p.active = 1
        ORDER BY p.successes DESC
        LIMIT ?
    """, (fts_query, agent_id, limit)).fetchall()

    results = []
    for row in rows:
        proc = get_procedure(conn, agent_id, row["name"])
        if proc:
            results.append(proc)

    return results


def anticipate(conn, agent_id, task_description):
    """Predict what the agent will need for a task.

    Returns: best procedure, relevant warnings, past errors to avoid.
    """
    procedures = find_procedure(conn, agent_id, task_description)

    warnings = []
    best_steps = []
    errors_to_avoid = []

    if procedures:
        best = procedures[0]
        for step in best.get("steps", []):
            if step.get("skip_recommended"):
                warnings.append(
                    f"Step '{step['description']}' fails {step['failures']}/{step['total_runs']} times — "
                    f"consider skipping. Last error: {step.get('last_error', 'unknown')}"
                )
            else:
                best_steps.append(step["description"])

            if step.get("last_error"):
                errors_to_avoid.append({
                    "step": step["description"],
                    "error": step["last_error"],
                    "fail_rate": (step["failures"] / step["total_runs"]
                                 if step["total_runs"] > 0 else 0),
                })

    return {
        "procedures": procedures,
        "recommended_steps": best_steps,
        "warnings": warnings,
        "errors_to_avoid": errors_to_avoid,
    }


def list_procedures(conn, agent_id):
    """List all active procedures for an agent."""
    rows = conn.execute(
        "SELECT * FROM procedures WHERE agent_id = ? AND active = 1 "
        "ORDER BY updated_at DESC",
        (agent_id,)
    ).fetchall()
    return [dict(r) for r in rows]
