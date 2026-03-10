"""
Memorine amygdala — emotional weight, decay, and reinforcement.
Memories that matter stick. The rest fades away.
"""

import logging
import math
import time
from typing import Any, Optional

from .constants import (
    CLEANUP_BATCH_SIZE,
    CLEANUP_THRESHOLD,
    ERROR_IMPORTANCE_MULTIPLIER,
    MAX_STABILITY_ACCESS_COUNT,
    MIN_RETENTION,
    REINFORCE_BOOST,
    REINFORCE_WEIGHT_CAP,
    SECONDS_PER_DAY,
    STABILITY_PER_ACCESS,
    WEAKEN_PENALTY,
    WEIGHT_MIN,
)

logger = logging.getLogger(__name__)


def decay_factor(last_accessed: float, access_count: int, now: Optional[float] = None) -> float:
    """Ebbinghaus forgetting curve with reinforcement.

    More access = more stable memory.
    Untouched memories fade over days.
    Heavily used memories last months.
    """
    now = now or time.time()
    elapsed_days = max((now - last_accessed) / SECONDS_PER_DAY, 0)
    stability = 1 + min(access_count, MAX_STABILITY_ACCESS_COUNT) * STABILITY_PER_ACCESS
    retention = math.exp(-elapsed_days / stability)
    return round(max(retention, MIN_RETENTION), 4)


def effective_weight(fact_row: dict, now: Optional[float] = None) -> float:
    """Combine base weight, confidence, and decay into a single score."""
    decay = decay_factor(fact_row["last_accessed"], fact_row["access_count"], now)
    return round(fact_row["weight"] * fact_row["confidence"] * decay, 4)


def importance_from_error(is_error: bool) -> float:
    """Errors get high emotional weight — pain sticks."""
    return ERROR_IMPORTANCE_MULTIPLIER if is_error else 1.0


def reinforce(conn: Any, fact_id: int, boost: float = REINFORCE_BOOST) -> None:
    """Accessing a memory reinforces it — like rehearsal."""
    now = time.time()
    conn.execute("""
        UPDATE facts SET
            last_accessed = ?,
            access_count = access_count + 1,
            weight = MIN(weight + ?, ?)
        WHERE id = ?
    """, (now, boost, REINFORCE_WEIGHT_CAP, fact_id))
    conn.commit()


def weaken(conn: Any, fact_id: int, penalty: float = WEAKEN_PENALTY) -> None:
    """Contradicted or wrong memories get weakened."""
    conn.execute("""
        UPDATE facts SET
            weight = MAX(weight - ?, ?),
            confidence = MAX(confidence - ?, ?)
        WHERE id = ?
    """, (penalty, WEIGHT_MIN, penalty, WEIGHT_MIN, fact_id))
    conn.commit()


def cleanup_faded(
    conn: Any,
    agent_id: Optional[str] = None,
    threshold: float = CLEANUP_THRESHOLD,
    batch_size: int = CLEANUP_BATCH_SIZE,
) -> int:
    """Deactivate memories that have faded below threshold.

    Processes in batches to avoid loading the entire table.
    """
    now = time.time()
    deactivated = 0
    offset = 0

    while True:
        sql = ("SELECT id, last_accessed, access_count, weight, confidence "
               "FROM facts WHERE active = 1")
        params = []
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        sql += " LIMIT ? OFFSET ?"
        params.extend([batch_size, offset])

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            break

        batch_count = 0
        for row in rows:
            if effective_weight(row, now) < threshold:
                conn.execute(
                    "UPDATE facts SET active = 0 WHERE id = ?", (row["id"],)
                )
                batch_count += 1

        if batch_count:
            conn.commit()
            logger.debug("Deactivated %d faded memories in batch", batch_count)
        deactivated += batch_count

        if len(rows) < batch_size:
            break
        offset += batch_size

    return deactivated
