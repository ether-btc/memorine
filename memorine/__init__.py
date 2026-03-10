"""
Memorine — Human-like memory for AI agents.
Zero dependencies. Zero tokens. Zero cost. Just memory that works.

    from memorine import Mind

    brain = Mind("marc")
    brain.learn("abuse@hostkey.nl is the contact for HOSTKEY")
    brain.recall("hostkey")  # finds it + reinforces the memory
"""

__version__ = "0.2.1"

import logging
import time
from typing import Any, Optional

from .constants import PROFILE_WEIGHT_FLOOR
from .db import get_connection, init_schema, get_stats
from . import cortex, hippocampus, cerebellum, amygdala, synapses

logger = logging.getLogger(__name__)


class Mind:
    """A single agent's mind. Each agent gets their own Mind instance.

    Usage:
        brain = Mind("marc")
        brain.learn("some fact")
        brain.recall("search query")
    """

    def __init__(self, agent_id: str, db_path: Optional[str] = None):
        self.agent_id = agent_id
        self.conn = get_connection(db_path)
        init_schema(self.conn)

    # -- CORTEX: Facts ----------------------------------------------------

    def learn(self, fact: str, category: str = "general", confidence: float = 1.0,
              source: Optional[str] = None, weight: Optional[float] = None,
              relates_to: Optional[str] = None) -> tuple[int, list[dict]]:
        """Store a fact. Returns (fact_id, contradictions).

        Automatically detects contradictions with existing knowledge.
        Near-duplicates are reinforced instead of duplicated.
        """
        return cortex.learn(
            self.conn, self.agent_id, fact,
            category=category, confidence=confidence,
            source=source, weight=weight, relates_to=relates_to
        )

    def learn_batch(self, facts: list[dict]) -> list[tuple[int, list[dict]]]:
        """Batch-learn multiple facts at once. Much faster for bulk imports.

        facts: list of dicts with keys: fact, category, confidence, source, weight
        Returns: list of (fact_id, contradictions) tuples.
        """
        return cortex.learn_batch(self.conn, self.agent_id, facts)

    def recall(self, query: str, limit: int = 5, offset: int = 0,
               include_shared: bool = True) -> list[dict]:
        """Search memory for facts matching a query.

        Uses semantic search when embeddings are available, with FTS5
        as a complement and fallback. Results ranked by a blend of
        semantic similarity and effective weight.
        """
        return cortex.recall(
            self.conn, self.agent_id, query,
            limit=limit, offset=offset, include_shared=include_shared
        )

    def forget(self, fact_id: int) -> None:
        """Soft-delete a fact. Only works on facts owned by this agent."""
        cortex.forget(self.conn, fact_id, self.agent_id)

    def correct(self, fact_id: int, new_value: str,
                confidence: Optional[float] = None) -> None:
        """Update a fact that turned out to be wrong."""
        cortex.update_fact(self.conn, fact_id, new_value, self.agent_id, confidence)

    def connect(self, fact_a: int, fact_b: int, relation: str = "related") -> None:
        """Create an association between two facts."""
        cortex.link(self.conn, fact_a, fact_b, relation, agent_id=self.agent_id)

    def associations(self, fact_id: int, depth: int = 1) -> list[dict]:
        """Get facts associated with a given fact."""
        return cortex.associations(self.conn, fact_id, depth)

    def facts(self, limit: Optional[int] = None, offset: int = 0) -> list[dict]:
        """List all active facts."""
        return cortex.all_facts(self.conn, self.agent_id, limit=limit, offset=offset)

    # -- HIPPOCAMPUS: Events ----------------------------------------------

    def log(self, event: str, context: Any = None,
            tags: Optional[list[str]] = None,
            caused_by: Optional[int] = None) -> int:
        """Record something that happened.

        Use caused_by to build causal chains:
            e1 = brain.log("DNS timeout on domain X")
            e2 = brain.log("Scan failed", caused_by=e1)
        """
        return hippocampus.log_event(
            self.conn, self.agent_id, event,
            context=context, tags=tags, caused_by=caused_by
        )

    def events(self, query: Optional[str] = None,
               since: Optional[float] = None, until: Optional[float] = None,
               tags: Optional[list[str]] = None,
               limit: int = 20, offset: int = 0) -> list[dict]:
        """Search past events by text, time range, or tags."""
        return hippocampus.recall_events(
            self.conn, self.agent_id, query=query,
            since=since, until=until, tags=tags, limit=limit, offset=offset
        )

    def why(self, event_id: int) -> list[dict]:
        """Trace the causal chain: what caused this event?"""
        return hippocampus.causal_chain(self.conn, event_id, "up")

    def consequences(self, event_id: int) -> list[dict]:
        """What did this event cause?"""
        return hippocampus.causal_chain(self.conn, event_id, "down")

    def timeline(self, since: Optional[float] = None,
                 until: Optional[float] = None, limit: int = 50) -> list[dict]:
        """Get chronological event timeline."""
        return hippocampus.timeline(
            self.conn, self.agent_id, since=since, until=until, limit=limit
        )

    # -- CEREBELLUM: Procedures -------------------------------------------

    def procedure(self, name: str, description: Optional[str] = None,
                  steps: Optional[list[str]] = None) -> "ProcedureRun":
        """Get or create a procedure. Returns a ProcedureRun context manager.

        Usage:
            with brain.procedure("scan_site") as run:
                run.step("detect_cdn", success=True)
                run.step("probe_subdomains", success=False, error="timeout")
        """
        proc = cerebellum.get_procedure(self.conn, self.agent_id, name)
        if not proc:
            proc_id = cerebellum.create_procedure(
                self.conn, self.agent_id, name, description, steps
            )
        else:
            proc_id = proc["id"]
        return ProcedureRun(self.conn, proc_id)

    def anticipate(self, task_description: str) -> dict:
        """Predict what you'll need for a task.

        Returns best procedure, recommended steps, warnings about
        steps that often fail, and past errors to avoid.
        """
        return cerebellum.anticipate(
            self.conn, self.agent_id, task_description
        )

    def procedures(self) -> list[dict]:
        """List all active procedures."""
        return cerebellum.list_procedures(self.conn, self.agent_id)

    # -- AMYGDALA: Maintenance --------------------------------------------

    def cleanup(self, threshold: float = 0.05) -> int:
        """Deactivate faded memories below threshold."""
        return amygdala.cleanup_faded(self.conn, self.agent_id, threshold)

    def stats(self) -> dict:
        """Database statistics: fact counts, events, procedures, db size."""
        return get_stats(self.conn, self.agent_id)

    def reindex_embeddings(self) -> int:
        """Rebuild all embeddings. Run after installing memorine[embeddings]."""
        try:
            from . import embeddings
            if embeddings.is_available():
                return embeddings.reindex_all(self.conn, self.agent_id)
        except ImportError:
            pass
        except Exception:
            logger.warning("Failed to reindex embeddings", exc_info=True)
        return 0

    # -- SYNAPSES: Sharing ------------------------------------------------

    def share(self, fact_text: str, to_agent: Optional[str] = None,
              category: str = "shared") -> int:
        """Learn a fact and share it with another agent (or everyone)."""
        return synapses.share_fact(
            self.conn, self.agent_id, fact_text,
            to_agent=to_agent, category=category
        )

    def shared_with_me(self, limit: int = 20) -> list[dict]:
        """Get facts other agents have shared with me."""
        return synapses.shared_with_me(self.conn, self.agent_id, limit)

    def team_knowledge(self, category: Optional[str] = None,
                       limit: int = 50) -> list[dict]:
        """Get collective team knowledge."""
        return synapses.team_knowledge(self.conn, category, limit)

    # -- PROFILE ----------------------------------------------------------

    def profile(self, max_facts: int = 20, max_events: int = 10) -> str:
        """Generate a cognitive profile — a summary of what this agent knows.

        Returns a plain text block ready to inject into a system prompt.
        """
        now = time.time()
        lines = [f"# Memory Profile: {self.agent_id}"]

        # Top facts by effective weight
        all_facts = cortex.all_facts(self.conn, self.agent_id)
        weighted = []
        for fact in all_facts:
            weight = amygdala.effective_weight(fact, now)
            if weight > PROFILE_WEIGHT_FLOOR:
                weighted.append((weight, fact))
        weighted.sort(key=lambda x: x[0], reverse=True)

        if weighted:
            lines.append("\n## Key Knowledge")
            for _weight, fact in weighted[:max_facts]:
                lines.append(f"- {fact['fact']} [{fact['category']}]")

        # Shared knowledge
        shared = synapses.shared_with_me(self.conn, self.agent_id, limit=10)
        if shared:
            lines.append("\n## Team Knowledge")
            for shared_fact in shared:
                lines.append(f"- {shared_fact['fact']} (from {shared_fact['from_agent']})")

        # Recent events
        recent = hippocampus.timeline(self.conn, self.agent_id, limit=max_events)
        if recent:
            lines.append("\n## Recent Activity")
            for event in recent:
                lines.append(f"- {event['event']}")

        # Active procedures
        procs = cerebellum.list_procedures(self.conn, self.agent_id)
        if procs:
            lines.append("\n## Known Procedures")
            for proc in procs:
                rate = ""
                if proc["total_runs"] > 0:
                    success_pct = round(proc["successes"] / proc["total_runs"] * 100)
                    rate = f" ({success_pct}% success, {proc['total_runs']} runs)"
                lines.append(f"- {proc['name']}{rate}")

        return "\n".join(lines)


class ProcedureRun:
    """Context manager for tracking a procedure execution."""

    def __init__(self, conn, procedure_id):
        self.conn = conn
        self.procedure_id = procedure_id
        self.run_id = None
        self._step_count = 0
        self._success = True
        self._error = None

    def __enter__(self):
        self.run_id = cerebellum.start_run(self.conn, self.procedure_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._success = False
            self._error = str(exc_val)
        cerebellum.complete_run(
            self.conn, self.run_id, self._success, self._error
        )
        return False

    def step(self, description, success=True, error=None, duration_ms=None):
        """Record a step result."""
        self._step_count += 1
        if not success:
            self._success = False
            self._error = error
        cerebellum.log_step(
            self.conn, self.run_id, self._step_count, description,
            success=success, error=error, duration_ms=duration_ms
        )
