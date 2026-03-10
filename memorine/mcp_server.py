#!/usr/bin/env python3
"""
Memorine MCP Server — exposes memory tools to OpenClaw agents.

Add to openclaw.json:
    "mcpServers": {
        "memorine": {
            "command": "python3",
            "args": ["/path/to/memorine/memorine/mcp_server.py"],
            "env": {}
        }
    }
"""

import json
import logging
import sys

from . import Mind

logger = logging.getLogger(__name__)

_MAX_STRING_LEN = 10000
_MAX_AGENT_ID_LEN = 100


def _read_message():
    """Read a JSON-RPC message from stdin (Content-Length framing)."""
    header = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        header += line
        if line == "\r\n" or line == "\n":
            break
    length = 0
    for header_line in header.strip().split("\n"):
        if header_line.lower().startswith("content-length:"):
            length = int(header_line.split(":")[1].strip())
    if length == 0:
        return None
    body = sys.stdin.read(length)
    return json.loads(body)


def _send_message(msg):
    body = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(body)}\r\n\r\n{body}")
    sys.stdout.flush()


def _success(id, result):
    _send_message({"jsonrpc": "2.0", "id": id, "result": result})


def _error(id, code, message):
    _send_message({
        "jsonrpc": "2.0", "id": id,
        "error": {"code": code, "message": message}
    })


TOOLS = [
    {
        "name": "memorine_learn",
        "description": "Store a new fact in memory. Automatically detects contradictions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent name"},
                "fact": {"type": "string", "description": "The fact to remember"},
                "category": {"type": "string", "default": "general"},
                "confidence": {"type": "number", "default": 1.0},
                "relates_to": {"type": "string", "description": "Text of a related fact to link to"},
            },
            "required": ["agent_id", "fact"]
        }
    },
    {
        "name": "memorine_recall",
        "description": "Search memory for facts matching a query. Results ranked by importance and recency.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["agent_id", "query"]
        }
    },
    {
        "name": "memorine_log_event",
        "description": "Record an event that happened. Supports causal chains.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "event": {"type": "string"},
                "tags": {"type": "string", "description": "Comma-separated tags"},
                "caused_by": {"type": "integer", "description": "Event ID that caused this"},
            },
            "required": ["agent_id", "event"]
        }
    },
    {
        "name": "memorine_events",
        "description": "Search past events by text or tags.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "query": {"type": "string"},
                "tags": {"type": "string", "description": "Comma-separated tags"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["agent_id"]
        }
    },
    {
        "name": "memorine_share",
        "description": "Share a fact with another agent or the whole team.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Your agent name"},
                "fact": {"type": "string", "description": "Fact to share"},
                "to_agent": {"type": "string", "description": "Target agent (empty = share with everyone)"},
            },
            "required": ["agent_id", "fact"]
        }
    },
    {
        "name": "memorine_team_knowledge",
        "description": "Get collective knowledge shared across the team.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["agent_id"]
        }
    },
    {
        "name": "memorine_profile",
        "description": "Get the cognitive profile — a summary of everything this agent knows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
            },
            "required": ["agent_id"]
        }
    },
    {
        "name": "memorine_anticipate",
        "description": "Predict what you'll need for a task. Returns best procedure, warnings, and past errors.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "task": {"type": "string", "description": "What you're about to do"},
            },
            "required": ["agent_id", "task"]
        }
    },
    {
        "name": "memorine_procedure_start",
        "description": "Start tracking a procedure execution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "name": {"type": "string", "description": "Procedure name"},
                "description": {"type": "string"},
            },
            "required": ["agent_id", "name"]
        }
    },
    {
        "name": "memorine_procedure_step",
        "description": "Log a step result in a running procedure.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "integer"},
                "step": {"type": "string", "description": "Step description"},
                "success": {"type": "boolean", "default": True},
                "error": {"type": "string"},
            },
            "required": ["run_id", "step"]
        }
    },
    {
        "name": "memorine_procedure_complete",
        "description": "Mark a procedure run as complete.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "integer"},
                "success": {"type": "boolean", "default": True},
                "error": {"type": "string"},
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "memorine_correct",
        "description": "Correct a fact that turned out to be wrong.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "fact_id": {"type": "integer"},
                "new_value": {"type": "string"},
            },
            "required": ["agent_id", "fact_id", "new_value"]
        }
    },
    {
        "name": "memorine_stats",
        "description": "Get database statistics: fact counts, events, procedures, db size, embedding status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
            },
            "required": ["agent_id"]
        }
    },
    {
        "name": "memorine_learn_batch",
        "description": "Batch-learn multiple facts at once. Much faster than learning one by one.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string"},
                            "category": {"type": "string", "default": "general"},
                            "confidence": {"type": "number", "default": 1.0},
                        },
                        "required": ["fact"]
                    },
                    "description": "List of facts to learn"
                },
            },
            "required": ["agent_id", "facts"]
        }
    },
]

_minds = {}


def _get_mind(agent_id):
    if agent_id not in _minds:
        _minds[agent_id] = Mind(agent_id)
    return _minds[agent_id]


def _validate_string(val, field, max_len=_MAX_STRING_LEN):
    if not val or not isinstance(val, str) or not val.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(val) > max_len:
        raise ValueError(f"{field} exceeds max length ({max_len})")
    return val.strip()


def _parse_tags(raw):
    """Parse comma-separated tags into a list, or None."""
    if not raw:
        return None
    parsed = [t.strip() for t in raw.split(",") if t.strip()]
    return parsed or None


def handle_tool(name, args):
    """Route a tool call to the appropriate Mind method."""
    agent_id = _validate_string(
        args.get("agent_id", "default"), "agent_id",
        max_len=_MAX_AGENT_ID_LEN
    )
    mind = _get_mind(agent_id)

    if name == "memorine_learn":
        fact_id, contradictions = mind.learn(
            args["fact"],
            category=args.get("category", "general"),
            confidence=args.get("confidence", 1.0),
            relates_to=args.get("relates_to"),
        )
        result = {"fact_id": fact_id, "contradictions": contradictions}
        return [{"type": "text", "text": json.dumps(result, default=str)}]

    elif name == "memorine_recall":
        facts = mind.recall(
            args["query"], limit=args.get("limit", 5),
            offset=args.get("offset", 0),
        )
        return [{"type": "text", "text": json.dumps(facts, default=str)}]

    elif name == "memorine_log_event":
        tags = _parse_tags(args.get("tags"))
        event_id = mind.log(
            args["event"], tags=tags,
            caused_by=args.get("caused_by"),
        )
        return [{"type": "text", "text": json.dumps({"event_id": event_id})}]

    elif name == "memorine_events":
        tags = _parse_tags(args.get("tags"))
        event_list = mind.events(
            query=args.get("query"), tags=tags,
            limit=args.get("limit", 10),
        )
        return [{"type": "text", "text": json.dumps(event_list, default=str)}]

    elif name == "memorine_share":
        fact_id = mind.share(
            args["fact"], to_agent=args.get("to_agent"),
        )
        return [{"type": "text", "text": json.dumps({"fact_id": fact_id})}]

    elif name == "memorine_team_knowledge":
        facts = mind.team_knowledge(limit=args.get("limit", 20))
        return [{"type": "text", "text": json.dumps(facts, default=str)}]

    elif name == "memorine_profile":
        profile = mind.profile()
        return [{"type": "text", "text": profile}]

    elif name == "memorine_anticipate":
        result = mind.anticipate(args["task"])
        return [{"type": "text", "text": json.dumps(result, default=str)}]

    elif name == "memorine_procedure_start":
        from . import cerebellum
        proc = cerebellum.get_procedure(mind.conn, agent_id, args["name"])
        if not proc:
            pid = cerebellum.create_procedure(
                mind.conn, agent_id, args["name"], args.get("description")
            )
        else:
            pid = proc["id"]
        run_id = cerebellum.start_run(mind.conn, pid)
        return [{"type": "text", "text": json.dumps({"run_id": run_id})}]

    elif name == "memorine_procedure_step":
        from . import cerebellum
        cerebellum.log_step(
            mind.conn, args["run_id"], 0, args["step"],
            success=args.get("success", True),
            error=args.get("error"),
            agent_id=agent_id,
        )
        return [{"type": "text", "text": "ok"}]

    elif name == "memorine_procedure_complete":
        from . import cerebellum
        cerebellum.complete_run(
            mind.conn, args["run_id"],
            success=args.get("success", True),
            error=args.get("error"),
            agent_id=agent_id,
        )
        return [{"type": "text", "text": "ok"}]

    elif name == "memorine_correct":
        mind.correct(args["fact_id"], args["new_value"])
        return [{"type": "text", "text": "corrected"}]

    elif name == "memorine_stats":
        stats = mind.stats()
        return [{"type": "text", "text": json.dumps(stats, default=str)}]

    elif name == "memorine_learn_batch":
        results = mind.learn_batch(args["facts"])
        output = [{"fact_id": fid, "contradictions": c} for fid, c in results]
        return [{"type": "text", "text": json.dumps(output, default=str)}]

    return [{"type": "text", "text": f"Unknown tool: {name}"}]


def main():
    """MCP server main loop."""
    while True:
        msg = _read_message()
        if msg is None:
            break

        method = msg.get("method", "")
        request_id = msg.get("id")

        if method == "initialize":
            _success(request_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "memorine",
                    "version": __import__("memorine").__version__,
                },
            })

        elif method == "notifications/initialized":
            pass

        elif method == "tools/list":
            _success(request_id, {"tools": TOOLS})

        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            try:
                content = handle_tool(tool_name, tool_args)
                _success(request_id, {"content": content})
            except Exception as e:
                logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
                _success(request_id, {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                })

        elif request_id is not None:
            _error(request_id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
