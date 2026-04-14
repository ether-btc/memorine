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
import sys
import time

from . import Mind

_MAX_STRING_LEN = 10000
_MAX_AGENT_ID_LEN = 100


def _read_message():
    header = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        header += line
        if line == "\r\n" or line == "\n":
            break
    length = 0
    for h in header.strip().split("\n"):
        if h.lower().startswith("content-length:"):
            length = int(h.split(":")[1].strip())
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


def _validate_string(val, field, max_len=10000):
    if not val or not isinstance(val, str) or not val.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(val) > max_len:
        raise ValueError(f"{field} exceeds max length ({max_len})")
    return val.strip()


def _validate_int(val, field, min_val=0, max_val=10000):
    val = int(val)
    if val < min_val or val > max_val:
        raise ValueError(f"{field} must be between {min_val} and {max_val}")
    return val


def handle_tool(name, args):
    agent_id = _validate_string(args.get("agent_id", "default"), "agent_id", max_len=100)
    mind = _get_mind(agent_id)

    if name == "memorine_learn":
        fact = _validate_string(args["fact"], "fact", max_len=_MAX_STRING_LEN)
        fid, contras = mind.learn(
            fact,
            category=args.get("category", "general"),
            confidence=args.get("confidence", 1.0),
            relates_to=args.get("relates_to"),
        )
        result = {"fact_id": fid, "contradictions": contras}
        return [{"type": "text", "text": json.dumps(result, default=str)}]


    elif name == "memorine_recall":
        query = _validate_string(args["query"], "query", max_len=_MAX_STRING_LEN)
        facts = mind.recall(
            query, limit=args.get("limit", 5),
            offset=args.get("offset", 0),
        )
        return [{"type": "text", "text": json.dumps(facts, default=str)}]


    elif name == "memorine_log_event":
        tags = None
        if args.get("tags"):
            tags = [t.strip() for t in args["tags"].split(",") if t.strip()]
            tags = tags or None
        eid = mind.log(
            args["event"], tags=tags,
            caused_by=args.get("caused_by"),
        )
        return [{"type": "text", "text": json.dumps({"event_id": eid})}]

    elif name == "memorine_events":
        tags = None
        if args.get("tags"):
            tags = [t.strip() for t in args["tags"].split(",") if t.strip()]
            tags = tags or None
        evts = mind.events(
            query=args.get("query"), tags=tags,
            limit=args.get("limit", 10),
        )
        return [{"type": "text", "text": json.dumps(evts, default=str)}]

    elif name == "memorine_share":
        fact = _validate_string(args["fact"], "fact", max_len=_MAX_STRING_LEN)
        fid = mind.share(
            fact, to_agent=args.get("to_agent"),
        )
        return [{"type": "text", "text": json.dumps({"fact_id": fid})}]


    elif name == "memorine_team_knowledge":
        facts = mind.team_knowledge(limit=args.get("limit", 20))
        return [{"type": "text", "text": json.dumps(facts, default=str)}]

    elif name == "memorine_profile":
        profile = mind.profile()
        return [{"type": "text", "text": profile}]

    elif name == "memorine_anticipate":
        task = _validate_string(args["task"], "task", max_len=_MAX_STRING_LEN)
        result = mind.anticipate(task)
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
        validated_facts = []
        for i, f in enumerate(args["facts"]):
            validated_facts.append({
                "fact": _validate_string(f["fact"], f"facts[{i}].fact", max_len=_MAX_STRING_LEN),
                "category": f.get("category", "general"),
                "confidence": f.get("confidence", 1.0),
            })
        results = mind.learn_batch(validated_facts)
        output = [{"fact_id": fid, "contradictions": c} for fid, c in results]
        return [{"type": "text", "text": json.dumps(output, default=str)}]

    return [{"type": "text", "text": f"Unknown tool: {name}"}]


def main():
    while True:
        msg = _read_message()
        if msg is None:
            break

        method = msg.get("method", "")
        id = msg.get("id")

        if method == "initialize":
            _success(id, {
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
            _success(id, {"tools": TOOLS})

        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            try:
                content = handle_tool(tool_name, tool_args)
                _success(id, {"content": content})
            except ValueError as e:
                _error(id, -32000, str(e))
            except Exception as e:
                _error(id, -32000, f"Internal error: {e}")

        elif id is not None:
            _error(id, -32601, f"Method not found: {method}")


if __name__ == "__main__":
    main()
