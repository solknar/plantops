"""PlantOps MCP server: exposes the customer's warehouse and document corpus as MCP tools.

This is the integration layer between the customer's systems and any MCP-capable agent or IDE
(ADK, Claude Desktop, Gemini CLI, ...). The guards (read-only SQL, PII redaction, filter
whitelisting) live in the tool functions, so every client gets the same security boundary.

    python -m plantops.mcp_server                 # stdio (what ADK's McpToolset launches)
    python -m plantops.mcp_server --http 8765     # streamable HTTP
"""

from __future__ import annotations

import argparse

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from plantops.tools import functions as fn

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def build_server() -> MCPServer:
    server = MCPServer(
        name="plantops",
        instructions="Read-only access to Northwind Fabrication's plant warehouse and maintenance document corpus.",
    )
    for f in fn.ALL_TOOLS:
        server.add_tool(f, annotations=READ_ONLY)
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, metavar="PORT", help="serve streamable HTTP on this port instead of stdio")
    args = ap.parse_args()
    server = build_server()
    if args.http:
        server.run("streamable-http", port=args.http)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
