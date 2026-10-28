"""ADK toolset that reaches the PlantOps tools through the MCP server over stdio."""

from __future__ import annotations

import os
import sys

from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from plantops.config import ROOT


def plantops_toolset(tool_names: list[str]) -> McpToolset:
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=["-m", "plantops.mcp_server"],
                cwd=str(ROOT),
                env={**os.environ},
            ),
            timeout=30,
        ),
        tool_filter=tool_names,
    )
