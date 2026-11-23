"""
MCP server that exposes the agent's FDA MAUDE lookup as a tool any MCP client can call
(Claude Desktop, Claude Code, an IDE, or another agent).

Tool: search_maude_reports(device_term) -> MAUDE report counts by event type for a device name,
plus the three most recent reports. It is the same function the LangGraph agent uses
(agent/tools.py: search_maude), so both paths return identical results.

Run over stdio (how desktop clients start it):
    python mcp_server/maude_server.py

The server reads OPENFDA_API_KEY from the environment if set; without it openFDA allows
240 requests a minute. Nothing is written to stdout except protocol messages.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # clients start the server from any working directory

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from agent.tools import search_maude  # noqa: E402

server = MCPServer(
    name="regulatory-copilot-maude",
    instructions=(
        "Looks up US FDA MAUDE adverse event reports for a medical device type. "
        "Pass the generic device name (for example 'infusion pump' or 'implantable defibrillator'). "
        "MAUDE is a voluntary US reporting database: counts are a signal, not an incidence rate."
    ),
)


@server.tool(
    name="search_maude_reports",
    description=(
        "Count FDA MAUDE adverse event reports for a device, by event type (Death, Injury, Malfunction, Other), "
        "and list the three most recently received reports. Matches the device's generic or brand name."
    ),
    annotations=ToolAnnotations(title="Search FDA MAUDE reports", readOnlyHint=True, openWorldHint=True),
)
def search_maude_reports(device_term: str) -> dict:
    """device_term: generic or brand device name, e.g. 'insulin pump'."""
    return search_maude(device_term)


if __name__ == "__main__":
    server.run("stdio")
