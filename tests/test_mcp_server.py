"""
The MAUDE MCP server, exercised through a real MCP client connected in memory.
openFDA is replaced by a stand-in, so no network is needed.
"""

import json

import anyio
import pytest

pytest.importorskip("mcp")

from mcp.client.client import Client  # noqa: E402

import agent.tools as agent_tools  # noqa: E402
from mcp_server import maude_server  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code, self._payload = status_code, payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def fake_get(url, params, timeout):
    if "count" in params:
        return FakeResponse(200, {"results": [{"term": "Injury", "count": 12}]})
    return FakeResponse(200, {"results": [{"report_number": "R-9", "date_received": "20261101",
                                            "event_type": "Injury", "device": [{"brand_name": "Pump A"}]}]})


@pytest.fixture
def offline_openfda(monkeypatch):
    original = agent_tools.search_maude

    def patched(term, http_get=None, api_key=None):
        return original(term, http_get=fake_get, api_key=api_key)

    monkeypatch.setattr(maude_server, "search_maude", patched)


def run(coro_fn):
    return anyio.run(coro_fn)


def payload(result) -> dict:
    """The tool's dict as a client receives it: JSON in the first text content block."""
    return json.loads(result.content[0].text)


def test_server_lists_one_read_only_tool():
    async def go():
        async with Client(maude_server.server) as client:
            return await client.list_tools()
    tools = run(go).tools
    assert [t.name for t in tools] == ["search_maude_reports"]
    tool = tools[0]
    assert tool.input_schema["required"] == ["device_term"]
    assert tool.annotations.read_only_hint is True


def test_tool_call_returns_maude_counts(offline_openfda):
    async def go():
        async with Client(maude_server.server) as client:
            return await client.call_tool("search_maude_reports", {"device_term": "insulin pump"})
    result = run(go)
    assert not result.is_error
    data = payload(result)
    assert data["ok"] and data["total_reports"] == 12
    assert data["sample_reports"][0]["report_number"] == "R-9"


def test_bad_input_comes_back_as_a_tool_result_not_a_crash():
    async def go():
        async with Client(maude_server.server) as client:
            return await client.call_tool("search_maude_reports", {"device_term": "!!!"})
    data = payload(run(go))
    assert data["ok"] is False and data["error"] == "empty device term"
