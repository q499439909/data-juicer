import json
import unittest
from unittest.mock import MagicMock, patch

import anyio
from pydantic import BaseModel, Field

from data_juicer.agent_tools.core.tool import ToolContext, ToolResult, ToolSpec
from data_juicer.agent_tools.mcp.server import (
    build_mcp_server,
    build_mcp_tool_function,
    build_tool_signature,
    invoke_tool_spec,
    select_tool_specs,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

SELECT_TARGET = "data_juicer.agent_tools.mcp.server.list_tool_specs"


class EchoInput(BaseModel):
    text: str = Field(description="Text to echo back.")
    repeat: int = Field(default=1, ge=1, le=5, description="Repeat count.")


def _echo(_ctx, args: EchoInput) -> ToolResult:
    return ToolResult.success(summary="echoed", data={"echo": args.text * args.repeat})


def _boom(_ctx, _args: EchoInput) -> ToolResult:
    raise RuntimeError("kaboom")


ECHO_SPEC = ToolSpec(
    name="echo_tool",
    description="Echo the given text.",
    input_model=EchoInput,
    output_model=None,
    executor=_echo,
    tags=("test",),
    effects="read",
)

WRITE_SPEC = ToolSpec(
    name="write_tool",
    description="Pretend to write something.",
    input_model=EchoInput,
    output_model=None,
    executor=_echo,
    tags=("test",),
    effects="write",
    confirmation="required",
)


def _ctx_factory() -> ToolContext:
    return ToolContext(working_dir="./.djx")


def _build_server(specs, **kwargs):
    return build_mcp_server(name="test-server", **kwargs)


def _list_registered(mcp):
    async def _run():
        tools = await mcp.list_tools()
        return [json.loads(tool.model_dump_json()) for tool in tools]

    return anyio.run(_run)


def _call(mcp, name, arguments):
    async def _run():
        result = await mcp.call_tool(name, arguments)
        content = result[0] if isinstance(result, tuple) else result
        return json.loads(content[0].text)

    return anyio.run(_run)


class AgentToolsAdapterTest(DataJuicerTestCaseBase):
    """Tests for converting tool specs into MCP tools."""

    def test_signature_flattens_input_model(self):
        signature = build_tool_signature(ECHO_SPEC)
        self.assertEqual(list(signature.parameters), ["text", "repeat"])
        self.assertIs(signature.parameters["text"].default, signature.empty)
        self.assertEqual(signature.parameters["repeat"].default, 1)

    def test_tool_function_executes_spec(self):
        func = build_mcp_tool_function(ECHO_SPEC, ctx_factory=_ctx_factory)
        payload = func(text="ab", repeat=2)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["echo"], "abab")
        self.assertEqual(payload["action"], "echo_tool")

    def test_invalid_arguments_return_error_payload(self):
        func = build_mcp_tool_function(ECHO_SPEC, ctx_factory=_ctx_factory)
        payload = func(text="ab", repeat=99)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "invalid_arguments")
        self.assertTrue(payload["validation_errors"])

    def test_executor_exception_is_reported(self):
        spec = ToolSpec(
            name="boom_tool",
            description="always fails",
            input_model=EchoInput,
            output_model=None,
            executor=_boom,
        )
        payload = invoke_tool_spec(spec, ctx=_ctx_factory(), raw_kwargs={"text": "x"})
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "tool_exception")
        self.assertIn("kaboom", payload["error_message"])

    def test_server_publishes_tool_schema(self):
        with patch(SELECT_TARGET, lambda tags=None, profile=None: [ECHO_SPEC, WRITE_SPEC]):
            mcp = _build_server([ECHO_SPEC, WRITE_SPEC])
        tools = {tool["name"]: tool for tool in _list_registered(mcp)}

        self.assertEqual(set(tools), {"echo_tool", "write_tool"})
        schema = tools["echo_tool"]["inputSchema"]
        self.assertEqual(schema["required"], ["text"])
        self.assertEqual(schema["properties"]["repeat"]["maximum"], 5)
        self.assertEqual(schema["properties"]["text"]["description"], "Text to echo back.")
        self.assertTrue(tools["echo_tool"]["annotations"]["readOnlyHint"])
        self.assertTrue(tools["write_tool"]["annotations"]["destructiveHint"])
        self.assertIn("confirmation=required", tools["write_tool"]["description"])

    def test_server_executes_tool(self):
        with patch(SELECT_TARGET, lambda tags=None, profile=None: [ECHO_SPEC]):
            mcp = _build_server([ECHO_SPEC])
        payload = _call(mcp, "echo_tool", {"text": "hi", "repeat": 2})
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["echo"], "hihi")

    def test_selection_filters(self):
        with patch(SELECT_TARGET, lambda tags=None, profile=None: [ECHO_SPEC, WRITE_SPEC]):
            self.assertEqual([spec.name for spec in select_tool_specs(include=["echo_tool"])], ["echo_tool"])
            self.assertEqual([spec.name for spec in select_tool_specs(exclude=["echo_tool"])], ["write_tool"])
            with self.assertRaises(KeyError):
                select_tool_specs(include=["missing_tool"])


class AgentToolsServerModeTest(DataJuicerTestCaseBase):
    """Tests for the built-in tool registry and the dj-mcp entry point."""

    def test_builtin_registry_is_exposed(self):
        mcp = build_mcp_server()
        names = {tool["name"] for tool in _list_registered(mcp)}
        self.assertIn("inspect_dataset", names)
        self.assertIn("retrieve_operators", names)
        self.assertIn("apply_recipe", names)

    def test_agent_tools_mode_launches(self):
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "agent-tools", "--transport", "stdio"]):
            with patch("data_juicer.tools.DJ_mcp_agent_tools.create_mcp_server") as mock_create:
                mock_server = MagicMock()
                mock_create.return_value = mock_server
                main()
                mock_create.assert_called_once_with(port="8080")
                mock_server.run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    unittest.main()
