from agent_framework.agents.state import AgentContext


class TestAgentContext:
    def test_get_history_filters_internal_roles(self):
        ctx = AgentContext()

        ctx.append_message("system", "System prompt")
        ctx.append_message("user", "User query")
        ctx.append_message("assistant", "Thinking...")

        ctx.messages.append(
            {"role": "tool_call", "content": {"tool": "test"}, "timestamp": "now"}
        )
        ctx.messages.append(
            {"role": "tool_result", "content": {"result": "ok"}, "timestamp": "now"}
        )

        ctx.append_message("user", "Tool Results: ...")

        history = ctx.get_history_for_llm()

        roles = [m["role"] for m in history]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool_call" not in roles
        assert "tool_result" not in roles

        assert len(history) == 4
