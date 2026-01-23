"""
Tests for AgentContext state management.
"""

from agent_framework.agents.state import AgentContext


class TestAgentContext:
    def test_get_history_filters_internal_roles(self):
        """Test that get_history_for_llm filters out tool_call and tool_result roles."""
        ctx = AgentContext()

        # Add a mix of messages
        ctx.append_message("system", "System prompt")
        ctx.append_message("user", "User query")
        ctx.append_message("assistant", "Thinking...")

        # Add internal tool messages (simulating manual addition or tool recording)
        ctx.messages.append(
            {"role": "tool_call", "content": {"tool": "test"}, "timestamp": "now"}
        )
        ctx.messages.append(
            {"role": "tool_result", "content": {"result": "ok"}, "timestamp": "now"}
        )

        # Add the user observation that usually follows a tool result
        ctx.append_message("user", "Tool Results: ...")

        history = ctx.get_history_for_llm()

        # Verify filtering
        roles = [m["role"] for m in history]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert "tool_call" not in roles
        assert "tool_result" not in roles

        # Verify content integrity
        assert len(history) == 4  # system, user, assistant, user (observation)
