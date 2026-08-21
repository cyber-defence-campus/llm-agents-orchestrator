import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework.tools.terminal import actions


class TestExecTimeoutCap(unittest.TestCase):
    def _run(self, **kwargs):
        tm = MagicMock()
        tm.execute_command = AsyncMock(return_value={"status": "completed"})
        with patch.object(actions, "get_terminal_manager", return_value=tm):
            result = asyncio.run(actions.run_shell_command(**kwargs))
        return tm, result

    def test_oversized_timeout_is_capped_and_reported(self):
        tm, result = self._run(command="# check", exec_timeout=15000)
        self.assertEqual(
            tm.execute_command.call_args.kwargs["timeout"], actions.MAX_EXEC_TIMEOUT
        )
        self.assertIn("SECONDS", result["timeout_note"])

    def test_normal_timeout_passes_through_untouched(self):
        tm, result = self._run(command="ls", exec_timeout=120)
        self.assertEqual(tm.execute_command.call_args.kwargs["timeout"], 120)
        self.assertNotIn("timeout_note", result)

    def test_absent_timeout_stays_none(self):
        tm, result = self._run(command="ls")
        self.assertIsNone(tm.execute_command.call_args.kwargs["timeout"])
        self.assertNotIn("timeout_note", result)


if __name__ == "__main__":
    unittest.main()
