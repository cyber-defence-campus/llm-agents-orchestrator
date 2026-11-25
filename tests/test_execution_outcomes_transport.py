"""Explicit executor failures remain failures beneath a successful transport ACK."""
import pytest
from agent_framework.tools.outcomes import execution_failed


@pytest.mark.parametrize('result', [
    {'result': {'status': 'execution_failed', 'reason': 'fixture executor failure'}},
    {'success': True, 'data': {'status': 'timeout'}},
    {'result': {'status': 'transport_timeout'}},
    {'isError': True, 'content': [{'type': 'text', 'text': 'fixture failure'}]},
    {'result': {'is_error': True}},
])
def test_explicit_executor_failure_survives_transport_ack(result):
    assert execution_failed(result)


def test_pending_and_nonerror_mcp_flags_are_successful_acknowledgments():
    assert not execution_failed({'isError': False, 'result': {'status': 'running', 'pending': True}})
