from unittest.mock import AsyncMock, patch
from datetime import UTC, datetime, timedelta

import litellm
import pytest

from agent_framework.llm.config import LLMConfig
from agent_framework.llm.llm import LLM
from agent_framework.llm.request_queue import LLMRequestQueue, should_retry_exception
from agent_framework.llm.types import LLMRequestFailedError


@pytest.mark.asyncio
async def test_wrapped_billing_error_is_not_retried_or_hidden(monkeypatch):
    monkeypatch.setenv('AGENT_LLM_RETRY_BASE_DELAY', '0')
    config = LLMConfig(model_name='openrouter/test/model', delay_between_requests=0)
    queue = LLMRequestQueue(config)
    upstream = litellm.InternalServerError(
        message='Provider returned error: {"type":"insufficient_balance_error"}',
        model='test/model', llm_provider='openrouter',
    )
    with patch('agent_framework.llm.request_queue.acompletion', new_callable=AsyncMock) as request, \
         patch('agent_framework.llm.llm.get_shared_queue', return_value=queue):
        request.side_effect = upstream
        llm = LLM(config)
        with pytest.raises(LLMRequestFailedError, match='selected provider route and billing') as caught:
            await llm.generate([{'role':'user', 'content':'test'}])
        request.assert_awaited_once()
        assert 'insufficient_balance_error' not in str(caught.value)


def test_transient_outage_still_retries():
    error = litellm.InternalServerError(
        message='Upstream temporarily unavailable', model='test', llm_provider='openrouter')
    assert should_retry_exception(error)
    assert not should_retry_exception(LLMRequestFailedError('Billing requires attention'))


@pytest.mark.asyncio
async def test_model_failure_is_visible_in_agent_status():
    from agent_framework.agents.base import BaseAgent

    with patch('agent_framework.agents.base.db') as db:
        agent = BaseAgent({'state': {'agent_id': 'billing-test'},
                           'llm_config': {'model_name': 'test-model'}})
        agent.llm.generate = AsyncMock(side_effect=LLMRequestFailedError('Provider billing blocked'))
        assert await agent._execute_cycle() is False
        assert agent.context.llm_failed
        assert agent.context.status == 'llm_failed'
        db.update_agent_status.assert_called_with(
            'billing-test', 'llm_failed', 'Provider billing blocked')


@pytest.mark.asyncio
async def test_billing_failure_does_not_resume_on_the_idle_timer():
    from agent_framework.agents.base import BaseAgent

    with patch('agent_framework.agents.base.db') as db, \
         patch('agent_framework.agents.base.asyncio.sleep', new_callable=AsyncMock):
        agent = BaseAgent({'state': {'agent_id': 'billing-wait'},
                          'llm_config': {'model_name': 'test-model'}})
        agent.context.set_waiting(error_state=True)
        agent.context.waiting_since = datetime.now(UTC) - timedelta(minutes=10)
        db.reset_mock()
        await agent._wait_cycle()
        assert agent.context.llm_failed and agent.context.waiting_for_input
        db.update_agent_status.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_request_is_limited_to_authorized_providers(monkeypatch):
    monkeypatch.setenv('OPENROUTER_PROVIDER_ORDER', 'deepseek,deepinfra')
    monkeypatch.setenv('OPENROUTER_PROVIDER_ONLY', 'deepseek,deepinfra')
    monkeypatch.setenv('OPENROUTER_ALLOW_FALLBACKS', 'true')
    queue = AsyncMock()
    with patch('agent_framework.llm.llm.get_shared_queue', return_value=queue):
        llm = LLM(LLMConfig(model_name='openrouter/test/model'))
        await llm._make_request([{'role': 'user', 'content': 'test'}])
    assert queue.make_request.call_args.args[0]['extra_body']['provider'] == {
        'order': ['deepseek', 'deepinfra'], 'only': ['deepseek', 'deepinfra'],
        'allow_fallbacks': True,
    }
