"""Native model profile and actual wait telemetry, without contacting a provider."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock,MagicMock,patch
import pytest
from agent_framework.llm.llm import LLM
from agent_framework.llm.config import LLMConfig
from agent_framework.agents.state import AgentContext

@pytest.mark.asyncio
async def test_job_specific_model_route_seed_and_recorded_wait_are_not_global():
    state=AgentContext(agent_id='native-test-agent',task='audit',context={'capabilities':['audit_receive'],'native_audit':{'kind':'arena-readonly-audit-v1','role':'integration'}},sandbox_info={'job_id':'native-test-job'})
    response=SimpleNamespace(model='reported-model',usage=SimpleNamespace(prompt_tokens=5,completion_tokens=7,cost=.001))
    queue=SimpleNamespace(make_request=AsyncMock(return_value=response))
    redis=MagicMock()
    config=LLMConfig(model_name='openrouter/test-model',api_key='test',seed=10,temperature=.2,openrouter_provider_order=['DeepSeek'],openrouter_allow_fallbacks=False)
    with patch('agent_framework.llm.llm.get_shared_queue',return_value=queue),patch('agent_framework.llm.llm.state_manager.redis_client',redis):
        llm=LLM(config,agent_state=state)
        assert await llm._make_request([{'role':'user','content':'audit'}]) is response
    args=queue.make_request.call_args.args[0]
    assert args['seed']==10 and args['temperature']==.2
    assert args['extra_body']['provider']=={'order':['DeepSeek'],'allow_fallbacks':False}
    records=[json.loads(call.args[1]) for call in redis.rpush.call_args_list]
    assert [e['kind'] for e in records]==['model.request','model.response']
    assert records[0]['request_id']==records[1]['request_id']
    assert records[1]['timestamp']>=records[0]['timestamp'] and records[1]['response_model']=='reported-model'
    assert all('messages' not in record for record in records)

@pytest.mark.asyncio
async def test_native_error_retained_and_legacy_calls_do_not_gain_audit_records():
    queue=SimpleNamespace(make_request=AsyncMock(side_effect=ValueError('provider failed')))
    redis=MagicMock()
    state=AgentContext(agent_id='native-test-agent',task='audit',context={'native_audit':{'role':'events'}},sandbox_info={'job_id':'native-test-job'})
    with patch('agent_framework.llm.llm.get_shared_queue',return_value=queue),patch('agent_framework.llm.llm.state_manager.redis_client',redis):
        native=LLM(LLMConfig(model_name='test-model'),agent_state=state)
        with pytest.raises(ValueError):await native._tracked_request({})
        assert json.loads(redis.rpush.call_args_list[-1].args[1])['kind']=='model.error'
        redis.reset_mock()
        legacy=LLM(LLMConfig(model_name='test-model'))
        with pytest.raises(ValueError):await legacy._tracked_request({})
        redis.rpush.assert_not_called()
