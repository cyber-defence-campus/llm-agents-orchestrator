import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import redis
from agent_framework.tools import operation_limits, executor


@pytest.fixture
def operation(monkeypatch):
    client=redis.Redis(host='localhost',port=26379,decode_responses=True)
    client.ping()
    job='test-budget-'+uuid.uuid4().hex
    key=f'platform:job_config:{job}'
    config={'status':'running','execution_limits':{'max_actions':3,'deadline':time.time()+60}}
    client.set(key,json.dumps(config))
    monkeypatch.setattr(operation_limits.redis_manager,'redis_client',client)
    state=SimpleNamespace(context_data={'bounded_operation':True},sandbox_info={'job_id':job})
    yield state,client,key,config
    client.delete(key,f'platform:operation_actions:{job}')


def test_atomic_budget_is_shared_across_concurrent_agents(operation):
    state,_,_,_=operation
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:operation_limits.claim_action(state,'run_shell_command'),range(30)))
    assert sum(results)==3
    assert operation_limits.claim_action(state,'complete_assignment')


@pytest.mark.parametrize('change',[{'status':'paused'},{'status':'stopped'},{'execution_limits':{'deadline':0,'max_actions':3}},{}])
def test_closed_or_missing_configuration_never_dispatches(operation,change):
    state,client,key,config=operation
    if change:client.set(key,json.dumps({**config,**change}))
    else:client.delete(key)
    assert not operation_limits.claim_action(state,'run_shell_command')


@pytest.mark.asyncio
async def test_executor_refuses_before_calling_a_tool(operation,monkeypatch):
    state,client,key,config=operation
    client.set(key,json.dumps({**config,'status':'stopped'}))
    dispatch=AsyncMock()
    monkeypatch.setattr(executor,'_dispatch_tool',dispatch)
    monkeypatch.setattr(executor,'_tool_contract_error',lambda *a:None)
    monkeypatch.setattr(executor,'_scope_error',lambda *a:None)
    result=await executor.execute_tool('run_shell_command',state,command='never executed')
    assert result['type']=='OperationLimitError'
    dispatch.assert_not_called()


def test_existing_unbounded_operations_keep_their_contract(monkeypatch):
    monkeypatch.setattr(operation_limits.redis_manager,'redis_client',None)
    assert operation_limits.claim_action(None,'run_shell_command')
    assert operation_limits.claim_action(SimpleNamespace(context_data={}), 'run_shell_command')
