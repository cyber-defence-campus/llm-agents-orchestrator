import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from agent_framework.tools import audit_journal as journal


def test_large_result_is_explicitly_truncated_with_full_digest():
    import hashlib,json
    value={'stdout':'x'*100000+'tail'}
    row=journal.bounded(value)
    assert row['truncated'] and row['preview'].endswith('tail"}')
    assert row['sha256']==hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
    assert 'value' not in row


@pytest.mark.asyncio
async def test_journal_is_separate_from_compactable_messages_and_preserves_call_identity(monkeypatch):
    events=[]
    class Pipe:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def rpush(self,key,row):events.append((key,row))
        def expire(self,*args):pass
        def execute(self):pass
    monkeypatch.setattr(journal.db,'redis_client',SimpleNamespace(pipeline=lambda **kw:Pipe()))
    state=SimpleNamespace(agent_id='worker',sandbox_info={'job_id':'scope'},messages=[])
    inv={'toolName':'read_file','args':{'path':'/tmp/a','tool_call_id':'call-a','tool_result_id':'result-a'}}
    assert await journal.record(state,inv,'started')
    state.messages.clear()
    assert await journal.record(state,inv,'finished',result={'success':False,'error':'missing'})
    import json
    a,b=[json.loads(row) for _,row in events]
    assert a['call_id']==b['call_id']=='call-a'
    assert b['result']['value']['success'] is False
    assert all(key=='platform:tool_journal:scope' for key,_ in events)


@pytest.mark.asyncio
async def test_missing_sink_is_observable_without_changing_action_outcome(monkeypatch,caplog):
    monkeypatch.setattr(journal.db,'redis_client',None)
    state=SimpleNamespace(agent_id='worker',sandbox_info={'job_id':'scope'})
    assert await journal.record(state,{'toolName':'read_file','args':{}},'started') is False
    assert 'write failed' in caplog.text


@pytest.mark.asyncio
async def test_unserializable_result_does_not_break_the_action(caplog):
    state=SimpleNamespace(agent_id='worker',sandbox_info={'job_id':'scope'})
    result={};result['cycle']=result
    assert await journal.record(state,{'toolName':'run','args':{}},'finished',result=result) is False
    assert 'serialization failed' in caplog.text
