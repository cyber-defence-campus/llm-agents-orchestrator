"""Readonly audit waits keep fixed budgets and resume only on recorded input."""
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from agent_framework.agents.base import BaseAgent
from agent_framework.agents.state import AgentContext
from agent_framework.tools.native_audit import actions
from agent_framework.tools.agent_management.actions import enter_wait_mode


def agent(policy='message_driven', native=True):
    scope={'kind':'arena-readonly-audit-v1','role':'integration','waiting_policy':policy}
    state=AgentContext(agent_id='audit-lifecycle-test',iteration=9,max_iterations=18,
                       context={'native_audit':scope} if native else {})
    state.context_data['native_wait_deadline_at']=datetime.now(UTC).timestamp()+120
    return BaseAgent({'state':state,'llm_config':{'model_name':'test-model'}})


@pytest.mark.asyncio
async def test_idle_native_ticks_do_not_resume_or_consume_model_turns():
    with patch('agent_framework.agents.base.db') as db, patch('agent_framework.agents.base.asyncio.sleep',new_callable=AsyncMock):
        a=agent();a.context.set_waiting(timeout=0);before=len(a.context.messages)
        a.llm.generate=AsyncMock()
        for _ in range(4):await a._wait_cycle()
        assert a.context.waiting_for_input and a.context.iteration==9
        assert len(a.context.messages)==before
        a.llm.generate.assert_not_awaited()
        db.update_agent_status.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('native,policy',[(False,'message_driven'),(True,'timed')])
async def test_timed_and_ordinary_agents_preserve_timeout_behavior(native,policy):
    with patch('agent_framework.agents.base.db') as db:
        a=agent(policy,native);a.context.set_waiting(timeout=0)
        await a._wait_cycle()
        assert not a.context.waiting_for_input and a.context.status=='running'
        db.update_agent_status.assert_called_with(a.context.agent_id,'running')


@pytest.mark.asyncio
async def test_notification_resumes_without_renewing_deadline_or_assuming_evidence():
    with patch('agent_framework.agents.base.db') as db:
        a=agent();a.context.set_waiting(timeout=0)
        deadline=a.context.context_data['native_wait_deadline_at']
        db.pop_all_messages_for_agent.return_value=[{'type':'message','from':'inventory','content':'Audit message m1 accepted. Call audit_receive.'}]
        await a._check_messages()
        assert not a.context.waiting_for_input and not a.context.completed
        assert a.context.iteration==9 and a.context.context_data['native_wait_deadline_at']==deadline
        assert 'Call audit_receive' in a.context.messages[-1]['content']


@pytest.mark.asyncio
async def test_expired_restored_native_deadline_is_not_extended():
    with patch('agent_framework.agents.base.db') as db:
        a=agent();deadline=datetime.now(UTC).timestamp()-1
        a.context.context_data['native_wait_deadline_at']=deadline
        a._initialize_task=AsyncMock();a._execute_cycle=AsyncMock()
        db.get_agent_status.return_value='running'
        await a.start_lifecycle('readonly audit')
        assert a.context.stop_requested and not a.context.completed
        assert a.context.context_data['native_wait_deadline_at']==deadline
        assert a.context.iteration==9
        a._execute_cycle.assert_not_awaited()
        db.update_agent_node_fields.assert_called_with(a.context.agent_id,{'stop_reason':'native audit run deadline reached'})


@pytest.mark.asyncio
async def test_control_stop_interrupts_wait_before_another_model_turn():
    with patch('agent_framework.agents.base.db') as db:
        a=agent();a.context.set_waiting(timeout=10)
        a._initialize_task=AsyncMock();a._execute_cycle=AsyncMock()
        db.get_agent_status.return_value='waiting'
        db.pop_all_messages_for_agent.return_value=[{'type':'control','content':'stop'},
            {'type':'message','from':'inventory','content':'Late notification must not clear stop'}]
        await a.start_lifecycle('readonly audit')
        assert a.context.stop_requested and a.context.iteration==9
        a._execute_cycle.assert_not_awaited()


def test_wait_tool_declares_native_resume_condition_without_changing_capabilities():
    with patch('agent_framework.agents.base.db'),patch('agent_framework.tools.agent_management.actions.db'):
        a=agent();scope=dict(a.context.context_data['native_audit'])
        result=enter_wait_mode(a.context,max_wait_seconds=10)
        assert result['waiting_policy']=='message_driven' and result['idle_check_seconds']==10
        assert 'max_wait_seconds' not in result and 'fixed native run deadline' in result['resume_condition']
        assert a.context.context_data['native_audit']==scope and a.context.iteration==9


@pytest.mark.asyncio
@pytest.mark.parametrize('accepted',[True,False])
async def test_only_accepted_server_report_completes_native_integrator(monkeypatch,accepted):
    state=AgentContext(agent_id='root',iteration=14,max_iterations=18)
    reply={'pass':accepted,'verified_findings':36 if accepted else 0,'can_correct':not accepted}
    monkeypatch.setattr(actions,'call',AsyncMock(return_value=reply))
    result=await actions.audit_submit({'findings':[]},state)
    assert state.completed is accepted
    assert result.get('agent_completed',False) is accepted
    assert state.iteration==14 and state.max_iterations==18
    if accepted:assert state.final_result=={'native_report':'accepted','verified_findings':36}
    else:assert result['can_correct'] is True


def test_ending_a_wait_preserves_accepted_completion_and_stop_flags():
    with patch('agent_framework.agents.base.db'):
        a=agent();a.context.set_waiting(timeout=10);a.context.mark_completed({'accepted':True})
        a.context.signal_stop();a._end_native_wait('report_accepted')
        assert a.context.completed and a.context.stop_requested
        assert not a.context.waiting_for_input


def test_wait_telemetry_contains_no_context_or_message_payload():
    from agent_framework.runtime.native_audit_lifecycle import record_wait
    import json
    with patch('agent_framework.runtime.native_audit_lifecycle.db') as db:
        state=AgentContext(agent_id='role',iteration=3,sandbox_info={'job_id':'audit'},
            context={'native_audit':{'kind':'arena-readonly-audit-v1','role':'events',
                     'role_token':'PRIVATE_CREDENTIAL','waiting_policy':'message_driven'},'secret':'PRIVATE_CONTEXT'})
        record_wait(state,'wait.enter')
        payload=json.loads(db.redis_client.rpush.call_args.args[1])
        assert payload['role']=='events' and payload['iteration']==3
        assert 'PRIVATE_' not in json.dumps(payload) and 'messages' not in payload
        db.reset_mock();state.context_data={};record_wait(state,'wait.enter')
        db.redis_client.rpush.assert_not_called()


@pytest.mark.asyncio
async def test_lifecycle_preserves_the_accepted_report_completion_receipt():
    with patch('agent_framework.agents.base.db') as db:
        a=agent();a._initialize_task=AsyncMock()
        db.get_agent_status.return_value='running';db.pop_all_messages_for_agent.return_value=[]
        async def accepted_cycle():
            a.context.mark_completed({'native_report':'accepted','verified_findings':36})
            return True
        a._execute_cycle=accepted_cycle
        await a.start_lifecycle('readonly audit')
        assert a.context.completed and a.context.final_result['verified_findings']==36
        assert a.context.iteration==10
