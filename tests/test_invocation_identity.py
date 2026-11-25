import asyncio
from types import SimpleNamespace

import pytest

from agent_framework.tools import executor
from agent_framework.tools.invocation import current_invocation


@pytest.mark.asyncio
async def test_invocation_metadata_survives_await_and_thread_without_leaking(monkeypatch):
    async def tool():
        before=current_invocation()
        await asyncio.sleep(.01)
        threaded=await asyncio.to_thread(current_invocation)
        assert current_invocation()==before==threaded
        return (before.agent_id,before.call_id)
    monkeypatch.setattr(executor,"get_tool_by_name",lambda name:tool)
    monkeypatch.setattr(executor,"needs_agent_state",lambda name:False)
    monkeypatch.setattr(executor,"_publish_tool_event",lambda *args,**kwargs:None)
    async def invoke(index):
        state=SimpleNamespace(agent_id=f"agent-{index}",sandbox_info={})
        return await executor._execute_tool_locally("probe",state,tool_call_id=f"call-{index}")
    answers=await asyncio.gather(*(invoke(i) for i in range(12)))
    assert answers==[(f"agent-{i}",f"call-{i}") for i in range(12)]
    assert current_invocation() is None


@pytest.mark.asyncio
async def test_invocation_metadata_is_reset_on_exception_and_cancel(monkeypatch):
    async def tool():
        raise RuntimeError("fixture")
    monkeypatch.setattr(executor,"get_tool_by_name",lambda name:tool)
    monkeypatch.setattr(executor,"needs_agent_state",lambda name:False)
    answer=await executor._execute_tool_locally("probe",None,tool_call_id="call-fail")
    assert answer["type"]=="ExecutionError" and current_invocation() is None
    entered=asyncio.Event()
    async def waiting():
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(executor,"get_tool_by_name",lambda name:waiting)
    checked=[]
    async def invoke():
        try:
            await executor._execute_tool_locally("probe",None,tool_call_id="call-cancel")
        finally:
            checked.append(current_invocation())
    task=asyncio.create_task(invoke());await entered.wait();task.cancel()
    with pytest.raises(asyncio.CancelledError):await task
    assert checked==[None]


@pytest.mark.parametrize("tool",["terminal","run_shell_command","run","install_beacon","spawn_sub_agent"])
def test_strict_profile_rejects_bypass_even_without_capability_list(tool):
    state=SimpleNamespace(context_data={"native_execution_profile":"abs.operation/1-draft"})
    assert executor._tool_contract_error(tool,state)
    assert executor._tool_contract_error("operation_exec",state) is None


@pytest.mark.parametrize("state,result",[("accepted",None),("unknown",None),("succeeded",{"stdout":"done","exit_code":0})])
def test_model_presentation_preserves_operation_identity_and_state(state,result):
    envelope=dict(protocol="abs.operation/1-draft",type="status",operation_id="op-1",state=state,revision=2,result=result)
    text,_=executor._format_tool_result("operation_exec",envelope)
    assert '"operation_id": "op-1"' in text and '"state": "'+state+'"' in text
    if result:assert "done" in text


def test_model_presentation_preserves_unknown_control_error_identity():
    envelope=dict(protocol="abs.operation/1-draft",type="error",operation_id="op-1",error="gateway_unavailable",execution_started=None)
    text,_=executor._format_tool_result("operation_exec",envelope)
    assert "op-1" in text and "gateway_unavailable" in text


@pytest.mark.asyncio
async def test_distinct_planned_calls_get_distinct_host_ids_in_strict_profile(monkeypatch):
    from agent_framework.agents.state import AgentContext
    from agent_framework.tools.outcomes import execution_failed
    agent=AgentContext(agent_id="agent",context={"native_execution_profile":"abs.operation/1-draft"})
    observed=[]
    async def run(invocation,state):
        observed.append(invocation['args']['tool_call_id'])
        return dict(protocol="abs.operation/1-draft",type="status",state="accepted",operation_id=str(len(observed)),result=None)
    monkeypatch.setattr(executor,"execute_tool_invocation",run)
    history=[]
    await executor.process_tool_invocations([dict(toolName="operation_exec",args={"argv":["echo","same"]}),dict(toolName="operation_exec",args={"argv":["echo","same"]})],history,agent)
    assert len(observed)==2 and len(set(observed))==2
    assert "operation_id" in history[-1]["content"] and "accepted" in history[-1]["content"]
    assert not execution_failed(dict(protocol="abs.operation/1-draft",type="status",state="unknown",result=None))
    assert execution_failed(dict(protocol="abs.operation/1-draft",type="status",state="cancelled",result=None))


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,tool,args", [("abs.operation/1-draft","operation_exec",{"argv":["/bin/true"]}),("abs.work/1-draft","typed_work",{"item":"health"})])
async def test_required_host_checkpoint_precedes_dispatch_and_survives_restore(monkeypatch,profile,tool,args):
    from agent_framework.agents import base
    from agent_framework.agents.state import AgentContext
    snapshots=[];attempts=[]
    def checkpoint(agent_id,state):snapshots.append(state.model_dump_json())
    monkeypatch.setattr(base.db,"checkpoint_agent_state",checkpoint)
    async def interrupted(batch,history,state):
        assert snapshots
        attempts.append(batch[0]["host_call_id"])
        raise asyncio.CancelledError()
    monkeypatch.setattr(base,"process_tool_invocations",interrupted)
    agent=base.BaseAgent.__new__(base.BaseAgent)
    agent.state=AgentContext(agent_id="agent",context={"native_execution_profile":profile})
    with pytest.raises(asyncio.CancelledError):
        await agent._dispatch_tools([dict(toolName=tool,args=args)])
    restored=base.BaseAgent.__new__(base.BaseAgent);restored.state=AgentContext.model_validate_json(snapshots[0])
    assert restored.state.pending_tool_invocations[0]["host_call_id"]==attempts[0]
    async def recovered(batch,history,state):
        attempts.append(batch[0]["host_call_id"]);return False
    monkeypatch.setattr(base,"process_tool_invocations",recovered)
    # No LLM is configured: startup must reconcile pending work first.
    assert await restored._execute_cycle() is False
    assert attempts==[attempts[0],attempts[0]]
    assert restored.state.pending_tool_invocations==[]
    assert AgentContext.model_validate_json(snapshots[-1]).pending_tool_invocations==[]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile,tool,args", [("abs.operation/1-draft","operation_exec",{"argv":["/bin/true"]}),("abs.work/1-draft","typed_work",{"item":"health"})])
async def test_unavailable_required_checkpoint_prevents_any_dispatch(monkeypatch,profile,tool,args):
    from agent_framework.agents import base
    from agent_framework.agents.state import AgentContext
    def unavailable(*args):raise RuntimeError("checkpoint unavailable")
    async def forbidden(*args):pytest.fail("executed without required checkpoint")
    monkeypatch.setattr(base.db,"checkpoint_agent_state",unavailable)
    monkeypatch.setattr(base,"process_tool_invocations",forbidden)
    agent=base.BaseAgent.__new__(base.BaseAgent)
    agent.state=AgentContext(agent_id="agent",context={"native_execution_profile":profile})
    with pytest.raises(RuntimeError,match="checkpoint unavailable"):
        await agent._dispatch_tools([dict(toolName=tool,args=args)])
    assert agent.state.pending_tool_invocations


def test_prepared_host_calls_do_not_alias_or_accept_argument_identity():
    shared={"argv":["/bin/true"],"tool_call_id":"model-chosen"}
    batch=executor.prepare_host_invocations([dict(toolName="operation_exec",args=shared),dict(toolName="operation_exec",args=shared)])
    assert batch[0]["host_call_id"]!=batch[1]["host_call_id"]
    assert batch[0]["args"] is not batch[1]["args"]
    assert batch[0]["host_call_id"]!="model-chosen"


def test_operation_reconciliation_is_supervision_not_a_stale_action():
    from agent_framework.agents.state import AgentContext
    state=AgentContext(agent_id="agent",context={"native_execution_profile":"abs.operation/1-draft"})
    reply=dict(protocol="abs.operation/1-draft",type="status",state="running",operation_id="op",result=None)
    for _ in range(20):state.record_tool_result("operation_inspect",{"operation_id":"op"},reply)
    assert state.tactical_memory["stale_action_streak"]==0
    assert not state.tactical_memory.get("pivot_required")
