import pytest
from agent_framework.tools.executor import _check_error_result
from agent_framework.agents.state import AgentContext

@pytest.mark.parametrize('result',[
    {'result':{'result':{'exit_code':2,'content':'syntax error'}}},
    {'success':True,'data':{'exit_code':124,'timed_out':True}},
    {'success':True,'data':{'cancelled':True,'exit_code':130}},
    {'success':False,'error':'context changed'},
])
def test_transport_completion_is_not_command_success(result):
    failed,payload=_check_error_result(result)
    assert failed and payload is result
    state=AgentContext()
    state.record_tool_result('run',{},result)
    assert state.tactical_memory['last_action_failed'] is True

@pytest.mark.parametrize('result',[
    {'result':{'result':{'exit_code':0,'content':'literal error: in text'}}},
    {'status':'running','exit_code':None},
    {'success':True,'data':{'body':'{"error":"ordinary file content"}'}},
    {'success':True,'data':{'status':404,'body':'Not found'}},
])
def test_explicit_failures_are_distinguished_from_content_and_running(result):
    assert _check_error_result(result)==(False,None)
    assert not AgentContext._result_failed(result,False)
