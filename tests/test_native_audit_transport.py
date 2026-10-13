"""Exercise real native tool dispatch, XML argument conversion, HTTP grants and ABS."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import httpx
import pytest
from agent_framework.agents.state import AgentContext
from agent_framework.tools.executor import execute_tool_with_validation
from agent_framework.tools.native_audit import actions
from agent_framework.llm.utils import parse_tool_invocations

@pytest.mark.asyncio
@pytest.mark.parametrize('named_xml',[False,True])
async def test_forward_annotations_and_real_agent_ids_reach_remote_abs(monkeypatch,tmp_path,named_xml):
    path=Path(__file__).resolve().parents[2]/'arena/arena-sensors-and-algorithms/src/console_api/native_audit_node.py'
    spec=importlib.util.spec_from_file_location('native_test_audit_node',path)
    node=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,node);spec.loader.exec_module(node)
    node.ROOT=tmp_path;node.ADMIN='test-control'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=node.app),base_url='http://test-node') as client:
        response=await client.post('/runs',json={'run_id':'transport-test','arm':'intents-plan','interface':{'transport':'signed-plan','max_documents':3},'seed':10},headers={'X-Audit-Key':'test-control'})
        tokens=response.json()['role_tokens']
        async def call(state,path,body=None):
            contract=actions.contract(state)
            response=await client.request('GET' if body is None else 'POST','/runs/transport-test'+path,json=body,headers={'Authorization':'Bearer '+contract['role_token']})
            assert response.status_code==200,response.text
            return response.json()
        monkeypatch.setattr(actions,'call',call)
        state=AgentContext(agent_id='agent_actual_native_id',task='audit',iteration=1,sandbox_info={'job_id':'transport-test'},context={'capabilities':['audit_observe','audit_publish'],'native_audit':{'kind':'arena-readonly-audit-v1','role':'inventory','arm':'intents-plan','run_id':'transport-test','role_token':tokens['inventory']}})
        # This is what XML calls supply: strings, including JSON arrays.
        markup='<function=audit_observe>'+('<documents>' if named_xml else '<parameter=documents>')+'["I1","I2","I3"]'+('</documents>' if named_xml else '</parameter>')+'</function>'
        invocation=parse_tool_invocations(markup)[0]
        result=await execute_tool_with_validation(invocation['toolName'],state,**invocation['args'])
        assert [row['document'] for row in result['documents']]==['I1','I2','I3'],result
        args={'recipient':'integration','refs':json.dumps([fact['id'] for row in result['documents'] for fact in row['facts']]),'receipts':json.dumps([row['receipt'] for row in result['documents']])}
        markup='<function=audit_publish>'+''.join((f'<{key}>{value}</{key}>' if named_xml else f'<parameter={key}>{value}</parameter>') for key,value in args.items())+'</function>'
        invocation=parse_tool_invocations(markup)[0]
        published=await execute_tool_with_validation(invocation['toolName'],state,**invocation['args'])
        assert published['accepted'],published
        integration=AgentContext(agent_id='agent_integrator',task='audit',sandbox_info={'job_id':'transport-test'},context={'capabilities':['audit_publish'],'native_audit':{'kind':'arena-readonly-audit-v1','role':'integration','arm':'intents-plan','run_id':'transport-test','role_token':tokens['integration']}})
        question=await execute_tool_with_validation('audit_publish',integration,recipient='events',refs='[]',receipts='[]',question='Please send your observations')
        assert question['accepted'],question
        assert node.RUNS['transport-test'].state['reads']==3
