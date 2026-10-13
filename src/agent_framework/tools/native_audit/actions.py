from __future__ import annotations
import json
from typing import Any
import httpx
from agent_framework.tools import register_tool

def contract(state):
    data=(getattr(state,'context_data',None) or {}).get('native_audit')
    if not data or data.get('kind')!='arena-readonly-audit-v1':raise ValueError('native_audit_contract_required')
    return data
async def call(state,path,body=None):
    scope=contract(state)
    async with httpx.AsyncClient(timeout=30,follow_redirects=False) as client:
        response=await client.request('GET' if body is None else 'POST',scope['node_url']+'/runs/'+scope['run_id']+path,
                                      headers={'Authorization':'Bearer '+scope['role_token']},json=body)
        if response.status_code>=400:
            return {'error':response.json().get('detail','Audit request failed'),'http_status':response.status_code}
        return response.json()

@register_tool(sandbox_execution=False)
async def audit_observe(documents:list[str],agent_state:Any):
    """Read authorized documents through the native ABS signed readonly plan interface.

    Args:
        documents: Document IDs in your assigned partition; single mode permits one, plan mode up to three.
    """
    scope=contract(agent_state)
    if scope['arm'] not in ('intents-single','intents-plan'):return {'error':'ABS tool is not part of this platform arm'}
    from abs.authorization.declarative_plan import DeclarativePlan,SCHEMA
    from abs.authorization.contract import Permit
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    prefix=f"/at/audit/{scope['run_id']}/{scope['role']}/"
    plan=DeclarativePlan.parse(json.dumps({'schema':SCHEMA,'steps':[
        {'id':doc,'adapter':'audit.document.read','adapter_version':'1','operation':'observe',
         'address':prefix+doc,'args':{'document':doc}} for doc in documents]},sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
    body={'plan':json.loads(plan.encoded),'request_id':agent_state.agent_id.replace('_','-')+'-'+str(agent_state.iteration)+'-'+plan.digest[:12]}
    grant=await call(agent_state,'/authorize',body)
    if 'error' in grant:return grant
    permit=Permit.from_dict(grant['permit']);permit.verify(Ed25519PublicKey.from_public_bytes(bytes.fromhex(grant['public_key'])))
    if permit.plan_digest!=plan.digest or permit.target_node_id!=scope['run_id']:return {'error':'Grant differs from requested readonly plan'}
    return await call(agent_state,'/plans',{**body,'permit':grant['permit']})

@register_tool(sandbox_execution=False)
async def audit_receive(agent_state:Any):
    """Receive a bounded chunk of original facts and questions. Repeat while remaining_messages is positive."""
    return await call(agent_state,'/inbox')

@register_tool(sandbox_execution=False)
async def audit_publish(recipient:str,refs:list[str]=[],receipts:list[str]=[],question:str='',reply_to:str|None=None,agent_state:Any=None):
    """Deliver selected received source facts or a clarification to another native role.

    Args:
        recipient: inventory, events, requirements or integration.
        refs: At most 80 exact fact IDs actually received; values remain the original source values.
        receipts: Opaque receipts returned by your document reads; use [] for previously acknowledged evidence.
        question: An optional question or concise interpretation of the selected evidence.
        reply_to: Original message ID when answering a question; otherwise omit it.
    """
    return await call(agent_state,'/share',{'recipient':recipient,'refs':refs,'receipts':receipts,'question':question,'reply_to':reply_to})

@register_tool(sandbox_execution=False)
async def audit_submit(report:dict,agent_state:Any):
    """Submit the joint report for all assigned components to Arena's independent evaluator.

    Args:
        report: Object with findings, unresolved_findings and notes. Each finding includes component, check, classification, assessment_tick and its typed proof.
    """
    result=await call(agent_state,'/report',report)
    if result.get('pass') is True:
        agent_state.mark_completed({'native_report':'accepted','verified_findings':result.get('verified_findings')})
        return {**result,'agent_completed':True}
    return result
