"""Minimal, credential-free wait telemetry for the readonly native audit."""
import json
import time
from agent_framework.state import redis_manager as db


def record_wait(state, kind, reason=None):
    scope=(getattr(state,'context_data',None) or {}).get('native_audit') or {}
    job=(getattr(state,'sandbox_info',None) or {}).get('job_id')
    if scope.get('kind')!='arena-readonly-audit-v1' or not job:
        return
    event={'kind':kind,'timestamp':time.time(),'agent_id':state.agent_id,
           'role':scope.get('role'),'iteration':state.iteration,
           'waiting_policy':scope.get('waiting_policy','timed')}
    if reason is not None:event['reason']=reason
    try:
        key='native-audit:lifecycle:'+job
        db.redis_client.rpush(key,json.dumps(event))
        db.redis_client.expire(key,86400)
    except Exception:
        # Execution is not changed by a missing observation sink.
        pass
