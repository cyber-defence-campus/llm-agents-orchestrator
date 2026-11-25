"""Operator-only action/result ledger, independent of model context compaction.

Results may contain credentials and must be collected as private artifacts.
Oversize fields retain a digest and explicit truncation, never a silent rewrite.
"""
import asyncio
from datetime import UTC, datetime
import hashlib
import json
import logging
from agent_framework.state import redis_manager as db

logger = logging.getLogger('agent_framework.tools.audit')
FIELD_BYTES = 65536
RETENTION_SECONDS = 7 * 86400


def bounded(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode('utf-8')
    record = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
              'truncated': len(raw) > FIELD_BYTES}
    if record['truncated']:
        record['preview'] = raw[:FIELD_BYTES//2].decode('utf-8', errors='replace') + '\n[truncated]\n' + raw[-FIELD_BYTES//2:].decode('utf-8', errors='replace')
    else:
        record['value'] = json.loads(raw)
    return record


async def record(state, invocation, phase, result=None, error=None):
    info = getattr(state, 'sandbox_info', None)
    if not isinstance(info, dict) or not isinstance(info.get('job_id'), str):
        return False
    try:
        args = dict(invocation.get('args') or {})
        event = {'schema': 'tool-journal-v1', 'at': datetime.now(UTC).isoformat(),
                 'agent_id': state.agent_id, 'phase': phase,
                 'tool': invocation.get('toolName'), 'call_id': args.get('tool_call_id'),
                 'result_id': args.get('tool_result_id'), 'args': bounded(args)}
        if phase == 'finished': event['result'] = bounded(result)
        if error is not None: event['error_type'] = type(error).__name__
    except Exception:
        logger.warning('Tool journal serialization failed')
        return False
    def append():
        with db.redis_client.pipeline(transaction=True) as pipe:
            pipe.rpush('platform:tool_journal:' + info['job_id'], json.dumps(event, ensure_ascii=False))
            pipe.expire('platform:tool_journal:' + info['job_id'], RETENTION_SECONDS)
            pipe.execute()
    try:
        await asyncio.to_thread(append)
        return True
    except Exception:
        logger.warning('Tool journal write failed for call %s', event['call_id'])
        return False
