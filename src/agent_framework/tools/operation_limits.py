"""Optional local budgets shared by every agent in a prompt-led operation."""
from agent_framework.state import redis_manager


def claim_action(state, tool_name: str) -> bool:
    context = getattr(state, "context_data", None) or {}
    if context.get("bounded_operation") is not True or tool_name == "complete_assignment":
        return True
    job_id = (getattr(state, "sandbox_info", None) or {}).get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return False
    # Missing configuration, pause, stop, expiry and Redis failure fail closed.
    try:
        return bool(redis_manager.redis_client.eval('''
            local config = cjson.decode(redis.call('GET', KEYS[1]) or '{}')
            local limits = config.execution_limits
            local now = redis.call('TIME')
            if not limits or config.status ~= 'running' or tonumber(now[1]) >= limits.deadline then return 0 end
            local used = tonumber(redis.call('GET', KEYS[2]) or '0')
            if used >= limits.max_actions then return 0 end
            redis.call('INCR', KEYS[2])
            redis.call('EXPIRE', KEYS[2], 86400)
            return 1
        ''', 2, f"platform:job_config:{job_id}", f"platform:operation_actions:{job_id}"))
    except Exception:
        return False
