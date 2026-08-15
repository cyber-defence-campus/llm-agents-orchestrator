from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional, List
import redis
import logging
import json

logger = logging.getLogger("agent.state.usage")

UPDATE_MAX_SCRIPT = """
local current = tonumber(redis.call('HGET', KEYS[1], ARGV[1])) or 0
if tonumber(ARGV[2]) > current then
    redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
end
"""


def increment_usage_stats(
    client: redis.Redis,
    job_id: str,
    agent_id: Optional[str],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost: float = 0.0,
    requests: int = 1,
    failed_requests: int = 0,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
    model_name: Optional[str] = None,
) -> None:
    if not job_id or not client:
        return

    usage_base_key = f"platform:usage:job:{job_id}"
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    date_usage_key = f"platform:usage:date:{date_str}"
    global_usage_key = "platform:usage:global"

    try:
        pipe = client.pipeline()

        def track_context(key, tokens):
            if tokens > 0:
                pipe.hset(key, "active_context_size", tokens)
                pipe.eval(UPDATE_MAX_SCRIPT, 1, key, "max_context_size", tokens)

        def inc(key):
            pipe.hincrby(key, "input_tokens", input_tokens)
            pipe.hincrby(key, "output_tokens", output_tokens)
            pipe.hincrbyfloat(key, "cost", cost)
            pipe.hincrby(key, "requests", requests)
            pipe.hincrby(key, "failed_requests", failed_requests)
            pipe.hincrby(key, "cached_tokens", cached_tokens)
            pipe.hincrby(key, "cache_creation_tokens", cache_creation_tokens)
            track_context(key, input_tokens)

        inc(f"{usage_base_key}:global")

        if agent_id:
            inc(f"{usage_base_key}:agents:{agent_id}")

        if model_name:
            inc(f"{usage_base_key}:models:{model_name}")
            pipe.sadd(f"{usage_base_key}:used_models", model_name)

        inc(global_usage_key)
        if model_name:
            inc(f"{global_usage_key}:models:{model_name}")
            pipe.sadd(f"{global_usage_key}:used_models", model_name)

        inc(f"{date_usage_key}:global")
        if model_name:
            inc(f"{date_usage_key}:models:{model_name}")
        inc(f"{date_usage_key}:jobs:{job_id}")

        pipe.rpush(
            f"{usage_base_key}:calls",
            json.dumps(
                {
                    "inserted_at": datetime.now(UTC).isoformat(),
                    "agent_id": agent_id,
                    "model": model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": cost,
                }
            ),
        )

        pipe.execute()

        channel = f"platform:job-updates:{job_id}"
        try:
            client.publish(
                channel,
                json.dumps({"event": "usage_updated", "data": {"job_id": job_id}}),
            )
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Failed to increment usage: {e}")


def get_usage_calls(client: redis.Redis, job_id: str) -> List[Dict[str, Any]]:
    if not job_id or not client:
        return []
    try:
        raw = client.lrange(f"platform:usage:job:{job_id}:calls", 0, -1)
    except Exception:
        return []
    out = []
    for r in raw:
        try:
            out.append(json.loads(r))
        except (ValueError, TypeError):
            continue
    return out


def get_usage_stats(client: redis.Redis, job_id: str) -> Dict[str, Any]:
    if not job_id or not client:
        return {}

    base = f"platform:usage:job:{job_id}:global"
    stats = _fetch_hash(client, base)

    models_stats = {}
    try:
        models = client.smembers(f"platform:usage:job:{job_id}:used_models")
        for m in models:
            models_stats[m] = _fetch_hash(
                client, f"platform:usage:job:{job_id}:models:{m}"
            )
    except Exception:
        pass

    agents_stats = {}
    try:
        job_agents_key = f"platform:job_agents:{job_id}"
        agent_ids = client.smembers(job_agents_key)

        usage_base_key = f"platform:usage:job:{job_id}"

        agent_graph_data = {}
        try:
            for aid in agent_ids:
                node_raw = client.hget("agent_graph:nodes", aid)
                if node_raw:
                    node_data = json.loads(node_raw)
                    agent_graph_data[aid] = {
                        "name": node_data.get("name", aid),
                        "model": node_data.get("model", "Unknown"),
                    }
        except Exception:
            pass

        for agent_id in agent_ids:
            agent_key = f"{usage_base_key}:agents:{agent_id}"
            agent_data = _fetch_hash(client, agent_key)
            if agent_data:
                graph_info = agent_graph_data.get(agent_id, {})
                agent_name = graph_info.get("name", agent_id)
                agent_model = graph_info.get("model", "Unknown")

                agents_stats[agent_id] = {
                    "name": agent_name,
                    "model": agent_model,
                    "stats": agent_data,
                }

    except Exception as e:
        logger.warning(f"Failed to fetch agent stats for job {job_id}: {e}")

    stats["agents_stats"] = agents_stats
    stats["models_stats"] = models_stats
    return stats


def get_global_usage_stats(client: redis.Redis) -> Dict[str, Any]:
    if not client:
        return {}

    global_key = "platform:usage:global"

    total_stats = _fetch_hash(client, global_key)

    models_stats = {}
    try:
        models = client.smembers(f"{global_key}:used_models")
        for m in models:
            models_stats[m] = _fetch_hash(client, f"{global_key}:models:{m}")
    except Exception:
        pass

    return {
        "total": total_stats,
        "total_tokens": total_stats.get("input_tokens", 0)
        + total_stats.get("output_tokens", 0),
        "models_stats": models_stats,
    }


def get_usage_history(client: redis.Redis, days: int = 30) -> List[Dict[str, Any]]:
    if not client:
        return []

    history = []
    today = datetime.now(UTC)

    for i in range(days):
        date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        key = f"platform:usage:date:{date_str}:global"

        stats = _fetch_hash(client, key)
        if stats:
            stats["date"] = date_str
            history.append(stats)

    return history


def _fetch_hash(client: redis.Redis, key: str) -> Dict[str, Any]:
    try:
        raw = client.hgetall(key)
        return {k: (float(v) if k == "cost" else int(float(v))) for k, v in raw.items()}
    except Exception:
        return {}
