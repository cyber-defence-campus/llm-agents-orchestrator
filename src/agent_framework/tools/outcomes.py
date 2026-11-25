"""Interpret explicit execution outcomes, independently of transport success."""

def execution_failed(result):
    if isinstance(result, str):
        return result.strip().lower().startswith('error:')
    queue = [(result, 0)]
    seen = set()
    while queue:
        value, depth = queue.pop()
        if not isinstance(value, dict) or id(value) in seen:
            continue
        seen.add(id(value))
        if (value.get('protocol')=='abs.operation/1-draft' and value.get('type')=='status'
                and value.get('state') in {'failed','cancelled'}):
            return True
        if value.get('isError') is True or value.get('is_error') is True:
            return True
        if value.get('error') or value.get('success') is False or value.get('ok') is False:
            return True
        if str(value.get('status', '')).lower() in {'failed', 'failure', 'error', 'execution_failed', 'cancelled', 'timeout', 'timed_out', 'transport_timeout'}:
            return True
        if type(value.get('exit_code')) is int and value['exit_code'] != 0:
            return True
        if value.get('cancelled') is True or value.get('timed_out') is True:
            return True
        if depth < 5:
            for key in ('result', 'data'):
                queue.append((value.get(key), depth + 1))
    return False
