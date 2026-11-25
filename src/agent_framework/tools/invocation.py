"""Trusted invocation metadata, outside the model's tool argument schema."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Invocation:
    agent_id: str
    call_id: str | None


_current = ContextVar("tool_invocation", default=None)


def current_invocation():
    return _current.get()


@contextmanager
def invocation_scope(agent_id, call_id):
    token = _current.set(Invocation(agent_id, call_id))
    try:
        yield
    finally:
        _current.reset(token)
