from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from agent_framework.tools.terminal import actions
from agent_framework.tools.terminal.session_scope import scoped_terminal_id


def test_explicit_aliases_cannot_join_another_agent_terminal():
    first=scoped_terminal_id('worker-a','main')
    second=scoped_terminal_id('worker-b','main')
    assert first!=second
    assert scoped_terminal_id('worker-a',first)==first
    assert scoped_terminal_id('worker-b',first) not in [first,second]
    assert scoped_terminal_id('worker-a')=='worker-a'
    assert ':' not in scoped_terminal_id('worker-a','host:port/path')


@pytest.mark.asyncio
async def test_same_labels_keep_separate_persistent_state(monkeypatch):
    states={}
    async def execute(**kwargs):
        key=kwargs['terminal_id']
        if kwargs['command'].startswith('set '):states[key]=kwargs['command'][4:]
        return {'content':states.get(key,''),'status':'completed','exit_code':0,'terminal_id':key}
    manager=SimpleNamespace(execute_command=AsyncMock(side_effect=execute))
    monkeypatch.setattr(actions,'get_terminal_manager',lambda:manager)
    a=SimpleNamespace(agent_id='a');b=SimpleNamespace(agent_id='b')
    await actions.run_shell_command('set host-a',session_id='main',agent_state=a)
    await actions.run_shell_command('set host-b',session_id='main',agent_state=b)
    assert (await actions.run_shell_command('get',terminal_id='main',agent_state=a))['content']=='host-a'
    assert (await actions.run_shell_command('get',terminal_id='main',agent_state=b))['content']=='host-b'

@pytest.mark.asyncio
async def test_two_real_tmux_shells_with_same_alias_remain_private(tmp_path, monkeypatch):
    import shutil
    from uuid import uuid4
    if not shutil.which('tmux'):
        pytest.skip('tmux unavailable')
    from agent_framework.tools.terminal import terminal_manager
    from agent_framework.tools.terminal.shell_session import ShellExecutor
    manager = terminal_manager.get_terminal_manager()
    monkeypatch.setattr(terminal_manager, 'ShellExecutor',
                        lambda identity: ShellExecutor(identity, work_dir=str(tmp_path)))
    owners = [SimpleNamespace(agent_id='audit-'+uuid4().hex) for _ in range(2)]
    try:
        for owner, value in zip(owners, ['host-a', 'host-b']):
            result = await actions.run_shell_command('export AUDIT_HOST='+value,
                            session_id='main', agent_state=owner)
            assert result['exit_code'] == 0
        for owner, value in zip(owners, ['host-a', 'host-b']):
            result = await actions.run_shell_command('printf "%s" "$AUDIT_HOST"',
                            terminal_id='main', agent_state=owner)
            assert result['content'].strip() == value
    finally:
        for owner in owners:
            manager.close_session(scoped_terminal_id(owner.agent_id, 'main'))


def test_executor_scopes_before_dispatch_without_modifying_caller_arguments():
    from agent_framework.tools.executor import _scope_terminal_kwargs
    state = SimpleNamespace(agent_id='worker')
    original = {'command':'id', 'terminal_id':'main'}
    scoped = _scope_terminal_kwargs('run_shell_command', original, state)
    assert scoped['session_id'] == scoped_terminal_id('worker', 'main')
    assert 'terminal_id' not in scoped
    assert original['terminal_id'] == 'main'
    assert _scope_terminal_kwargs('run_shell_command', scoped, state) == scoped
