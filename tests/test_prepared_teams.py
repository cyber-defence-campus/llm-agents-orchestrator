import asyncio
from types import SimpleNamespace

import pytest

from agent_framework.services.prepared_teams import (
    CampaignBudgetExhausted, PreparedTeams, enforce_budget, install_guard)


def state(agent='root', parent=None):
    return SimpleNamespace(agent_id=agent, parent_id=parent,
                           sandbox_info={'job_id': 'job'}, completed=True)


def prepared():
    registry = PreparedTeams()
    registry.prepare(('job', 'O1'), {'task': 'one'}, lambda: ({}, state()))
    return registry


def test_the_manager_guard_blocks_calls_after_the_campaign_cap():
    registry = PreparedTeams()
    usage = [0]
    registry.configure('job', token_cap=10, max_agents=4,
                       usage=lambda: usage[0])
    install_guard(registry)
    try:
        enforce_budget('job')
        usage[0] = 10
        with pytest.raises(CampaignBudgetExhausted):
            enforce_budget('job')
        # A non-campaign job keeps its ordinary call path.
        enforce_budget('ordinary')
    finally:
        install_guard(None)


def test_prepare_idempotent_and_conflict_does_not_rebuild():
    registry = prepared()
    record = registry.prepare(('job', 'O1'), {'task': 'one'}, lambda: pytest.fail('rebuilt'))
    assert record['phase'] == 'prepared' and not record['tasks']
    with pytest.raises(ValueError):
        registry.prepare(('job', 'O1'), {'task': 'changed'}, lambda: pytest.fail('rebuilt'))


@pytest.mark.asyncio
async def test_activation_once_and_waits_for_child_after_root_finishes():
    registry = prepared()
    child_release = asyncio.Event()
    calls = []

    def start(config, root):
        calls.append(root.agent_id)
        registry.before_start(root)
        registry.track(root, asyncio.create_task(asyncio.sleep(0)))
        child = state('child', 'root')
        registry.before_start(child)
        registry.track(child, asyncio.create_task(child_release.wait()))

    registry.activate(('job', 'O1'), start)
    registry.activate(('job', 'O1'), start)
    await asyncio.sleep(.01)
    assert calls == ['root']
    assert registry.status(('job', 'O1'))['active'] == ['child']
    assert registry.status(('job', 'O1'))['outcome'] is None
    child_release.set()
    await asyncio.sleep(.01)
    assert registry.status(('job', 'O1'))['outcome'] == 'completed'


@pytest.mark.asyncio
async def test_stop_freezes_descendants_and_does_not_touch_other_team():
    registry = prepared()
    unrelated = asyncio.create_task(asyncio.Event().wait())

    def start(config, root):
        registry.before_start(root)
        registry.track(root, asyncio.create_task(asyncio.Event().wait()))

    registry.activate(('job', 'O1'), start)
    status = await registry.stop(('job', 'O1'), .1)
    assert status['quiescent'] and status['outcome'] == 'stopped'
    with pytest.raises(ValueError):
        registry.before_start(state('late-child', 'root'))
    assert not unrelated.done()
    unrelated.cancel()
    await asyncio.gather(unrelated, return_exceptions=True)


def test_failed_start_cannot_be_retried_implicitly():
    registry = prepared()
    def fail(*args):
        raise RuntimeError('lost start')
    with pytest.raises(RuntimeError):
        registry.activate(('job', 'O1'), fail)
    result = registry.activate(('job', 'O1'), lambda *args: pytest.fail('restarted'))
    assert result['phase'] == 'start_failed'


def test_budget_is_shared_immutable_and_latches_after_accounting_failure():
    registry = prepared()
    counter = [0]
    registry.configure('job', token_cap=10, max_agents=4, usage=lambda: counter[0])
    assert not registry.allowance('job')['exhausted']
    with pytest.raises(ValueError, match='immutable'):
        registry.configure('job', token_cap=20, max_agents=4, usage=lambda: 0)
    counter[0] = 12
    assert registry.allowance('job')['overshoot_tokens'] == 2
    registry.records[('job', 'O1')]['phase'] = 'running'
    with pytest.raises(ValueError, match='exhausted'):
        registry.before_start(state())
    counter[0] = 0
    with pytest.raises(ValueError, match='decreasing'):
        registry.allowance('job')
    counter[0] = 12
    assert registry.allowance('job')['exhausted']


def test_generic_agent_cannot_start_inside_a_campaign():
    registry = PreparedTeams()
    registry.configure('job', token_cap=100, max_agents=4, usage=lambda: 0)
    with pytest.raises(ValueError, match='not a member'):
        registry.before_start(state('out-of-band'))
    assert 'out-of-band' not in registry.membership
    # The platform job marker covers the time before the allowance exists.
    fresh = PreparedTeams()
    with pytest.raises(ValueError, match='not a member'):
        fresh.before_start(state('out-of-band'), campaign_job=True)


def test_descendant_admission_uses_one_global_agent_cap():
    registry = prepared()
    registry.configure('job', token_cap=100, max_agents=4, usage=lambda: 0)
    registry.records[('job', 'O1')]['phase'] = 'running'
    registry.before_start(state())
    for index in range(3):
        registry.before_start(state(f'child-{index}', 'root'))
    with pytest.raises(ValueError, match='agent allowance'):
        registry.before_start(state('extra', 'root'))
    assert registry.allowance('job')['agents'] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['sequential', 'parallel'])
async def test_campaign_preparation_and_budget_over_http(mode, test_client, mock_redis, monkeypatch, tmp_path):
    import json
    import main
    from arena_infra.swarm_campaign import Artifacts, Campaign

    class Counter:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return json.dumps(dict(status='paused'))

        def exists(self, key):
            return bool(self.values)

        def hsetnx(self, key, name, value):
            self.values.setdefault(name, value)

        def hgetall(self, key):
            return dict(self.values)

    counter = Counter()
    monkeypatch.setattr(main, '_experiment_redis', lambda: counter)
    monkeypatch.setattr(main, 'prepared_teams', PreparedTeams())
    monkeypatch.setattr(main, 'prepared_campaign_parents', {})
    parent = state('parent')
    parent.context_data = {'capabilities': ['read_file']}
    mock_redis.get_agent_state.return_value = parent
    mock_redis.get_agent_nodes_by_job_id.return_value = {}
    mock_redis.get_all_edges.return_value = []
    started = []
    bound = set()

    def start(config, job):
        from agent_framework.agents.state import AgentContext
        root = AgentContext(**config['state'])
        assert root.agent_id in bound
        main.prepared_teams.before_start(root)
        started.append(root.agent_id)

        async def finish():
            counter.values['input_tokens'] += 2
            root.completed = True

        main.prepared_teams.track(root, asyncio.create_task(finish()))
        return root.agent_id, root

    monkeypatch.setattr(main, 'start_agent_task', start)

    async def request(method, path, body):
        def call():
            response = test_client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()
        return await asyncio.to_thread(call)

    async def bind(team, agent_id, parent_id):
        bound.add(agent_id)

    source = dict(platform='tactics', job_id='job', model='chosen-model',
                  context={'capabilities': ['read_file']}, mode=mode,
                  duration_seconds=5, token_cap=100, max_agents=8,
                  teams=[dict(objective=f'O{i}', brief=f'Objective {i}', timeout_seconds=2)
                         for i in range(1, 5)])
    artifacts = Artifacts(tmp_path / 'run', create=True)
    campaign = Campaign(source, request, artifacts, bind)
    manifest = await campaign.prepare()
    assert len(bound) == 4 and not started
    # Preparation is retryable on the same manager without another parent.
    parent_count = len(main.prepared_campaign_parents)
    assert parent_count == 1
    result = await campaign.execute()
    assert set(result['outcomes'].values()) == {'completed'}
    assert result['tokens'] == 8
    assert len(started) == len(manifest['agents']) == 4


def test_api_prepares_without_start_and_rejects_changed_manager(test_client, mock_redis, monkeypatch):
    import main
    from unittest.mock import patch
    monkeypatch.setattr(main, 'prepared_teams', PreparedTeams())
    parent = state('parent')
    parent.context_data = {'capabilities': ['read_file'], 'scope': 'lab'}
    mock_redis.get_agent_state.return_value = parent
    mock_redis.get_agent_nodes_by_job_id.return_value = {}
    mock_redis.get_all_edges.return_value = []
    body = dict(name='O1', task='one', job_id='job', parent_id='parent',
                instance_id=main.MANAGER_INSTANCE_ID,
                context={'capabilities': ['read_file'], 'scope': 'lab'})
    with patch('main.start_agent_task') as start:
        first = test_client.put('/experiments/job/teams/O1', json=body)
        again = test_client.put('/experiments/job/teams/O1', json=body)
        assert first.status_code == 200, first.text
        assert first.json()['agent_id'] == again.json()['agent_id']
        start.assert_not_called()
        response = test_client.post('/experiments/job/teams/O1/activate', json={'instance_id': 'old'})
        assert response.status_code == 409
        start.assert_not_called()
    denied = dict(body, context={'capabilities': ['run'], 'scope': 'lab'})
    assert test_client.put('/experiments/job/teams/O2', json=denied).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['sequential', 'parallel'])
async def test_controller_to_native_api(mode, test_client, mock_redis, monkeypatch):
    """Exercise HTTP contracts and real admission registry; no LLM or range."""
    import main
    from arena_infra.four_swarms import Team, run
    from arena_infra.native_teams import NativeTeams

    monkeypatch.setattr(main, 'prepared_teams', PreparedTeams())
    parent = state('parent')
    parent.context_data = {'capabilities': ['read_file']}
    mock_redis.get_agent_state.return_value = parent
    mock_redis.get_agent_nodes_by_job_id.return_value = {}
    mock_redis.get_all_edges.return_value = []
    bound = set()
    started = []

    def start(config, job):
        from agent_framework.agents.state import AgentContext
        root = AgentContext(**config['state'])
        assert root.agent_id in bound
        main.prepared_teams.before_start(root)
        started.append(root.agent_id)

        async def finish():
            await asyncio.sleep(.01)
            root.completed = True

        task = asyncio.create_task(finish())
        main.prepared_teams.track(root, task)
        return root.agent_id, root

    monkeypatch.setattr(main, 'start_agent_task', start)

    async def request(method, path, body):
        # TestClient owns its ASGI event loop; calls must not block this loop.
        def call():
            response = test_client.request(method, path, json=body)
            response.raise_for_status()
            return response.json()
        return await asyncio.to_thread(call)

    async def budget():
        return True

    async def bind(team, agent_id):
        bound.add(agent_id)

    teams = [Team(f'O{i}', f'Objective {i}', 2) for i in range(1, 5)]
    adapter = NativeTeams(request, job_id='job', instance_id=main.MANAGER_INSTANCE_ID,
                          contracts={t.objective: dict(name=t.objective, parent_id='parent',
                                     context={'capabilities': ['read_file']}) for t in teams},
                          budget_guard=budget, poll_seconds=.01)
    await adapter.prepare(teams, bind)
    assert len(bound) == 4 and not started
    outcomes = await run(adapter, teams, mode=mode, duration_seconds=5, emit=lambda e: None)
    assert set(outcomes.values()) == {'completed'}
    assert len(started) == len(set(started)) == 4
