"""Single-manager, two-phase native team lifecycle for controlled experiments.

Preparation never starts a model. Records intentionally survive task completion
but not a manager restart: callers must pin the manager instance identifier.
"""
import asyncio
import hashlib
import json


class CampaignBudgetExhausted(RuntimeError):
    """The campaign allowance is exhausted; the agent must stop calling."""


_guard = None


def install_guard(instance):
    """Expose the manager's single allowance ledger to the model call path."""
    global _guard
    _guard = instance


def enforce_budget(job_id):
    """Block a model call once the campaign's observed allowance is spent.

    The runner is the normal budget guard, but it can disappear while the
    manager's agent tasks keep running (the interrupted I-P attempt reached
    19.15M tokens after the driver was lost). This check runs in the manager
    process, immediately before each provider call, so a lost runner cannot
    turn the 12M ceiling into an advisory number. It is a post-response
    measurement: one in-flight call may still overshoot by its own usage.
    """
    if _guard is None or not job_id or job_id not in _guard.policies:
        return
    status = _guard.allowance(job_id)
    if status['exhausted']:
        raise CampaignBudgetExhausted(
            f"campaign {job_id!r} exhausted its {status['token_cap']} token "
            f"allowance at {status['tokens']} observed tokens")


class PreparedTeams:
    def __init__(self):
        self.records = {}
        self.membership = {}
        self.policies = {}

    def configure(self, job_id, *, token_cap, max_agents, usage):
        """One immutable allowance for all teams and their descendants.

        Tokens are observed after provider replies. Admission stops at the
        observed limit; in-flight calls can overshoot it and must be reported.
        """
        limits = (token_cap, max_agents)
        if any(type(n) is not int or n <= 0 for n in limits) or max_agents < 4:
            raise ValueError('positive token cap and at least four agents required')
        if job_id in self.policies:
            if self.policies[job_id]['limits'] != limits:
                raise ValueError('campaign allowance is immutable')
            return
        if any(key[0] == job_id and record['phase'] != 'prepared'
               for key, record in self.records.items()):
            raise ValueError('configure allowance before activating teams')
        self.policies[job_id] = dict(limits=limits, usage=usage, peak=0,
                                     exhausted=False, agents=set())

    def allowance(self, job_id):
        policy = self.policies[job_id]
        try:
            total = policy['usage']()
            if type(total) is not int or total < policy['peak']:
                raise ValueError('invalid or decreasing usage counter')
        except Exception:
            policy['exhausted'] = True
            raise
        policy['peak'] = total
        cap, max_agents = policy['limits']
        policy['exhausted'] |= total >= cap
        return dict(token_cap=cap, tokens=total, max_agents=max_agents,
                    agents=len(policy['agents']), exhausted=policy['exhausted'],
                    overshoot_tokens=max(0, total - cap))

    def prepare(self, key, payload, build):
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if key in self.records:
            record = self.records[key]
            if record['digest'] != digest:
                raise ValueError('team key already has a different contract')
            return record
        config, state = build()
        record = dict(digest=digest, config=config, state=state,
                      phase='prepared', tasks={}, states={})
        self.records[key] = record
        self.membership[state.agent_id] = key
        return record

    def before_start(self, state, *, campaign_job=False):
        job_id = state.sandbox_info.get('job_id')
        key = self.membership.get(state.agent_id) or self.membership.get(state.parent_id)
        if key is None:
            # A campaign job has an allowance before any team exists. Once it
            # does, an agent that is not in a prepared team is a generic
            # creation path (or a stale identity), not experiment work. The
            # job marker covers the window before the allowance is configured.
            if campaign_job or job_id in self.policies:
                raise ValueError('agent is not a member of a prepared campaign team')
            return
        record = self.records[key]
        if record['phase'] != 'running':
            raise ValueError('team is not accepting new agents')
        expected_job = record['state'].sandbox_info.get('job_id')
        if state.sandbox_info.get('job_id') != expected_job:
            raise ValueError('team descendant changed campaign')
        if expected_job in self.policies:
            policy = self.policies[expected_job]
            allowance = self.allowance(expected_job)
            if allowance['exhausted']:
                raise ValueError('campaign token allowance exhausted')
            if (state.agent_id not in policy['agents']
                    and allowance['agents'] >= allowance['max_agents']):
                raise ValueError('campaign agent allowance exhausted')
            policy['agents'].add(state.agent_id)
        self.membership[state.agent_id] = key

    def track(self, state, task):
        key = self.membership.get(state.agent_id)
        if key is not None:
            record = self.records[key]
            record['tasks'][state.agent_id] = task
            record['states'][state.agent_id] = state

    def activate(self, key, start):
        record = self.records[key]
        if record['phase'] == 'prepared':
            record['phase'] = 'running'
            try:
                start(record['config'], record['state'])
            except BaseException:
                # Do not repeat an ambiguous start on the next request.
                record['phase'] = 'start_failed'
                raise
        return self.status(key)

    def status(self, key):
        record = self.records[key]
        tasks = record['tasks']
        active = [agent for agent, task in tasks.items() if not task.done()]
        phase = record['phase']
        outcome = None
        if phase == 'running' and tasks and not active:
            completed = all(
                not task.cancelled() and task.exception() is None
                and record['states'][agent].completed
                for agent, task in tasks.items())
            outcome = 'completed' if completed else 'failed'
        if phase == 'stopping' and not active:
            outcome = 'stopped'
        return dict(agent_id=record['state'].agent_id, phase=phase,
                    active=active, members=list(tasks), outcome=outcome,
                    quiescent=not active)

    async def stop(self, key, timeout):
        record = self.records[key]
        # Freeze admission synchronously before waiting for any cancellation.
        record['phase'] = 'stopping'
        pending = [task for task in record['tasks'].values() if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=timeout)
        return self.status(key)
