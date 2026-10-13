"""Task ownership and bounded cancellation without LLM or persisted graph truth."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest
import httpx
import main as manager


@pytest.mark.asyncio
@pytest.mark.parametrize('state',['absent','running','finished','error','cancelled'])
async def test_task_observation_ignores_cached_agent_status(monkeypatch,state):
    cached=MagicMock();cached.get_agent_status.return_value='running' if state=='absent' else 'completed'
    monkeypatch.setattr(manager,'state_manager',cached)
    monkeypatch.setattr(manager,'active_agents',{})
    task=None
    if state!='absent':
        async def work():
            if state=='running':await asyncio.Event().wait()
            if state=='error':raise RuntimeError('fixture failure')
        task=asyncio.create_task(work())
        if state=='cancelled':task.cancel()
        await asyncio.sleep(0)
        manager.active_agents['fixture']={'task':task}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app),base_url='http://manager') as client:
            response=await client.get('/tasks/fixture/status')
        assert response.status_code==200
        data=response.json()
        assert data['task_state']==state
        assert data['task_active'] is (state=='running')
        assert data['scope']=='single-manager-process' and data['instance_id']==manager.MANAGER_INSTANCE_ID
        cached.get_agent_status.assert_not_called()
    finally:
        if task and not task.done():
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass


@pytest.mark.asyncio
@pytest.mark.parametrize('delete',[False,True])
async def test_cancellation_suppression_stays_visible_and_blocks_deletion(monkeypatch,tmp_path,delete):
    monkeypatch.setattr(manager,'active_agents',{})
    monkeypatch.setattr(manager,'STOP_WAIT_SECONDS',.01)
    state=MagicMock();monkeypatch.setattr(manager,'state_manager',state)
    entered=asyncio.Event();release=asyncio.Event()
    async def work():
        entered.set()
        while not release.is_set():
            try:await release.wait()
            except asyncio.CancelledError:pass
    task=asyncio.create_task(work());await entered.wait()
    manager.active_agents['fixture']={'task':task}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app),base_url='http://manager') as client:
            response=await asyncio.wait_for(
                client.delete('/agents/fixture') if delete else client.post('/agents/fixture/stop'),.5)
            observed=(await client.get('/tasks/fixture/status')).json()
        (tmp_path/'noncooperative-stop.json').write_text(json.dumps({
            'scope':'Local ASGI manager with an asyncio fixture task; no model/native job',
            'operation':'delete' if delete else 'stop','http_status':response.status_code,
            'response':response.json(),'observed':observed},indent=2)+'\n')
        assert response.status_code==(409 if delete else 200)
        assert observed['task_active'] is True and not task.done()
        assert manager.active_agents['fixture']['task'] is task
        state.delete_agent.assert_not_called()
    finally:
        release.set();await task


def test_real_manager_http_keeps_noncooperative_task_until_it_exits(tmp_path):
    """Isolated manager service and bare asyncio task; no DefaultAgent/model job."""
    wrapper=tmp_path/'fixture_manager.py'
    wrapper.write_text('''import asyncio
from contextlib import asynccontextmanager
import main as manager
manager.STOP_WAIT_SECONDS=.02
app=manager.app
release=None
@asynccontextmanager
async def lifespan(app):
    global release
    release=asyncio.Event()
    async def hold():
        try:
            while not release.is_set():
                try:await release.wait()
                except asyncio.CancelledError:pass
        finally:manager.active_agents.pop('fixture',None)
    task=asyncio.create_task(hold())
    manager.active_agents['fixture']={'task':task}
    try:yield
    finally:
        release.set()
        await task
app.router.lifespan_context=lifespan
@app.post('/fixture/release')
async def release_fixture():
    release.set()
    return {'released':True}
''')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    log=(tmp_path/'manager.txt').open('w')
    process=subprocess.Popen([sys.executable,'-m','uvicorn','fixture_manager:app','--app-dir',str(tmp_path),
        '--host','127.0.0.1','--port',str(port)],env=os.environ.copy(),stdout=log,stderr=log)
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=2) as client:
            deadline=time.monotonic()+10
            while True:
                assert process.poll() is None,'Fixture manager exited before health'
                try:
                    if client.get('/health').status_code==200:break
                except httpx.HTTPError:pass
                assert time.monotonic()<deadline,'Fixture manager startup timeout'
                time.sleep(.02)
            before=client.get('/tasks/fixture/status').json()
            started=time.monotonic();stopped=client.post('/agents/fixture/stop')
            stop_elapsed=time.monotonic()-started
            pending=client.get('/tasks/fixture/status').json()
            assert stopped.status_code==200 and stopped.json()['task_active'] is True
            assert pending['task_active'] is True and stop_elapsed<.5
            client.post('/fixture/release').raise_for_status()
            deadline=time.monotonic()+2
            while True:
                after=client.get('/tasks/fixture/status').json()
                if not after['task_active']:break
                assert time.monotonic()<deadline
                time.sleep(.01)
            assert after['task_state']=='absent'
            assert before['instance_id']==pending['instance_id']==after['instance_id']
            (tmp_path/'manager-http.json').write_text(json.dumps({
                'scope':'Isolated actual manager HTTP service with one bare fixture asyncio task; no model/native job',
                'pid':process.pid,'stop_http_status':stopped.status_code,'stop_elapsed_seconds':stop_elapsed,
                'before':before,'after_stop':pending,'after_fixture_release':after},indent=2)+'\n')
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        log.close()
