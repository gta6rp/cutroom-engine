"""Offline integration checks: no capture, paid editor calls, model loading or posting."""
import argparse
import ast
import asyncio
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))
sys.path.insert(0, str(ROOT / 'pipeline/live'))
import fleet
import gateway
import live


class SourceTests(unittest.TestCase):
    def test_all_python_source_parses(self):
        paths = list(ROOT.rglob('*.py'))
        for path in paths:
            if '.venv' not in path.parts:
                with self.subTest(path=path.relative_to(ROOT)):
                    ast.parse(path.read_text(), filename=str(path))

    def test_core_modules_import_without_starting_services(self):
        for name in ('judge', 'semantic', 'quota', 'fleet', 'room_agents', 'rooms_watch',
                     'platforms', 'livelog', 'layouts', 'edit', 'catalog', 'safety', 'vcam'):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_cli_entrypoints(self):
        for name in ('live.py', 'fleet.py', 'rooms_watch.py', 'room_agents.py', 'gateway.py'):
            with self.subTest(entrypoint=name):
                result = subprocess.run([sys.executable, str(ROOT / 'pipeline/live' / name), '--help'],
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)

    def test_empty_configuration(self):
        self.assertEqual(json.loads((ROOT / 'config/kickrates.json').read_text()), {'rates': []})
        for name in ('watch.json', 'lore.json', 'library-index.json'):
            self.assertEqual(json.loads((ROOT / 'config' / name).read_text()), [])

    def test_fleet_creates_independent_room_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rates = root / 'rates.json'; rates.write_text('{"rates": []}')
            with patch.multiple(fleet, HERE=str(root / 'live'), ROOT=str(root),
                                AG=str(root / 'agents'), RATES=str(rates)):
                controller = fleet.Fleet(argparse.Namespace(rooms='example_a,twitch:example_b', first=''))
                self.assertEqual(set(controller.rooms), {'example_a', 'example_b'})
                self.assertEqual(controller.rooms['example_b'].platform, 'twitch')
                for room in controller.rooms.values():
                    room.save()
                    self.assertEqual(json.loads(Path(room.path).read_text())['slug'], room.slug)
                controller.rooms['example_a'].s['cuts'].append({'id': 'local-clip'})
                self.assertEqual(controller.rooms['example_b'].s['cuts'], [])


class LocalRoomTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.object(live, 'HERE', self.tmp.name)
        self.patch.start()
        args = argparse.Namespace(slug='example', minutes=1, model='small.en', platform='kick',
                                  chatroom=None, program='', rate=None, cap=None, min_views=None,
                                  program_url='', min_score=4, semantic_every=0)
        self.room = live.Live(args)

    async def asyncTearDown(self):
        self.room.log_f.close()
        self.patch.stop()
        self.tmp.cleanup()

    async def test_state_api_without_stream_or_model(self):
        app = web.Application()
        app.router.add_get('/state', self.room.h_state)
        async with TestClient(TestServer(app)) as client:
            response = await client.get('/state')
            state = await response.json()
            self.assertEqual(response.status, 200)
            self.assertEqual(state['meta']['slug'], 'example')
            self.assertEqual(state['clips'], [])
            self.assertEqual(state['T'], 0)

    async def test_disabled_semantic_scan_returns_without_editor(self):
        await asyncio.wait_for(self.room.semantic(), timeout=1)

    async def test_clips_remain_local(self):
        clip = {'id': 'example-clip', 'stage': 'ready'}
        with patch.object(live.subprocess, 'run', side_effect=AssertionError('external process')):
            await self.room.post(clip)
            await self.room.api_post(clip)
            self.room.to_kit(clip, 'review')
            await self.room.day_pick()
        self.assertEqual(clip['stage'], 'ready')
        self.assertIn('ready locally', clip['post']['status'])


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'agents').mkdir()
        (self.root / 'agents/example.json').write_text('{"slug":"example", "mode":"offline"}')
        self.patch = patch.multiple(gateway, HERE=self.root, REG=self.root / 'rooms.json')
        self.patch.start()
        self.client = TestClient(TestServer(await gateway.app_factory()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.patch.stop()
        self.tmp.cleanup()

    async def test_agent_and_status(self):
        response = await self.client.get('/agents')
        self.assertEqual((await response.json())['agents'][0]['slug'], 'example')
        response = await self.client.get('/agent/example')
        self.assertEqual((await response.json())['mode'], 'offline')
        response = await self.client.get('/status')
        self.assertEqual((await response.json())['rooms'], {})

    async def test_missing_room_and_invalid_registry_target(self):
        for entry in ({}, {'example': {'port': 8888}}, {'example': {'port': '8820'}}):
            (self.root / 'rooms.json').write_text(json.dumps(entry))
            response = await self.client.get('/example/state')
            self.assertEqual(response.status, 404)
        response = await self.client.get('/agent/bad.slug')
        self.assertEqual(response.status, 400)

    async def test_gateway_proxies_actual_room_state_and_events(self):
        upstream = web.Application()
        async def state(request):
            return web.json_response({'slug': 'example', 'T': 12})
        upstream.router.add_get('/state', state)

        async def events(request):
            return web.Response(text='event: tick\ndata: {"T":12}\n\n', content_type='text/event-stream')
        upstream.router.add_get('/events', events)
        async with TestServer(upstream) as server:
            (self.root / 'rooms.json').write_text(json.dumps({'example': {'port': server.port}}))
            with patch.object(gateway, 'PORTS', {server.port}):
                response = await self.client.get('/example/state')
                self.assertEqual(await response.json(), {'slug': 'example', 'T': 12})
                response = await self.client.get('/example/events')
                self.assertTrue(response.headers['Content-Type'].startswith('text/event-stream'))
                self.assertIn('event: tick', await response.text())


if __name__ == '__main__':
    unittest.main()
