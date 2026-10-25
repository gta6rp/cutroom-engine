#!/usr/bin/env python3
"""Read-only local API for per-streamer agents and active room progress.

GET /status, /agents, /agent/<slug>, /<slug>/state, /<slug>/events
No website, token launch, social account, posting or public tunnel integration.
"""
import argparse
import json
from pathlib import Path
import re

import aiohttp
from aiohttp import web

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REG = HERE / 'rooms.json'
RATES = ROOT / 'config/kickrates.json'
SLUG = re.compile(r'[a-zA-Z0-9_-]{1,80}')
PORTS = frozenset(range(8820, 8828))
HTTP = web.AppKey('http', aiohttp.ClientSession)


def load(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def room_slugs():
    return set(load(REG, {})) | {p.stem for p in (HERE / 'agents').glob('*.json')}


def slug_for(request):
    slug = request.match_info['slug']
    if not SLUG.fullmatch(slug):
        raise web.HTTPBadRequest(text='invalid room slug')
    return slug


async def status(request):
    return web.json_response({'rooms': load(REG, {}), 'agents': sorted(room_slugs())})


async def agents(request):
    return web.json_response({'agents': [load(p, {}) for p in sorted((HERE / 'agents').glob('*.json'))]})


async def agent(request):
    data = load(HERE / 'agents' / f'{slug_for(request)}.json', None)
    if data is None:
        raise web.HTTPNotFound(text='no agent state for this room')
    return web.json_response(data)


async def progress(request):
    slug = slug_for(request)
    entry = load(REG, {}).get(slug, {})
    port = entry.get('port')
    if type(port) is not int or port not in PORTS:
        raise web.HTTPNotFound(text='room is not running')
    endpoint = request.match_info['endpoint']
    if endpoint not in {'state', 'events'}:
        raise web.HTTPNotFound()
    try:
        async with request.app[HTTP].get(f'http://127.0.0.1:{port}/{endpoint}') as upstream:
            headers = {'Content-Type': upstream.headers.get('Content-Type', 'application/json'),
                       'Cache-Control': 'no-store'}
            response = web.StreamResponse(status=upstream.status, headers=headers)
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            return response
    except aiohttp.ClientError:
        raise web.HTTPBadGateway(text='room service is unavailable')


async def app_factory():
    app = web.Application()
    app[HTTP] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=5))
    app.router.add_get('/status', status)
    app.router.add_get('/agents', agents)
    app.router.add_get('/agent/{slug}', agent)
    app.router.add_get('/{slug}/{endpoint}', progress)

    async def close(application):
        await application[HTTP].close()
    app.on_cleanup.append(close)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8829)
    web.run_app(app_factory(), host='127.0.0.1', port=parser.parse_args().port)
