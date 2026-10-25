#!/usr/bin/env python3
"""Start live rooms for streamers with a Kick Clipping bounty as they go live, within the Spark's limits.

Every 5 minutes it checks which bounty streamers (web/src/kickrates.json) are live, and starts live.py for the best
ones (viewers x rate) up to --max-rooms at once. Each room runs for --minutes, and each streamer is clipped once per
watcher run. A room only starts with --min-avail-gb free, and the newest room is stopped below 10 GB. The shared GPU
ASR is started if it's down. Finished runs get REPORT.md.

Writes pipeline/live/rooms.json (slug -> port, read by the Vite proxy) and web/public/live-rooms.json (room tabs).

Always-on (for launch: the Live page never goes dark): --always-on keeps at least one room on air. When every
bounty streamer has had a room this run, the most-watched live one gets another after --again-min minutes, and
streamers on --fallback (paying streamers on Kick without a bounty) count too. Pair it with --judge off to test
the rotation without editor calls.

usage: rooms_watch.py [--max-rooms 2] [--minutes 45] [--hours 10] [--judge sonnet] [--only n3on,xqc]
                      [--always-on] [--again-min 120] [--fallback slug:usd,slug:usd]
stop:  kill <pid>   (rooms stop cleanly and save their state)
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, '.venv', 'bin', 'python')
REG = os.path.join(HERE, 'rooms.json')
PUB = os.path.join(ROOT, 'runtime', 'live-rooms.json')
RATES = os.path.join(ROOT, 'config', 'kickrates.json')
PORTS = [8820, 8826, 8827, 8821, 8822, 8823, 8824, 8825]   # one per room on air (8830/8831 belong to other projects)
# never clip these: TikTok bans gambling promotion, and it's off-brand
SKIP_CATEGORIES = {'slots & casino', 'gambling', 'slots', 'casino', 'poker'}
UA = 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36'


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


def avail_gb():
    try:
        line = open(os.path.expanduser('~/spark-health/health.log')).read().strip().splitlines()[-1]
        return float(line.split('availGB=')[1].split()[0])
    except Exception:  # noqa: BLE001
        with open('/proc/meminfo') as f:
            kv = {l.split(':')[0]: int(l.split()[1]) for l in f}
        return kv['MemAvailable'] / 1e6


def kick(slug):
    from curl_cffi import requests as rq   # Kick's API wants a browser TLS fingerprint
    j = rq.get(f'https://kick.com/api/v2/channels/{slug}', impersonate='chrome', timeout=15).json()
    ls = j.get('livestream') or {}
    return {'live': bool(ls), 'viewers': ls.get('viewer_count') or 0, 'title': ls.get('session_title') or '',
            'chatroom': (j.get('chatroom') or {}).get('id'), 'name': (j.get('user') or {}).get('username') or slug,
            'category': ((ls.get('categories') or [{}])[0] or {}).get('name') if ls else None}


def asr_up():
    try:
        return urllib.request.urlopen('http://127.0.0.1:8835/health', timeout=3).status == 200
    except Exception:  # noqa: BLE001
        return False


MINE = set()      # the slugs this watcher started: it only ever adds or removes its own entries in the registry


class Adopted:
    """A room a previous watcher started and left running (a restart no longer stops the rooms on air): poll, terminate,
    kill and wait by pid, like the Popen of a room this watcher started itself."""
    def __init__(self, pid):
        self.pid, self.returncode = int(pid), None

    def _running(self):
        try:
            return b'live.py' in open(f'/proc/{self.pid}/cmdline', 'rb').read()   # gone, or the pid went to something else
        except OSError:
            return False

    def poll(self):
        if self.returncode is None and not self._running():
            self.returncode = 0
        return self.returncode

    def terminate(self):
        if self._running():
            os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        if self._running():
            os.kill(self.pid, signal.SIGKILL)

    def wait(self, timeout=None):
        t = time.time()
        while self.poll() is None:
            if timeout is not None and time.time() - t > timeout:
                raise subprocess.TimeoutExpired('live.py', timeout)
            time.sleep(0.5)
        return self.returncode


def _alive(pid):
    """A room process still running: its live.py, not an exited one waiting to be reaped (a zombie answers kill -0, and
    the registry kept a stopped room on the site, 2026-10-21) nor a reused pid."""
    try:
        return b'live.py' in open(f'/proc/{int(pid)}/cmdline', 'rb').read()
    except (OSError, ValueError, TypeError):
        return False


def save(rooms):
    """Write this watcher's rooms into the shared registry, keeping other watchers' live rooms."""
    try:
        reg = json.load(open(REG))
    except (OSError, ValueError):
        reg = {}
    reg = {s: r for s, r in reg.items() if s not in MINE and _alive(r.get('pid'))}
    reg.update({s: {'port': r['port'], 'pid': r['proc'].pid, 'started': r['started'], 'name': r['name'], 'run_log': r['log'], 'rank': r.get('rank', 0)} for s, r in rooms.items()})
    MINE.update(rooms)
    json.dump(reg, open(REG, 'w'), indent=1)
    json.dump({s: {'name': r['name'], 'started': r['started'], 'rank': r.get('rank', 0)} for s, r in reg.items()}, open(PUB, 'w'))


def posting_on(slug):
    """This engine release produces local clips; social publishing is not included."""
    return False


# when each launched room first went on air (its first hour is boosted: live.py --boost)
FIRST = os.path.join(HERE, 'runs', 'launched_first.json')


def _first_sessions():
    try:
        return json.load(open(FIRST))
    except (OSError, ValueError):
        return {}


# rooms launched with a coin on cutroom.fun (launch/server.mjs keeps their Kick streamers here): watched too, and first
LAUNCHED = os.path.join(ROOT, 'config', 'watch.json')


def twitch(slug):
    import fleet
    return fleet.twitch(slug)


# streamers whose live room ended recently: their room's agent keeps editing while they're offline (room_agents.py)
RECENT = os.path.join(HERE, 'runs', 'recent_rooms.json')


def ended_room(slug, platform):
    try:
        d = json.load(open(RECENT))
    except Exception:  # noqa: BLE001
        d = {}
    d[slug] = {'platform': platform, 'ended': time.time()}
    d = {k: v for k, v in d.items() if time.time() - v['ended'] < 48 * 3600}
    json.dump(d, open(RECENT + '.tmp', 'w'), indent=1); os.replace(RECENT + '.tmp', RECENT)


def rank(r):
    """Who gets a live slot first: a room launched with a coin (2), then a --pin streamer (1), then everyone else (0)."""
    return 2 if r.get('launched') else 1 if r.get('pinned') else 0


def launched_slugs():
    try:
        return [s for s in json.load(open(LAUNCHED)) if isinstance(s, str) and s]
    except Exception:  # noqa: BLE001
        return []


def main(a):
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'runtime'), exist_ok=True)
    rates = {r['kick']: r for r in json.load(open(RATES))['rates']}
    pins = [x for x in a.pin.split(',') if x]
    if a.only:                                          # --only narrows the bounty list; --fallback slugs are added after it
        rates = {k: v for k, v in rates.items() if k in a.only.split(',') or k in pins}
    for slug in pins:                                   # pinned: on air whenever they're live (after launched coin rooms)
        rates.setdefault(slug, {'kick': slug, 'usd': 0, 'name': slug, 'program': 'their own clipping program (cutroom.fun/library)'})['pinned'] = True
    for f in filter(None, a.fallback.split(',')):      # "slug:usd" (Kick) or "twitch:login"
        plat = 'twitch' if f.startswith('twitch:') else 'kick'
        slug, _, usd = (f[7:] if plat == 'twitch' else f).partition(':')
        rates.setdefault(slug, {'kick': slug, 'usd': float(usd or 0), 'name': slug, 'fallback': True, 'platform': plat,
                                'program': 'their own clipping program (cutroom.fun/library)'})
    rooms, done, stop_at = {}, {}, time.time() + a.hours * 3600   # done: slug -> when its last room ended
    # rooms still on air from before a restart keep running and are this watcher's now (the user, 2026-10-21: "make sure
    # every room doesn't die"; a restart used to stop every live session)
    try:
        for slug, x in json.load(open(REG)).items():
            pr = Adopted(x.get('pid') or 0)
            if slug not in rooms and x.get('port') in PORTS and pr.poll() is None:
                if a.launched_only and slug not in {e.rpartition(':')[2] for e in launched_slugs()} | set(pins):
                    log(f"{slug}: a test room (no coin) on :{x['port']}: stopping it (--launched-only)")
                    pr.terminate()
                    try:
                        pr.wait(40)                   # it saves its state and ends; then the registry drops it
                    except subprocess.TimeoutExpired:
                        pr.kill()
                    ended_room(slug, x.get('platform') or 'kick'); continue
                rooms[slug] = {'proc': pr, 'port': x['port'], 'started': x.get('started') or time.time(), 'name': x.get('name') or slug,
                               'log': x.get('run_log'), 'rank': x.get('rank', 0)}
                done[slug] = rooms[slug]['started']
                log(f"{slug}: room already on air on :{x['port']}: keeping it")
    except (OSError, ValueError):
        pass
    running = {'on': True}
    signal.signal(signal.SIGTERM, lambda *_: running.update(on=False))
    signal.signal(signal.SIGINT, lambda *_: running.update(on=False))
    save(rooms)
    # the gateway gives every room one address (/live-api/<slug>/...) for the dashboard
    # (its own systemd unit normally runs it: wait out a restart of that before starting a second one, which then holds the
    # port while the unit crash-loops on 'address in use', as on 2026-10-21)
    gw = None
    for i in range(20):
        try:
            urllib.request.urlopen('http://127.0.0.1:8829/status', timeout=2); break
        except urllib.error.HTTPError:
            break                                       # it's up (it answered, with an error)
        except Exception:  # noqa: BLE001
            if i == 19:
                gw = subprocess.Popen([PY, os.path.join(HERE, 'gateway.py')], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                log('started the room gateway on :8829')
            else:
                time.sleep(1)
    log(f'watching {len(rates)} bounty streamers; up to {a.max_rooms} rooms, {a.minutes} min each, for {a.hours} h')
    while running['on'] and time.time() < stop_at:
        # reap finished rooms; a room still running 10 min past its hour is hung (2026-10-19: Clavicular's room said "stream
        # ended; stopping" at 21:49, a stuck thread kept it alive 16 h, and he went uncut all night): stop it, then kill it
        for slug in list(rooms):
            r = rooms[slug]
            if r['proc'].poll() is None and time.time() - r['started'] > (a.minutes + 10) * 60:
                log(f'{slug}: room still running {(time.time() - r["started"]) / 60:.0f} min after starting: stopping it')
                r['proc'].terminate()
                try:
                    r['proc'].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    r['proc'].kill(); r['proc'].wait(timeout=10)
            if r['proc'].poll() is not None:
                log(f'{slug}: room finished (exit {r["proc"].returncode})')
                ended_room(slug, rates.get(slug, {}).get('platform') or 'kick')
                runs = sorted(d for d in os.listdir(os.path.join(HERE, 'runs')) if d.startswith(slug + '-'))
                if runs:
                    subprocess.run([PY, os.path.join(HERE, 'report.py'), os.path.join(HERE, 'runs', runs[-1])], capture_output=True)
                done[slug] = time.time()
                del rooms[slug]
                save(rooms)
        # quiet streams: four editor calls in a row at 3/10 or below (asleep, off cam, AFK) free the slot for someone live
        for slug, r in list(rooms.items()):
            try:
                st = json.load(urllib.request.urlopen(f"http://127.0.0.1:{r['port']}/state", timeout=5))
            except Exception:  # noqa: BLE001
                continue
            scores = []
            for c in st.get('clips', []):
                m = re.match(r'(\d+(?:\.\d+)?)/10', c.get('why') or '')
                if m:
                    scores.append(float(m.group(1)))
            if len(scores) >= 4 and max(scores[-4:]) <= 3 and len(rooms) >= min(a.max_rooms, len(PORTS)):
                log(f'{slug}: last four moments scored {scores[-4:]}: stream looks quiet, freeing the slot')
                r['proc'].terminate()
        # memory guard: never let rooms push the Spark into the red
        gb = avail_gb()
        if gb < 10 and rooms:
            newest = max(rooms, key=lambda s: rooms[s]['started'])
            log(f'only {gb:.0f} GB free: stopping {newest}')
            rooms[newest]['proc'].terminate()
        for entry in launched_slugs():                  # a coin launched since the last loop joins the watch list
            plat, _, slug = entry.rpartition(':')       # "n3on" (Kick) or "twitch:kaicenat"
            rates.setdefault(slug, {'kick': slug, 'usd': 0, 'name': slug, 'program': 'its cutroom.fun coin room'}).update(launched=True, platform=plat or 'kick')
        # a room launched with a coin goes on air the moment its streamer is live: a room that isn't launched hands over the
        # slot now instead of finishing its hour (that is what the coin buys: the agent editing them, live)
        # a --pin streamer does the same to a room that's neither launched nor pinned (the user, 2026-10-18: Clavicular's room
        # page should be the live editor whenever he's live)
        if len(rooms) >= a.max_rooms:
            low = min(rank(rates.get(s, {})) for s in rooms)
            busy = [s for s in rooms if rank(rates.get(s, {})) == low]
            for slug, r in sorted(rates.items(), key=lambda x: -rank(x[1])):
                if rank(r) <= low or slug in rooms:
                    continue
                try:
                    k = twitch(slug) if r.get('platform') == 'twitch' else kick(slug)
                except Exception:  # noqa: BLE001
                    continue
                if k['live'] and (k.get('category') or '').lower() not in SKIP_CATEGORIES:
                    victim = min(busy, key=lambda s: rooms[s]['started'])
                    log(f"{slug}: {'its coin room' if r.get('launched') else 'a pinned room'} is live, so {victim} hands over the slot")
                    rooms[victim]['proc'].terminate()
                    break
        # start rooms for live bounty streamers
        # the limit is the machine, not a fixed count (the user, 2026-10-21: "remove the cap" with Jynxzi live and waiting):
        # a room starts while there's memory (--min-avail-gb) and the CPU isn't nearly full (--max-load, 1-min load average)
        if len(rooms) < min(a.max_rooms, len(PORTS)) and gb >= a.min_avail_gb and os.getloadavg()[0] < a.max_load:
            live = []
            # always-on: with nothing on air, a streamer whose room ended --again-min ago can get another
            again = a.always_on and (not rooms or len(rooms) < min(a.max_rooms, len(PORTS)))   # free slots: no waiting
            cooling = [s for s, r in rates.items() if s not in rooms and s in done and not rank(r) and not (again and time.time() - done[s] > a.again_min * 60)]
            for slug, r in rates.items():
                if slug in rooms or slug in cooling or (a.launched_only and not (r.get('launched') or r.get('pinned'))):
                    continue
                try:
                    k = twitch(slug) if r.get('platform') == 'twitch' else kick(slug)
                except Exception as e:  # noqa: BLE001
                    log(f'{slug}: live check failed ({e})'); continue
                if k['live'] and (k.get('category') or '').lower() not in SKIP_CATEGORIES:
                    # launched rooms first, then pinned, then bounty streamers, then the --fallback backups (Twitch now reports real
                    # viewer counts, so a big Twitch backup would otherwise outrank a paying Kick bounty stream)
                    live.append(((1e12 if r.get('launched') else 1e9 if r.get('pinned') else 1e-6 if r.get('fallback') else 1) * max(k['viewers'], 1) * max(r['usd'], 1), slug, r, k))
            # nothing else is live and nothing is on air: a streamer whose room just ended goes straight back on rather than
            # the site playing a replay while they're live (2026-10-21: Jynxzi and zackrawrr were live, both in their
            # --again-min cooldown, and /live showed a replay for half an hour)
            if again and not live:
                for slug in cooling:
                    r = rates[slug]
                    if a.launched_only and not (r.get('launched') or r.get('pinned')):
                        continue
                    try:
                        k = twitch(slug) if r.get('platform') == 'twitch' else kick(slug)
                    except Exception:  # noqa: BLE001
                        continue
                    if k['live'] and (k.get('category') or '').lower() not in SKIP_CATEGORIES:
                        live.append(((1e-6 if r.get('fallback') else 1) * max(k['viewers'], 1) * max(r['usd'], 1), slug, r, k))
            # at most two new rooms a minute: each start is a burst (stream, chat, transcriber, editor) and the site shares the machine
            for _, slug, r, k in sorted(live, reverse=True)[: min(2, min(a.max_rooms, len(PORTS)) - len(rooms))]:
                if not asr_up():
                    log('shared ASR unavailable; using the CPU fallback')
                port = next(p for p in PORTS if p not in {x['port'] for x in rooms.values()})
                lf = os.path.join(HERE, 'runs', f'{slug}-watch-{time.strftime("%Y%m%d-%H%M")}.log')
                # --end-offline: when the stream drops, the room ends after 2 min and the slot goes to someone live
                cmd = [PY, os.path.join(HERE, 'live.py'), slug, '--port', str(port), '--minutes', str(a.minutes), '--end-offline', '2',
                       *(['--chatroom', str(k['chatroom'])] if k.get('chatroom') else []), '--platform', r.get('platform') or 'kick',
                       '--rate', str(r['usd']), '--program', r.get('program') or 'Kick Clipping (clipping.net) bounty', '--judge', a.judge, '--min-score', '3', '--cooldown', '60', '--asr-live', a.asr_live]
                if a.op_token:
                    cmd += ['--op-token', a.op_token]
                if posting_on(slug):                    # this streamer's account is connected AND past its warm-up (the user's call)
                    cmd += ['--post', 'on']                 # no #ad: the user, 2026-10-18 ("this isn't an ad")
                if r.get('launched'):                        # a new coin room's first hour cuts more freely (live.boosting)
                    first = _first_sessions()
                    t0 = first.setdefault(slug, time.time())
                    left = 60 - (time.time() - t0) / 60
                    if left > 5:
                        cmd += ['--boost', f'{left:.0f}']
                    json.dump(first, open(FIRST, 'w'))
                p = subprocess.Popen(cmd, cwd=HERE, stdout=open(lf, 'w'), stderr=subprocess.STDOUT)
                rooms[slug] = {'proc': p, 'port': port, 'started': time.time(), 'name': k['name'], 'log': lf, 'rank': rank(r)}
                done[slug] = time.time()
                save(rooms)
                log(f'{slug}: live with {k["viewers"]:,} viewers (${r["usd"]}/100K): room on :{port}, {a.minutes} min')
        for _ in range(a.every):
            if not running['on'] or any(r['proc'].poll() is not None for r in rooms.values()):
                break                                   # a room just ended: hand the slot to someone live right away
            time.sleep(1)
    if os.environ.get('CUTROOM_STOP_ROOMS') != '1':
        # the rooms stay on air and in the registry: the next watcher keeps them (systemd: KillMode=process)
        log(f"stopping; leaving {len(rooms)} room(s) on air for the next watcher: {', '.join(rooms) or 'none'}")
        if gw:
            gw.terminate()
        return
    log('stopping rooms')
    for r in rooms.values():
        r['proc'].terminate()
    for r in rooms.values():
        try:
            r['proc'].wait(30)
        except subprocess.TimeoutExpired:
            r['proc'].kill()
    rooms.clear(); save(rooms)
    if gw:
        gw.terminate()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-rooms', type=int, default=2)
    ap.add_argument('--launched-only', action='store_true', help='rooms only for streamers with a launched coin (and --pin): no test rooms (the user, 2026-10-21)')
    ap.add_argument('--max-load', type=float, default=18.0, help="don't start a room while the 1-minute load average is above this (20 cores)")
    ap.add_argument('--minutes', type=float, default=45)
    ap.add_argument('--hours', type=float, default=10)
    ap.add_argument('--min-avail-gb', type=float, default=20)
    ap.add_argument('--every', type=int, default=300, help='seconds between live checks')
    ap.add_argument('--judge', default='sonnet')
    ap.add_argument('--asr-live', default='cpu', choices=['cpu', 'gpu'], help='gpu: fresher transcript for one room (the live demo)')
    ap.add_argument('--only', default='')
    ap.add_argument('--op-token', default=os.environ.get('CUTROOM_OP'))
    ap.add_argument('--always-on', action='store_true', help='keep at least one room on air (launch mode)')
    ap.add_argument('--again-min', type=float, default=120, help='always-on: minutes before a streamer can get another room')
    ap.add_argument('--pin', default='', help='comma-separated Kick slugs on air whenever they are live (after launched coin rooms)')
    ap.add_argument('--fallback', default='', help='always-on: extra Kick slugs (slug:usd per 100K) when no bounty streamer is live')
    main(ap.parse_args())
