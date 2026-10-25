#!/usr/bin/env python3
"""Room agents. Launched rooms (and any streamer whose live room ended in the last day) get one: when a coin launches on cutroom.fun (launch/server.mjs), its streamers land in
launch/data/<network>/watch.json, and this keeps one fleet.py running for all of them: it digs each streamer's
most-clipped past moments (Claude is the editor), cuts the strong ones, makes studio edits, and writes its activity
to agents/<slug>.json, which the room page shows live (gateway /agent/<slug>). Going live is rooms_watch.py's job
(it also reads watch.json and puts launched rooms first), so the fleet here runs with --max-live 0.

usage: room_agents.py [--network mainnet] [--every 20] [--always clavicular,n3on]   (--always: rooms that keep an agent whatever happens)
"""
import argparse, json, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, '.venv', 'bin', 'python')


def slugs(path):
    try:
        return sorted({s for s in json.load(open(path)) if isinstance(s, str) and s})
    except Exception:  # noqa: BLE001
        return []


RECENT = os.path.join(HERE, 'runs', 'recent_rooms.json')


def recent(hours=24):
    """Streamers whose live room ended in the last day (rooms_watch.py): while they're offline their room keeps editing
    (evergreen edits of their most-clipped moments on trending sounds, rankings), newest first."""
    try:
        d = json.load(open(RECENT))
    except Exception:  # noqa: BLE001
        return []
    keep = sorted(((v['ended'], k, v.get('platform')) for k, v in d.items() if time.time() - v['ended'] < hours * 3600), reverse=True)
    return [f'twitch:{k}' if p == 'twitch' else k for _, k, p in keep]


def digging(rooms):
    """Is the fleet in the middle of a dig (an edit may be rendering)? Then a restart waits for it."""
    for e in rooms:
        try:
            if json.load(open(os.path.join(HERE, 'agents', f"{e.split(':')[-1]}.json"))).get('mode') == 'digging':
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--network', default='mainnet')
    ap.add_argument('--every', type=int, default=20)
    ap.add_argument('--always', default='', help='comma-separated rooms that always get an agent (the user, 2026-10-18: clavicular, n3on)')
    a = ap.parse_args()
    watch = os.path.join(ROOT, 'config', 'watch.json')
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    proc, current, waited = None, [], 0
    stop = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    while not stop['now']:
        rec = recent()
        want = sorted(set(slugs(watch)) | set(rec) | {x.strip() for x in a.always.split(',') if x.strip()})
        # the running fleet adds and drops rooms itself (fleet.sync reads this file every tick): a launch no longer restarts it
        try:
            tmp = os.path.join(HERE, 'runs', 'agent_rooms.json.tmp'); json.dump(want, open(tmp, 'w')); os.replace(tmp, os.path.join(HERE, 'runs', 'agent_rooms.json'))
        except OSError:
            pass
        if want != current and proc and proc.poll() is None:
            current = want; time.sleep(a.every); continue
        if want != current and proc and proc.poll() is None and digging(current) and waited < 20 * 60:
            waited += a.every; time.sleep(a.every); continue          # let the running dig finish its edit first
        waited = 0
        if want != current or (proc and proc.poll() is not None):
            first = next((x for x in rec if x not in current), rec[0] if rec else '').split(':')[-1]   # who just went offline
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
            proc, current = None, want
            if want:
                log = os.path.join(HERE, 'runs', f'room-agents-{time.strftime("%Y%m%d-%H%M%S")}.log')
                print(f'{time.strftime("%H:%M:%S")} room agents for: {", ".join(want)} (log {log})', flush=True)
                proc = subprocess.Popen(['nice', '-n', '10', PY, os.path.join(HERE, 'fleet.py'), '--rooms', ','.join(want),
                                         '--max-live', '0', '--dig', 'on', '--dig-every', '600', *(['--first', first] if first else [])],
                                        cwd=HERE, stdout=open(log, 'a'), stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(a.every)
    if proc and proc.poll() is None:
        proc.terminate()


if __name__ == '__main__':
    main()
