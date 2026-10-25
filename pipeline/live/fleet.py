#!/usr/bin/env python3
"""Always-on rooms: every room is either watching its streamer live or digging through their past streams.

Every --poll seconds (default 5 min) each room checks whether its streamer is live on Kick.
  live    -> the room's live session (live.py) watches the whole stream and cuts as it goes; it stops by itself
             once the stream has been offline for 10 minutes. At most --max-live sessions run at once.
  offline -> the room digs, one room at a time across the fleet: it grows its catalog (catalog.py) by a batch of
             the moments the streamer's viewers clipped most (Kick clips point into the VODs), scored in one editor
             call, then cuts the best uncut moment (clipcut.py). Once a theme (funny, wild, ...) has five strong
             moments, it makes that ranking from the catalog (catalog.py --rank), at most one a week.
Gambling streams and clips are skipped (TikTok bans gambling promotion).

Each room's activity is written to pipeline/live/agents/<slug>.json (mode, last check, what it's doing, a log, its
cuts); the gateway serves it at /agent/<slug> for the room page, and rooms.py copies it into the site build.

usage: fleet.py --rooms n3on,xqc,asmongold [--poll 300] [--max-live 2] [--judge sonnet] [--dig on|off] [--dig-every 600]
       fleet.py --rooms xqc --once --dig off      one check, no editor calls (test)
stop:  kill <pid>   (live sessions stop cleanly and save their state)
"""
import argparse, glob, json, os, re, signal, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quota  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, '.venv', 'bin', 'python')
AG = os.path.join(HERE, 'agents')
YTDLP = os.path.join(os.path.dirname(os.path.dirname(HERE)), '.venv', 'bin', 'yt-dlp')
CLIPS = os.path.join(ROOT, 'portfolio', 'clips')
RATES = os.path.join(ROOT, 'config', 'kickrates.json')
REG = os.path.join(HERE, 'rooms.json')
PUB = os.path.join(ROOT, 'runtime', 'live-rooms.json')
PORTS = [8820, 8826, 8827]
GAMBLING = {'slots & casino', 'gambling', 'slots', 'casino', 'poker'}
ALIAS = {}                         # per-streamer overrides if a handle is taken (the N3on room is @cutroomn3on)


def handle(slug, name=None):
    """The room's account name on TikTok / Instagram / YouTube: cutroom + the streamer."""
    return 'cutroom' + re.sub(r'[^a-z0-9]', '', (ALIAS.get(slug) or name or slug).lower())


def twitch(slug):
    """The same shape as kick(), for rooms launched for Twitch streamers: Twitch's public GraphQL (livelog.twitch_gql: the
    real viewer count and the category, so a slots stream is skipped like on Kick), else yt-dlp, whose viewer count is 0."""
    try:
        import livelog
        t = livelog.twitch_gql([slug]).get(slug.lower())
        if t is not None:
            return {'live': t['live'], 'viewers': t['viewers'], 'title': t['title'] if t['live'] else '', 'chatroom': None,
                    'name': t['name'] or slug, 'category': t['category'] if t['live'] else None}
    except Exception:  # noqa: BLE001
        pass
    r = subprocess.run([YTDLP, '--no-warnings', '-J', '--skip-download', f'https://www.twitch.tv/{slug}'], capture_output=True, text=True, timeout=60)
    try:
        d = json.loads(r.stdout or 'null') or {}
    except ValueError:
        d = {}
    live = bool(d.get('is_live'))
    return {'live': live, 'viewers': d.get('concurrent_view_count') or 0, 'title': (d.get('description') or d.get('title') or '') if live else '',
            'chatroom': None, 'name': d.get('uploader') or slug, 'category': None}


def chan(r):
    return twitch(r.slug) if r.platform == 'twitch' else kick(r.slug)


def kick(slug):
    from curl_cffi import requests as rq
    j = rq.get(f'https://kick.com/api/v2/channels/{slug}', impersonate='chrome', timeout=15).json()
    ls = j.get('livestream') or {}
    return {'live': bool(ls), 'viewers': ls.get('viewer_count') or 0, 'title': ls.get('session_title') or '',
            'chatroom': (j.get('chatroom') or {}).get('id'), 'name': (j.get('user') or {}).get('username') or slug,
            'category': ((ls.get('categories') or [{}])[0] or {}).get('name') if ls else None}


def viewer_clips(slug):
    """The streamer's most-viewed viewer clips, this week then this month, minus gambling and odd lengths."""
    from curl_cffi import requests as rq
    out, seen = [], set()
    for tm in ('week', 'month'):
        try:
            r = rq.get(f'https://kick.com/api/v2/channels/{slug}/clips?sort=view&time={tm}', impersonate='chrome', timeout=20)
            cl = r.json().get('clips') or [] if r.status_code == 200 else []
        except Exception:  # noqa: BLE001
            cl = []
        for c in sorted(cl, key=lambda c: -(c.get('view_count') or 0)):
            cat = ((c.get('category') or {}).get('name') or '').lower()
            if c['id'] in seen or cat in GAMBLING or not 12 <= (c.get('duration') or 0) <= 150:
                continue
            seen.add(c['id']); out.append(c)
        time.sleep(1)
    return out


def avail_gb():
    try:
        line = open(os.path.expanduser('~/spark-health/health.log')).read().strip().splitlines()[-1]
        return float(line.split('availGB=')[1].split()[0])
    except Exception:  # noqa: BLE001
        kv = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
        return kv['MemAvailable'] / 1e6


def visual_ok(path, n=6):
    """1 when a clip's picture reads on a phone (not dark, not blurry), 0 otherwise; 0.5 when it can't be checked."""
    if not path or not os.path.exists(path):
        return 0.5
    import cv2, numpy as np
    c = cv2.VideoCapture(path); N = c.get(7) or 0; lum, sh = [], []
    for k in range(n):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + 0.5) / n)); ok, f = c.read()
        if ok:
            g = cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2GRAY)
            lum.append(float(g.mean())); sh.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    if not lum:
        return 0.5
    return 1 if np.median(lum) >= 55 and np.median(sh) >= 40 else 0


def picture_q(path, n=12):
    """0-1: how well a clip's picture will edit: light, sharpness and a steady camera (the helicopter source, our first
    8/10 hype edit: bright, Laplacian ~5400, static camera; handheld dusk footage capped at 6 however it was cut).
    0.5 when it can't be checked."""
    if not path or not os.path.exists(path):
        return 0.5
    import cv2, numpy as np
    c = cv2.VideoCapture(path); N = c.get(7) or 0; fps = c.get(5) or 30
    lum, sh, mov = [], [], []
    for k in range(n):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + 0.5) / n)); ok, a = c.read(); ok2, b = c.read()
        if not (ok and ok2):
            continue
        ga = cv2.cvtColor(cv2.resize(a, (320, 180)), cv2.COLOR_BGR2GRAY); gb = cv2.cvtColor(cv2.resize(b, (320, 180)), cv2.COLOR_BGR2GRAY)
        f = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mov.append(float(np.median(np.hypot(f[..., 0], f[..., 1]))) * fps / 30 > 1.5)    # the whole frame moving: the camera
        lum.append(float(ga.mean())); sh.append(float(cv2.Laplacian(cv2.cvtColor(cv2.resize(a, (640, 360)), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
    if not lum:
        return 0.5
    light = np.clip((np.median(lum) - 40) / 60, 0, 1)
    sharp = np.clip((np.log10(max(1.0, np.median(sh))) - 2.5) / 1.0, 0, 1)
    steady = 1 - np.clip(np.mean(mov) / 0.5, 0, 1)
    return round(float((light + sharp + steady) / 3), 3)


class Room:
    def __init__(self, slug, rate):
        self.slug, self.rate = slug, rate
        self.platform = rate.get('platform') or 'kick'                   # 'twitch' for rooms launched for Twitch streamers
        self.home = f"{'twitch.tv' if self.platform == 'twitch' else 'kick.com'}/{slug}"
        self.path = os.path.join(AG, f'{slug}.json')
        old = json.load(open(self.path)) if os.path.exists(self.path) else {}
        self.s = {'slug': slug, 'name': old.get('name') or rate.get('name') or slug, 'handle': handle(slug, rate.get('name')),
                  'mode': 'starting', 'live': False, 'viewers': 0, 'title': '', 'checked': None, 'doing': '', 'since': time.time(),
                  'session': old.get('session'), 'last_dig': old.get('last_dig'), 'last_rank': old.get('last_rank'),
                  'dug': old.get('dug', {}), 'cuts': old.get('cuts', []), 'log': old.get('log', [])}
        self.proc = None

    def log(self, msg):
        self.s['log'] = (self.s['log'] + [{'t': time.time(), 'msg': msg}])[-80:]
        print(time.strftime('%H:%M:%S'), f'{self.slug}: {msg}', flush=True)

    def save(self):
        os.makedirs(AG, exist_ok=True)
        tmp = self.path + '.tmp'
        json.dump(self.s, open(tmp, 'w'), indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)


class Fleet:
    def __init__(self, a):
        os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
        os.makedirs(os.path.join(ROOT, 'runtime'), exist_ok=True)
        self.a = a
        rates = {r['kick']: r for r in json.load(open(RATES))['rates']}
        self.rates = rates
        self.rooms = {}
        for s in filter(None, a.rooms.split(',')):                        # "slug" (Kick) or "twitch:slug"
            self.add_room(s)
        self.digging = None          # (slug, thread): the shared dig, for rooms without a coin
        self.coin_digs = {}          # slug -> thread: a coin room's own agent (schedule_coin_digs)
        self.studio_lock = threading.Lock()   # one archive re-edit render at a time across the fleet (the Spark's CPU)
        if a.first in self.rooms:   # the streamer who just went offline: their past streams get edited first
            self.rooms[a.first].s['last_dig'] = 0
        self.running = True

    def add_room(self, s):
        plat, _, slug = s.rpartition(':')
        rates = self.rates
        self.rooms[slug] = Room(slug, {**rates.get(slug, {'kick': slug, 'usd': 0, 'name': slug}), 'platform': plat or 'kick'} if plat == 'twitch'
                                else rates.get(slug, {'kick': slug, 'usd': 0, 'name': slug}))
        r = self.rooms[slug]
        if r.s.get('name') in (None, '', slug):          # a Twitch streamer without a bounty entry: their name, not their login
            try:
                L = json.load(open(os.path.join(ROOT, 'config', 'library-index.json')))
                e = next((x for x in L if slug in ((x.get('handles') or {}).get('twitch'), (x.get('handles') or {}).get('kick'))), None)
                if e:
                    r.s['name'] = re.sub(r'\s*\(.*\)$', '', e['name'])
            except (OSError, ValueError):
                pass
        return r

    def sync(self):
        """The rooms room_agents.py wants (runs/agent_rooms.json), added and dropped while running: a coin launched mid-day used to
        restart the whole fleet, every agent showed 'Paused' and a dig in progress was cut off; runs/agent_rooms.json (the user, 2026-10-21: "we need
        every agent for each coin to not be at risk of pausing")."""
        try:
            want = [x for x in json.load(open(os.path.join(HERE, 'runs', 'agent_rooms.json'))) if isinstance(x, str) and x]
        except (OSError, ValueError):
            return
        slugs = {x.rpartition(':')[2] for x in want}
        for s in want:
            if s.rpartition(':')[2] not in self.rooms:
                r = self.add_room(s); r.s['last_dig'] = 0                # a new room digs soon
                r.s.update(mode='offline', doing='offline: checking for a live stream every 5 minutes, digging past streams in between'); r.save()
                print(f'{time.strftime("%H:%M:%S")} fleet: added {s}', flush=True)
        for slug in [x for x in self.rooms if x not in slugs]:
            r = self.rooms[slug]
            if not r.proc and not (self.digging and self.digging[0] == slug and self.digging[1].is_alive()):
                del self.rooms[slug]
                print(f'{time.strftime("%H:%M:%S")} fleet: dropped {slug}', flush=True)

    # ---------- live ----------
    def start_live(self, r, k):
        used = {x.s.get('port') for x in self.rooms.values() if x.proc and x.proc.poll() is None}
        port = next((p for p in PORTS if p not in used), None)
        if port is None:
            return
        lf = os.path.join(HERE, 'runs', f'{r.slug}-fleet-{time.strftime("%Y%m%d-%H%M")}.log')
        cmd = [PY, os.path.join(HERE, 'live.py'), r.slug, '--port', str(port), '--minutes', '720', '--end-offline', '10',
               *(['--chatroom', str(k['chatroom'])] if k.get('chatroom') else []), '--platform', r.platform,
               '--rate', str(r.rate.get('usd') or 0), '--judge', self.a.judge, '--min-score', '3', '--cooldown', '60',
               '--program', r.rate.get('program') or 'Kick Clipping (clipping.net) bounty']
        r.proc = subprocess.Popen(cmd, cwd=HERE, stdout=open(lf, 'w'), stderr=subprocess.STDOUT)
        r.s.update(mode='live', port=port, since=time.time(), session={'started': time.time(), 'log': lf},
                   doing=f'watching {k["name"]} live with {k["viewers"]:,} viewers: hearing every word, reading chat, cutting as it goes')
        r.log(f'{k["name"]} is live ({k["viewers"]:,} watching, "{k["title"][:60]}"): watching the whole stream')
        self.registry()

    def registry(self):
        """This fleet's live rooms into the shared registry, KEEPING the rooms other processes run (rooms_watch.py's live
        room): written whole, it erased the room on air whenever the room agents (--max-live 0) restarted, and the room
        pages stopped showing the live board (2026-10-17)."""
        on = {s: r for s, r in self.rooms.items() if r.proc and r.proc.poll() is None}
        try:
            reg = json.load(open(REG))
        except (OSError, ValueError):
            reg = {}

        def alive(pid):
            try:
                os.kill(int(pid), 0); return True
            except (OSError, ValueError, TypeError):
                return False
        # drop only this fleet's own (or dead) entries: a streamer with an agent here can still be on air in rooms_watch.py's
        # room, and dropping every slug this fleet follows erased those live rooms (2026-10-21: /live showed a replay while
        # Asmongold and Jynxzi were on air)
        mine = {r.proc.pid for r in self.rooms.values() if r.proc}
        reg = {s: x for s, x in reg.items() if alive(x.get('pid')) and x.get('pid') not in mine}
        reg.update({s: {'port': r.s['port'], 'pid': r.proc.pid, 'started': r.s['session']['started'], 'name': r.s['name'], 'run_log': r.s['session']['log']}
                    for s, r in on.items()})
        json.dump(reg, open(REG, 'w'), indent=1)
        json.dump({s: {'name': x['name'], 'started': x['started']} for s, x in reg.items()}, open(PUB, 'w'))

    # ---------- dig ----------
    def dig(self, r):
        """One dig while the streamer is offline: grow the room's catalog by a batch of past moments (their viewers'
        most-clipped, transcribed and scored in one editor call), cut the best uncut one, and make a ranking once a theme
        has five strong moments."""
        sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
        import catalog
        try:
            r.s.update(mode='digging', doing=f"reading {r.s['name']}'s past streams: downloading the moments viewers clipped most, hearing every word, and scoring each one", since=time.time())
            W = r.s['work'] = {'step': 'find', 'started': time.time(), 'moments': [], 'pick': None, 'result': None, 'updated': time.time()}
            r.save()

            def say(ev, data):
                if ev == 'batch':
                    W['moments'] = [{**x, 'status': 'fetching'} for x in data]
                elif ev == 'heard':
                    W['step'] = 'hear'
                    for x in W['moments']:
                        if x['id'] == data['id']:
                            x.update(status='heard', words=data.get('words'))
                elif ev == 'scoring':
                    W['step'] = 'score'
                    for x in W['moments']:
                        if x['id'] in data:
                            x['status'] = 'scoring'
                elif ev == 'scored':
                    for s_ in data:
                        for x in W['moments']:
                            if x['id'] == s_['id']:
                                x.update(status='scored', score=s_['score'], label=s_['label'], hard_no=s_['hard_no'])
                W['updated'] = time.time(); r.save()
            n0 = len(r.s['dug'])
            cat = catalog.build(r.slug, r.s['name'], r.platform, 80, self.a.judge, 8, limit=8, progress=say)
            for m in cat['moments'].values():
                if 'score' in m and m['id'] not in r.s['dug']:
                    r.s['dug'][m['id']] = {'score': m['score'], 'tags': m.get('tags'), 'label': m.get('label'), 'hard_no': m.get('hard_no'), 'at': time.time()}
            if len(r.s['dug']) > n0:                     # a dig that read nothing new isn't news: it only filled the room's log
              r.log(f"read {len(r.s['dug']) - n0} past moments: " + (', '.join(f"{m['label']} ({m['score']}/10)" for m in sorted(
                [m for m in cat['moments'].values() if m['id'] in r.s['dug'] and r.s['dug'][m['id']]['at'] > time.time() - 900 and m.get('score')],
                key=lambda m: -m['score'])[:3]) or 'nothing strong'))
            cut_ids = {x.get('clip') for x in r.s['cuts']}
            # the daily minimum (quota.py): while the streamer's day is short of it, a solid 5/10 past moment is worth a cut
            low = quota.behind(r.slug)
            bar = 6 if low else 7
            # among equals, a picture a phone can read first: dark or blurry viewer clips cap our edits at ~6/10 (critic.py)
            # what the streamer's own viewers clipped and watched counts too: the editor's score barely tracked views
            # (Spearman ~0 for N3on/xQc/Asmongold), so the most-viewed half of the catalog gets up to +1 (never a penalty:
            # fresh clips haven't had time to collect views)
            vs = sorted(m.get('views') or 0 for m in cat['moments'].values())
            # and the picture: a sharp, bright, steady source edits up to 8; a dark or handheld one stalls at 6 (picture_q, +-0.75)
            pq = {}
            def blend(m):
                pct = sum(1 for v in vs if v <= (m.get('views') or 0)) / max(1, len(vs))
                if m['id'] not in pq:
                    pq[m['id']] = picture_q(m.get('low')) if (m.get('score') or 0) >= bar - 1.75 else 0.5
                return (m.get('score') or 0) + max(0.0, 2 * (pct - 0.5)) + 1.5 * (pq[m['id']] - 0.5)
            best = sorted([m for m in cat['moments'].values() if blend(m) >= bar and not m.get('hard_no') and m['id'] not in cut_ids],
                          key=lambda m: (-round(blend(m), 1), -m['views']))
            if best:
                m = best[0]
                W.update(step='cut', pick={'id': m['id'], 'label': m.get('label'), 'score': m.get('score'), 'views': m.get('views'), 'date': m.get('date')}, updated=time.time())
                cid = f"{r.slug}-{m['id'][-6:].lower()}"
                r.s['doing'] = f'cutting "{m["label"]}" from the {m.get("date") or "past"} stream ({m["views"]:,} views on the viewer clip)'
                r.log(r.s['doing']); r.save()
                job = {'id': cid, 'streamer': r.s['name'], 'slug': r.slug, 'platform': r.platform, 'url': m['url'], 'title': m['title'], 'views': m['views'],
                       'program': r.rate.get('program') or ('Kick Clipping (clipping.net) bounty' if r.platform == 'kick' else "the room's own coin room"), 'source_url': m['source_url'], 'credit': r.home, 'date': m.get('date')}
                jf = os.path.join(AG, r.slug, f'{cid}.json'); os.makedirs(os.path.dirname(jf), exist_ok=True)
                json.dump([job], open(jf, 'w'), indent=1)
                subprocess.run(['nice', '-n', '10', PY, os.path.join(ROOT, 'pipeline', 'clipcut.py'), jf, '--judge', self.a.judge],
                               capture_output=True, text=True, timeout=1800, env={**os.environ, **({'CUTROOM_KEEP_MIN': '5'} if low else {})})
                ej = os.path.join(CLIPS, cid, 'editor.json')
                v = json.load(open(ej)) if os.path.exists(ej) else {'keep': False, 'why': 'failed'}
                if v.get('keep') and os.path.exists(os.path.join(CLIPS, cid, f'{cid}.mp4')):
                    r.s['doing'] = f'checking every frame of "{v.get("hook") or m["label"]}" before it goes anywhere (on-screen text included)'
                    W.update(step='check', updated=time.time()); W['pick']['hook'] = v.get('hook'); r.save()
                    # brand safety on the picture (pipeline/safety.py): politics, slurs in a tweet, gambling on screen...
                    # a harmless title can hide them (xQc's "grilled cheese" cut was Trump military news, 2026-10-11)
                    import safety
                    try:
                        W_ = json.load(open(os.path.join(CLIPS, cid, 'words.json')))
                        said = ' '.join(w['w'] for sg in W_.get('segments', []) for w in sg.get('words', []))
                    except Exception:  # noqa: BLE001
                        said = ''
                    sv = safety.check(os.path.join(CLIPS, cid, f'{cid}.mp4'), said)
                    if not sv['safe']:
                        v = {**v, 'keep': False, 'why': f"held back: {sv['why']}"}
                        r.s['cuts'].append({'id': cid, 'clip': m['id'], 'dropped': True, 'held': sv['why'], 'at': time.time()})
                        r.log(f"a cut was held back: {sv['why'][:110]}")
                        W['result'] = {'kind': 'held', 'why': sv['why'][:160]}
                if v.get('keep') and os.path.exists(os.path.join(CLIPS, cid, f'{cid}.mp4')):
                    # a past moment has been posted before, by the streamer's viewers and other clippers: what goes out is
                    # a transformative studio edit of it (tease, slow motion, captions, a drop, a montage), never the plain clip
                    ed = self.studio_edit(r, cid, v)
                    r.s['cuts'].append({'id': cid, 'clip': m['id'], 'hook': v['hook'], 'at': time.time(), 'from': m.get('date'), 'views': m['views'],
                                        'kind': 'past stream', **({'edit': ed} if ed else {'plain_only': True})})
                    W['result'] = {'kind': 'edit', 'title': ed['title'], 'critic': ed.get('critic')} if ed else {'kind': 'cut', 'title': v['hook']}
                    r.log((f'studio edit: "{ed["title"]}"' if ed else f'cut: "{v["hook"]}" (studio edit failed: kept off the posting list)')
                          + (f' (the day was short of {quota.DAILY_MIN})' if low else ''))
                    if ed:
                        quota.add(r.slug)
                elif not str(v.get('why', '')).startswith('held back'):
                    r.s['cuts'].append({'id': cid, 'clip': m['id'], 'dropped': True, 'at': time.time()})   # don't retry it
                    r.log(f'dropped at the cut: {v.get("why", "")[:90]}')
                    W['result'] = {'kind': 'dropped', 'why': str(v.get('why', ''))[:160]}
            did = len(r.s['dug']) > n0 or bool(best)
            if not best:
                W['result'] = {'kind': 'none', 'why': f'nothing scored {bar}/10 or better'}
                if r.slug in self.launched():             # a coin room keeps working: its own best live moments that never got an edit
                    did = self.archive_edit(r, W) or did
            r.s['next_dig'] = time.time() + (self.COIN_GAP if did else self.COIN_IDLE_GAP)
            self.maybe_rank(r, catalog)
        except Exception as e:  # noqa: BLE001
            r.log(f'dig failed: {str(e)[:120]}')
        finally:
            if isinstance(r.s.get('work'), dict):
                r.s['work'].update(step='done', updated=time.time())
            r.s['last_dig'] = time.time()
            if r.s['mode'] == 'digging':
                r.s.update(mode='offline', doing='between digs: checking for a live stream every 5 minutes')
            r.save()

    def archive_edit(self, r, W):
        """A coin room whose viewer clips gave nothing to cut re-edits its own best live moment that never got a studio edit
        (the live rooms' studio edit crashed 2026-10-09..21, so most top cuts have none): a 7/10+ kept cut, safe, its source still
        on disk, newest first among the best. Returns True if it worked on one."""
        done = ('ready', 'posting', 'posted', 'submitted', 'paid')
        tried = set(r.s.setdefault('archived', []))
        cands = []
        for d in sorted(glob.glob(os.path.join(HERE, 'runs', f'{r.slug}-20*')), reverse=True)[:120]:
            sf = os.path.join(d, 'state.json')
            if not os.path.isdir(d) or not os.path.exists(sf):
                continue
            try:
                st = json.load(open(sf))
            except (OSError, ValueError):
                continue
            if st.get('status') == 'live' and time.time() - os.path.getmtime(sf) < 900:
                continue                                 # a session still on air: its sources are in use (and its own edits run there)
            for c in st.get('clips') or []:
                key = f"{os.path.basename(d)}/{c.get('id')}"
                if c.get('stage') not in done or (c.get('edit') or {}).get('stage') == 'ready' or key in tried or not c.get('file'):
                    continue
                src, vid = os.path.join(d, 'clips', c['id'], 'src.mp4'), os.path.join(d, c['file'])
                sj = vid + '.safety.json'
                try:
                    if not (os.path.exists(src) and os.path.exists(sj) and json.load(open(sj)).get('safe')):
                        continue
                except (OSError, ValueError):
                    continue
                m = re.match(r'(\d+(?:\.\d+)?)/10', c.get('why') or '')
                sc = float(m.group(1)) if m else 0
                if sc >= 7:
                    cands.append((sc, os.path.getmtime(sf), key, d, c, src, vid))
        if not cands:
            return False
        sc, _, key, d, c, src, vid = sorted(cands, key=lambda x: (-x[0], -x[1]))[0]
        r.s['archived'].append(key)
        run = os.path.basename(d); ymd, hm = run.rsplit('-', 2)[1:]
        date = f'{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}'
        cid = f"{r.slug}-live-{ymd}{hm}-{c['id'].rsplit('-', 1)[-1]}"
        cd = os.path.join(CLIPS, cid); os.makedirs(cd, exist_ok=True)
        for name, target in (('src.mp4', src), (f'{cid}.mp4', vid), (f'{cid}.jpg', vid[:-4] + '.jpg')):
            if os.path.exists(target) and not os.path.lexists(os.path.join(cd, name)):
                os.symlink(target, os.path.join(cd, name))
        hook = c.get('hook') or c['id']
        W.update(step='edit', moments=[{'id': cid, 'title': hook, 'label': hook, 'date': date, 'status': 'scored', 'score': sc}],
                 pick={'id': cid, 'label': hook, 'hook': hook, 'score': sc, 'date': date}, result=None, source='live archive', updated=time.time())
        r.s['doing'] = f'out of fresh viewer clips: re-editing its own best live moment from the {date} stream, "{hook}"'
        r.log(f're-editing "{hook}" from the {date} live stream (cut live at {sc:.0f}/10; its studio edit never ran)'); r.save()
        with self.studio_lock:
            ed = self.studio_edit(r, cid, {'hook': hook, 'why': c.get('why') or ''})
        if ed:
            r.s['cuts'].append({'id': cid, 'clip': f'live:{key}', 'hook': hook, 'at': time.time(), 'from': date, 'kind': 'live stream', 'edit': ed})
            W['result'] = {'kind': 'edit', 'title': ed['title'], 'critic': ed.get('critic')}
            r.log(f'studio edit: "{ed["title"]}" (from the {date} live stream)')
            quota.add(r.slug)
        elif not W.get('result'):
            W['result'] = {'kind': 'none', 'why': 'the studio edit failed'}
        return True

    STORYISH = re.compile(r'wholesome|sweet|emotional|awkward|reunion|gift|confess|cries|crying|tear|proud|hug|surprise|kind', re.I)

    def studio_edit(self, r, cid, v):
        """A studio edit of a dug moment: a vertical mini-story (storyedit --mode mini) when the moment is a story (wholesome,
        awkward, a reunion, a gift: the 100M+ story pages' kind), else a hype edit (hypecut). Returns its record or None."""
        d = os.path.join(CLIPS, cid)
        src = os.path.join(d, 'src.mp4')
        if not os.path.exists(src):
            return None
        if self.STORYISH.search(f"{v.get('why') or ''} {v.get('hook') or ''}"):
            return self.mini_edit(r, cid, v, src)
        a, b = (v.get('trim') or [0, 0])[:2]
        r.s['doing'] = f'the studio is editing "{v.get("hook") or cid}": the tease, the slow motion, the captions, where the drop lands'
        if isinstance(r.s.get('work'), dict):
            r.s['work'].update(step='edit', updated=time.time())
        r.save()
        cmd = ['nice', '-n', '12', PY, os.path.join(ROOT, 'pipeline', 'studio', 'hypecut.py'), src, os.path.join(CLIPS, f'{cid}-edit'),
               '--streamer', r.s['name'], '--credit', r.home, '--handle', handle(r.slug, r.s['name']), '--hint', v.get('hook') or '',
               '--id', f'{cid}-edit', '--workers', '4']
        if b and b > a:
            cmd += ['--a', str(max(0, a - 2)), '--b', str(b + 1)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
        except subprocess.TimeoutExpired:
            return None
        meta = os.path.join(CLIPS, f'{cid}-edit', f'{cid}-edit.json')
        if res.returncode or not os.path.exists(meta):
            r.log(f'studio edit failed: {(res.stderr or res.stdout)[-120:]}')
            return None
        m = json.load(open(meta))
        # the sound, heard back from the finished files (pipeline/safety.audio_gate): a slur the transcript missed, or one in
        # the song's lyrics, is bleeped; one still audible after that keeps the edit off the list. Fails closed
        held = None
        r.s['doing'] = f'listening back to "{m["title"]}" for anything that has to be bleeped'; r.save()
        try:
            sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
            import safety
            for f_ in (f'{cid}-edit.mp4', f'{cid}-edit-nomusic.mp4', f'{cid}-edit-site.mp4'):
                fp = os.path.join(CLIPS, f'{cid}-edit', f_)
                if os.path.exists(fp) and os.path.getsize(fp) > 10000:
                    au = safety.audio_gate(fp)
                    if au['bleeped']:
                        r.log(f'studio edit: {au["why"]} ({f_})')
                    if not au['safe']:
                        held = au['why']; break
        except Exception as e:  # noqa: BLE001
            held = f'the sound check failed ({str(e)[:60]})'
        if held:
            r.log(f'studio edit "{m["title"]}" held back: {held}')
            return None
        # quality over quantity: the head editor (pipeline/critic.py) watches it twice (one read swings +-1); under 7 on
        # average it stays a draft, off the posting list (of 9 hype edits cut on 2026-10-09 only the helicopter made 7+)
        mp4 = os.path.join(CLIPS, f'{cid}-edit', f'{cid}-edit.mp4')
        r.s['doing'] = f'the head editor is watching "{m["title"]}" twice: 7/10 or it stays a draft'
        if isinstance(r.s.get('work'), dict):
            r.s['work'].update(step='grade', updated=time.time())
        r.save()
        try:
            sys.path.insert(0, os.path.join(ROOT, 'pipeline'))
            import critic
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(2) as ex:
                reads = [v.get('score') for v in ex.map(lambda _: critic.critique(mp4, {'hook': m['title'], 'credit': m.get('credit'), 'kind': 'edit'}), range(2))]
            reads = [x for x in reads if isinstance(x, (int, float))]
            avg = sum(reads) / len(reads) if reads else None
        except Exception as e:  # noqa: BLE001
            r.log(f'critic failed on the studio edit: {str(e)[:80]}'); avg = None
        if avg is not None and avg < 7:
            r.log(f'studio edit "{m["title"]}" read {avg:.1f}/10 by the head editor: kept as a draft, not posted')
            if isinstance(r.s.get('work'), dict):
                r.s['work']['result'] = {'kind': 'draft', 'title': m['title'], 'critic': round(avg, 1)}
            return None
        return {'id': f'{cid}-edit', 'title': m['title'], 'file': f'portfolio/clips/{cid}-edit/{cid}-edit.mp4', 'duration': m['duration'],
                'critic': avg}

    def mini_edit(self, r, cid, v, src):
        sd = os.path.join(CLIPS, f'{cid}-mini')
        os.makedirs(os.path.join(sd, 'src'), exist_ok=True)
        if not os.path.exists(os.path.join(sd, 'src', 'clip.mp4')):
            os.symlink(src, os.path.join(sd, 'src', 'clip.mp4'))
        json.dump([{'id': 'clip', 'file': 'src/clip.mp4', 'date': None, 'views': v.get('views'), 'note': f"viewer clip: {v.get('hook') or ''}"}],
                  open(os.path.join(sd, 'pool.json'), 'w'))
        cmd = ['nice', '-n', '12', PY, os.path.join(ROOT, 'pipeline', 'studio', 'storyedit.py'), sd, '--name', v.get('hook') or r.s['name'],
               '--streamer', r.s['name'], '--mode', 'mini', '--credit', r.home, '--handle', handle(r.slug, r.s['name']),
               '--id', f'{cid}-mini', '--workers', '4']
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return None
        meta = os.path.join(sd, f'{cid}-mini.json')
        if res.returncode or not os.path.exists(meta):
            r.log(f'mini-story failed: {(res.stderr or res.stdout)[-120:]}')
            return None
        m = json.load(open(meta))
        return {'id': f'{cid}-mini', 'title': m['title'], 'file': f'portfolio/clips/{cid}-mini/{cid}-mini.mp4', 'duration': m['duration'], 'kind': 'mini'}

    def maybe_rank(self, r, catalog):
        """A ranking once a theme has five strong moments nobody's ranked yet, at most one a week per room."""
        if (r.s.get('last_rank') or 0) > time.time() - 7 * 86400:
            return
        used = set(r.s.get('ranked', []))
        themes = catalog.themes_ready(r.slug, 7, 5, used)
        if not themes:
            return
        theme = themes[0]
        r.s['doing'] = f'making a ranking: {catalog.THEMES[theme].lower()}'; r.log(r.s['doing']); r.save()
        try:
            catalog.rank(r.slug, theme, None, 7, 5, 2, self.a.judge)
            rid = f"{r.slug}-{theme}-{time.strftime('%Y%m%d')}"
            spec = json.load(open(os.path.join(CLIPS, rid, 'spec.json')))
            r.s['cuts'].append({'id': rid, 'hook': spec.get('rank_title'), 'at': time.time(), 'kind': 'ranking'})
            r.s['ranked'] = list(used | {p['source_url'].rsplit('/', 1)[-1] for p in json.load(open(os.path.join(CLIPS, f'{rid}.json')))['parts']})
            r.log(f'ranking: "{spec.get("rank_title")}"')
        except (SystemExit, Exception) as e:  # noqa: BLE001
            r.log(f'ranking failed: {str(e)[:120]}')
        r.s['last_rank'] = time.time()

    # ---------- loop ----------
    def tick(self):
        self.sync()
        for slug, r in self.rooms.items():
            if r.proc and r.proc.poll() is not None:
                r.log(f'live session finished (exit {r.proc.returncode})')
                r.proc = None; r.s.update(mode='offline', since=time.time()); self.registry()
            try:
                k = chan(r)
            except Exception as e:  # noqa: BLE001
                r.log(f'check failed: {str(e)[:80]}'); continue
            gamble = (k.get('category') or '').lower() in GAMBLING
            r.s.update(checked=time.time(), live=k['live'], viewers=k['viewers'], title=k['title'], category=k.get('category'),
                       name=next((n for n in (r.rate.get('name'), r.s.get('name')) if n and n != r.slug), None) or k['name'])   # "Stable Ronaldo", not the login
            if k['live'] and not gamble and not r.proc:
                live_now = sum(1 for x in self.rooms.values() if x.proc and x.proc.poll() is None)
                if live_now < self.a.max_live and avail_gb() >= self.a.min_avail_gb:
                    self.start_live(r, k)
                else:
                    try:
                        on_air = r.slug in json.load(open(REG))   # rooms_watch.py's live room has them on air already
                    except (OSError, ValueError):
                        on_air = False
                    if on_air:
                        r.s.update(mode='live', doing=f"on air: the room is watching {r.s['name']} live and cutting the moments as they happen")
                    else:
                        r.s.update(mode='queued', doing=f'{r.s["name"]} is live and waiting its turn for one of the live rooms; digging past streams meanwhile')
            elif k['live'] and gamble and not r.proc:
                r.s.update(mode='offline', doing=f'{k["name"]} is live in {k["category"]}: gambling streams aren\'t clipped')
            elif not k['live'] and not r.proc and r.s['mode'] not in ('digging',):
                r.s.update(mode='offline', doing='offline: checking for a live stream every 5 minutes, digging past streams in between')
            r.save()
            time.sleep(1)                   # Kick answers bogus "offline" to parallel checks
        # one dig at a time across the fleet, to the room that dug longest ago
        if self.a.dig == 'on' and not (self.digging and self.digging[1].is_alive()):
            coin = self.launched()
            idle = [r for r in self.rooms.values() if not r.proc and r.slug not in coin and time.time() - (r.s.get('last_dig') or 0) > self.a.dig_every]
            if idle and avail_gb() >= self.a.min_avail_gb:
                # a room launched with a coin goes first: its holders watch its page, and it should be seen working (the user,
                # 2026-10-21: "people should see it actively do work"); then the room that dug longest ago
                try:
                    launched = {x.split(':')[-1] for x in json.load(open(os.path.join(ROOT, 'config', 'watch.json')))}
                except (OSError, ValueError):
                    launched = set()
                r = min(idle, key=lambda x: (x.slug not in launched, x.s.get('last_dig') or 0))
                t = threading.Thread(target=self.dig, args=(r,), daemon=True); t.start()
                self.digging = (r.slug, t)

    COIN_GAP, COIN_IDLE_GAP = 90, 300   # s between a coin room's digs: back to back, or after one with nothing at all to do

    def launched(self):
        try:
            return {x.split(':')[-1] for x in json.load(open(os.path.join(ROOT, 'config', 'watch.json')))}
        except (OSError, ValueError):
            return set()

    def schedule_coin_digs(self):
        """Every room launched with a coin has its own agent, working continuously (the user, 2026-10-21: "we should have a
        dedicated agent spawned ... continuously doing work, not every 5 min"): its own dig thread, in parallel with the other
        coin rooms, the next one 90 s after the last ends (5 min after a dig that found nothing to do at all)."""
        if self.a.dig != 'on' or avail_gb() < self.a.min_avail_gb:
            return
        try:
            on_air = set(json.load(open(REG)))         # rooms_watch.py's live rooms: on air, the room page shows the live board instead
        except (OSError, ValueError):
            on_air = set()
        for slug in self.launched():
            r = self.rooms.get(slug)
            if not r or r.proc or slug in on_air or (self.coin_digs.get(slug) and self.coin_digs[slug].is_alive()):
                continue
            if self.digging and self.digging[0] == slug and self.digging[1].is_alive():
                continue
            if not r.s.get('dedicated'):
                r.s['dedicated'] = time.time(); r.log(f"dedicated agent started for {r.s['name']}'s room: it digs, cuts and edits continuously while they're offline")
            if time.time() < (r.s.get('next_dig') or 0):
                continue
            t = threading.Thread(target=self.dig, args=(r,), daemon=True); t.start()
            self.coin_digs[slug] = t

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'running', False))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'running', False))
        print(f'fleet: {", ".join(self.rooms)}; check every {self.a.poll}s, up to {self.a.max_live} live, dig {self.a.dig}', flush=True)
        while self.running:
            self.tick()
            if self.a.once:
                break
            for i in range(self.a.poll):
                if not self.running:
                    break
                if i % 15 == 0:
                    try:
                        self.schedule_coin_digs()
                    except Exception as e:  # noqa: BLE001
                        print(f'{time.strftime("%H:%M:%S")} coin digs: {e}', flush=True)
                time.sleep(1)
        for r in self.rooms.values():
            if r.proc and r.proc.poll() is None:
                r.proc.terminate()
        for r in self.rooms.values():
            if r.proc:
                try:
                    r.proc.wait(30)
                except subprocess.TimeoutExpired:
                    r.proc.kill()
                r.proc = None
            # never 'stopped' ("Paused" on the room page): a restart takes seconds, so the room keeps its last real state; a dig
            # cut off by it is just between digs again
            if r.s.get('mode') in ('digging', 'starting', 'stopped'):
                r.s.update(mode='offline', doing='between digs: checking for a live stream every 5 minutes')
            r.save()
        self.registry()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rooms', required=True, help='comma-separated Kick slugs')
    ap.add_argument('--poll', type=int, default=300)
    ap.add_argument('--max-live', type=int, default=2)
    ap.add_argument('--min-avail-gb', type=float, default=20)
    ap.add_argument('--judge', default='sonnet')
    ap.add_argument('--dig', default='on', choices=['on', 'off'])
    ap.add_argument('--dig-every', type=int, default=600, help='seconds between digs for one room')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--first', default='', help='dig this room before the others (its live room just ended)')
    Fleet(ap.parse_args()).run()
