#!/usr/bin/env python3
"""cutroom live: watch a Kick stream while it happens, find the moments, cut them, and show the work.

  ingest   the stream's HLS segments (1080p) into a rolling folder, placed on the stream's own clock (PROGRAM-DATE-TIME)
  listen   audio -> loudness every 0.5 s; faster-whisper (CPU) transcribes 16 s chunks with word times
  chat     Kick's public chat socket: messages/sec, laughs, "clip it", repeated lines (no usernames kept)
  find     per-second score = loudness + chat burst + laughs + "clip it" + hype words; a peak becomes a moment
  judge    an editor pass (judge.py) keeps or drops the moment and writes the hook and caption
  cut      ../edit.py renders it 9:16 with word captions, the hook and a credit; one render at a time
  post     TikTok, once the account owner connects it (../../posting/tiktok.py); until then clips wait as "ready"
  show     /state + /events (SSE) feed the progress page, web/live.html

usage: live.py <kick-slug> [--port 8820] [--minutes 60] [--model small.en] [--threads 4] [--judge claude|off]
"""
import argparse, random, asyncio, collections, importlib, json, os, re, shutil, signal, subprocess, sys, threading, time, urllib.request
from datetime import datetime, timezone

import aiohttp
from aiohttp import web
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
ROOT = os.path.dirname(PIPE)
FF = os.path.join(ROOT, '.venv', 'bin', 'ffmpeg')
YTDLP = os.path.join(ROOT, '.venv', 'bin', 'yt-dlp')
PY = os.path.join(ROOT, '.venv', 'bin', 'python')
sys.path.insert(0, PIPE)
try:   # display names ("N3on", not the Kick username "n3on") from the bounty list
    DISPLAY = {r['kick']: r['name'] for r in json.load(open(os.path.join(os.path.dirname(PIPE), 'config', 'kickrates.json')))['rates'] if r.get('name')}
except (OSError, ValueError, KeyError):
    DISPLAY = {}
sys.path.insert(0, HERE)
import judge as J  # noqa: E402
import quota  # noqa: E402

UA = 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36'
PUSHER = 'wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679?protocol=7&client=js&version=8.4.0-rc2&flash=false'
KICK_RT = 'https://web.kick.com/api/v1/realtime'      # Kick's realtime directory: which chat server, and a token
SR = 16000
LAUGH = re.compile(r"\b(lo+l|lmf?ao+|lmfao|kek\w*|omegalul|icant|i can'?t|ha(ha)+h?|dead|crying|bruh+)\b|😂|🤣|💀|kekw|omegalul", re.I)
CLIPIT = re.compile(r'\bclip(ped|ping| it| that|$)|\bclipit\b|\bclip\b', re.I)
EMOTE = re.compile(r'\[emote:\d+:([^\]]+)\]')
HYPE = re.compile(r"\b(no way|let'?s go+|oh my god|what the|holy|insane|crazy|wtf|are you kidding|i can'?t believe|oh no|nooo+|"
                  r"yo+|bro+|shut up|get out|run|stop|fight|police|security|money|million|thousand|married|kiss|crazy)\b", re.I)


def ff(*args, nice=False):
    cmd = (['nice', '-n', '10'] if nice else []) + [FF, '-nostdin', '-loglevel', 'error', '-y', *map(str, args)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='ignore')[-800:])
    return r.stdout


CTA = re.compile(r'\b(follow (?:me|us|for more)|like and (?:follow|share)|link in bio|buy (?:now|it|this)|get in (?:now|early)|'
                 r'(?:going to|gonna|will|about to) (?:moon|100x|10x|pump|explode|send)|not financial advice|nfa|to the moon|ape in|easy money|'
                 r'free money|guaranteed)\b', re.I)


DAY_PICK_MIN = 7


def with_tags(slug, caption):
    """The account's own hashtags (posting/accounts.json "tags", e.g. #camplooks on Clavicular's), each once."""
    try:
        tags = (json.load(open(os.path.join(ROOT, 'posting', 'accounts.json'))).get(slug) or {}).get('tags') or []
    except (OSError, ValueError):
        tags = []
    have = {w.lower() for w in caption.split() if w.startswith('#')}
    return ' '.join([caption.strip(), *[t for t in tags if t.lower() not in have]]).strip()


def clean_caption(t, hook=''):
    """Post captions stay plain: no aggressive calls to action, no pumpy claims, no $TICKER shilling. A caption that has
    one is replaced by the plain hook plus its hashtags (cutting a claim out mid-sentence reads worse)."""
    t = t or ''
    if CTA.search(t):
        tags = ' '.join(w for w in t.split() if w.startswith('#'))
        t = f"{re.sub(r'[“”"]', '', hook or '').strip()} {tags}".strip()
    t = re.sub(r'(?<![\w#])\$[A-Za-z]{2,10}\b', lambda m: m.group(0)[1:], t)    # $COIN -> COIN
    return re.sub(r'\s{2,}', ' ', t).strip(' ,')


def hms(t):
    t = max(0, int(t))
    return f'{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}'


def fmt_after(sec):
    sec = max(0, int(sec))
    return f'{sec // 60}:{sec % 60:02d}'


def pdt(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


class Live:
    def __init__(self, a):
        self.a = a
        self.slug = a.slug
        stamp = datetime.now().strftime('%Y%m%d-%H%M')
        self.dir = os.path.join(HERE, 'runs', f'{self.slug}-{stamp}')
        self.seg_dir = os.path.join(self.dir, 'seg')
        self.clip_dir = os.path.join(self.dir, 'clips')
        os.makedirs(self.seg_dir, exist_ok=True); os.makedirs(self.clip_dir, exist_ok=True)
        self.started = time.time()
        self.viewer_clips = 0
        self.edits_started = 0
        self.stop_at = self.started + a.minutes * 60
        self.was_live, self.offline_since = False, None   # --end-offline: the stream ended, not a blip
        self.running = True
        self.lock = threading.Lock()
        # the stream clock: T = seconds since the first segment we saw was broadcast
        self.t0 = None
        self.segs = collections.OrderedDict()           # seq -> {T, d, file}
        self.pcm = bytearray()                          # s16le mono 16 kHz; byte 0 is T = 0
        self.loud = []                                  # dB per 0.5 s
        self.words, self.lines = [], []                 # committed transcript
        self.tx_until = 0.0
        self.tx_rtf = None
        self.model_name = a.model
        self.chat_ps = collections.Counter()            # epoch second -> messages
        self.laugh_ps = collections.Counter()
        self.clip_ps = collections.Counter()
        self.chat_recent = collections.deque(maxlen=6000)   # (epoch, text) — text only, never usernames
        self.chat_total = 0
        self.chat_lag = 7.0                             # seconds from a moment to chat reacting to it (prior, then calibrated)
        self.chat_lag_fit = None
        self.edge_lag = None                            # live edge behind the broadcast clock, seconds
        self.target_dur = 4.0
        self.low_latency = False
        self.moments, self.clips = [], []
        self.last_peak = -1e9
        self.scanned_to = 0.0
        self.score = np.zeros(0)
        self.render_q = asyncio.Queue()
        self.meta = {'slug': self.slug, 'name': self.slug, 'platform': a.platform, 'title': '', 'viewers': None, 'live': None, 'chatroom': a.chatroom,
                     'program': a.program, 'rate': a.rate, 'cap': a.cap, 'program_url': a.program_url, 'min_views': a.min_views, 'credit': None}
        self.status = 'starting'
        self.events = collections.deque(maxlen=300)
        self.subs = set()
        self.thumb_at = 0.0
        self.m3u8 = None
        # Kick (bounty streamers) or Twitch (rooms launched for Twitch streamers): stream, chat and channel info differ
        self.platform = a.platform
        self.home = f"{'twitch.tv' if self.platform == 'twitch' else 'kick.com'}/{self.slug}"
        self.log_f = open(os.path.join(self.dir, 'events.jsonl'), 'a')

    # ---------- progress feed ----------
    def emit(self, kind, msg=None, **data):
        ev = {'k': kind, 'at': round(time.time(), 2), 'T': round(self.now_T(), 1), **data}
        if msg:
            ev['msg'] = msg
            self.events.append(ev)
            self.log_f.write(json.dumps(ev) + '\n'); self.log_f.flush()
            print(f"[{hms(ev['T'])}] {msg}", flush=True)
        line = f'event: {kind}\ndata: {json.dumps(ev)}\n\n'.encode()
        for q in list(self.subs):
            if q.qsize() < 200:
                q.put_nowait(line)

    def now_T(self):
        return len(self.pcm) / 2 / SR

    # ---------- ingest ----------
    async def resolve(self):
        fmt = self.a.fmt if self.platform == 'kick' or self.a.fmt != '4' else 'best[height<=1080]/best'   # '4' is Kick's 1080p id
        r = await asyncio.to_thread(subprocess.run, [YTDLP, '-f', fmt, '-g', f'https://{self.home}'], capture_output=True, text=True)
        u = r.stdout.strip().splitlines()
        return u[-1] if r.returncode == 0 and u else None

    async def ingest(self, http):
        fails, last_seq = 0, -1
        while self.running:
            if not self.m3u8:
                self.m3u8 = await self.resolve()
                if not self.m3u8:
                    self.status = 'offline'
                    self.offline_since = self.offline_since or time.time()
                    self.emit('status', f'{self.slug} is not live; checking again in 30 s', status='offline')
                    await asyncio.sleep(30); continue
                self.status = 'live'; self.was_live = True; self.offline_since = None
                self.emit('status', f'tuned in to {self.home}', status='live')
            try:
                async with http.get(self.m3u8, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status != 200:
                        raise RuntimeError(f'playlist {r.status}')
                    txt = await r.text()
                fails = 0
            except Exception as e:  # noqa: BLE001
                fails += 1
                if fails >= 3:
                    self.emit('status', f'lost the playlist ({e}); re-resolving', status='reconnect')
                    self.m3u8 = None
                await asyncio.sleep(2); continue
            seq = None; cur_pdt = None; dur = None; new = []
            for ln in txt.splitlines():
                if ln.startswith('#EXT-X-MEDIA-SEQUENCE:'):
                    seq = int(ln.split(':')[1])
                elif ln.startswith('#EXT-X-TARGETDURATION:'):
                    self.target_dur = float(ln.split(':')[1])
                elif ln.startswith('#EXT-X-PREFETCH:') and not self.low_latency:
                    # low-latency HLS: viewers get segments while they're made, so they see the stream ~2-3 s after broadcast
                    self.low_latency = True
                    self.emit('status', 'stream is low-latency (prefetch): chat reacts within seconds; re-aligned')
                elif ln.startswith('#EXT-X-PROGRAM-DATE-TIME:'):
                    cur_pdt = pdt(ln.split(':', 1)[1])
                elif ln.startswith('#EXTINF:'):
                    dur = float(ln.split(':')[1].split(',')[0])
                elif ln and not ln.startswith('#'):
                    if seq is not None and seq > last_seq and cur_pdt:
                        new.append((seq, cur_pdt, dur, ln))
                    seq = seq + 1 if seq is not None else None; cur_pdt = cur_pdt + dur if cur_pdt else None
            if cur_pdt:
                # cur_pdt is now the end of the newest segment: how far the live edge trails the broadcast clock
                e = time.time() - cur_pdt
                self.edge_lag = e if self.edge_lag is None else 0.9 * self.edge_lag + 0.1 * e
            if last_seq < 0 and new:
                new = new[-3:]              # start at the live edge, not 40 s back
            for sq, p, d, uri in new:
                try:
                    async with http.get(uri, timeout=aiohttp.ClientTimeout(total=20)) as r:
                        body = await r.read()
                except Exception as e:  # noqa: BLE001
                    self.emit('warn', f'segment {sq} failed: {e}'); continue
                if self.t0 is None:
                    self.t0 = p
                    self.emit('status', f'stream clock starts at {datetime.fromtimestamp(p, timezone.utc).strftime("%H:%M:%S")} UTC')
                fn = os.path.join(self.seg_dir, f'{sq:08d}.ts')
                with open(fn, 'wb') as f:
                    f.write(body)
                T = p - self.t0
                self.segs[sq] = {'T': T, 'd': d, 'file': fn}
                last_seq = sq
                await asyncio.to_thread(self.add_audio, fn, T)
                if time.time() - self.thumb_at > 4:
                    self.thumb_at = time.time()
                    asyncio.create_task(asyncio.to_thread(self.thumb, fn))
            self.prune()
            await asyncio.sleep(1.5)

    def add_audio(self, fn, T):
        try:
            raw = ff('-i', fn, '-vn', '-f', 's16le', '-ac', 1, '-ar', SR, '-')
        except RuntimeError as e:
            self.emit('warn', f'audio decode failed: {e}'); return
        with self.lock:
            at = int(T * SR) * 2
            if at > len(self.pcm):
                self.pcm.extend(b'\0' * (at - len(self.pcm)))   # a gap in the stream: silence
            self.pcm[at:at + len(raw)] = raw
            # loudness per 0.5 s for every complete half-second we now have
            x = np.frombuffer(bytes(self.pcm[len(self.loud) * SR:]), np.int16).astype(np.float32) / 32768.0
            n = SR // 2
            k = len(x) // n
            if k:
                fr = x[: k * n].reshape(k, n)
                self.loud.extend((20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-5)).tolist())

    def thumb(self, fn):
        out = os.path.join(self.dir, 'thumb.jpg')
        try:
            ff('-i', fn, '-frames:v', 1, '-vf', 'scale=640:-2', '-q:v', 5, out + '.tmp.jpg')
            os.replace(out + '.tmp.jpg', out)
        except RuntimeError:
            pass

    def prune(self):
        """Keep the last --keep minutes of video, plus anything a pending clip still needs."""
        need = min([c['src'][0] for c in self.clips if c['stage'] in ('queued', 'judging', 'cutting')] + [1e12])
        horizon = min(need - 10, self.now_T() - self.a.keep * 60)
        for sq in list(self.segs):
            s = self.segs[sq]
            if s['T'] + s['d'] < horizon:
                try:
                    os.unlink(s['file'])
                except OSError:
                    pass
                del self.segs[sq]
            else:
                break

    # ---------- chat ----------
    def heard(self, txt):
        now = time.time(); sec = int(now)
        self.chat_ps[sec] += 1; self.chat_total += 1
        if LAUGH.search(txt):
            self.laugh_ps[sec] += 1
        if CLIPIT.search(txt):
            self.clip_ps[sec] += 1
        self.chat_recent.append((now, txt[:140]))

    async def twitch_chat(self, http):
        """Twitch chat, read anonymously over Twitch's IRC websocket (a justinfan nick needs no account or token)."""
        while self.running:
            try:
                async with http.ws_connect('wss://irc-ws.chat.twitch.tv:443', heartbeat=60) as ws:
                    await ws.send_str('CAP REQ :twitch.tv/tags')
                    await ws.send_str('PASS SCHMOOPIIE')
                    await ws.send_str(f'NICK justinfan{random.randint(10000, 99999)}')
                    await ws.send_str(f'JOIN #{self.slug.lower()}')
                    self.emit('status', f'reading chat (twitch.tv/{self.slug})')
                    async for m in ws:
                        if not self.running:
                            break
                        if m.type != aiohttp.WSMsgType.TEXT:
                            continue
                        for line in m.data.split('\r\n'):
                            if line.startswith('PING'):
                                await ws.send_str('PONG :tmi.twitch.tv'); continue
                            i = line.find(' PRIVMSG #')
                            if i < 0:
                                continue
                            j = line.find(' :', i)
                            if j > 0:
                                self.heard(line[j + 2:].strip())
            except Exception as e:  # noqa: BLE001
                self.emit('warn', f'chat socket dropped ({e}); reconnecting')
                await asyncio.sleep(3)

    _ref = None

    def voice_lines(self, words, lines, a, b):
        """The moment's lines for the editor, each tagged with whose voice it is: 'streamer' (matches the voiceprint learned
        from this streamer's past sessions), 'other', or '?' (too short to tell, or no voiceprint yet)."""
        import voiceprint as VP
        if words:
            spans = [(ws[0]['s'], ws[-1]['e'], ' '.join(w['w'] for w in ws)) for ws in VP.lines(words)]
        else:
            spans = [(ln['start'], ln['end'], ln['text']) for ln in lines]
        out = [{'t': round(s0 - a, 1), 'text': t} for s0, s1, t in spans]
        try:
            if VP.load(self.slug):
                w0 = max(0.0, a - 1)
                with self.lock:
                    raw = bytes(self.pcm[int(w0 * SR) * 2: int((b + 1) * SR) * 2])
                x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                for o, lab in zip(out, VP.label(self.slug, x, [(s0 - w0, s1 - w0) for s0, s1, _ in spans])):
                    o['voice'] = lab
        except Exception as e:  # noqa: BLE001
            self.emit('warn', f'voiceprint failed ({str(e)[:80]})')
        return out

    def ref_photo(self):
        """The streamer's profile picture (Kick's or Twitch's) as a small JPEG, cached in runs/refs/<slug>.jpg; None if unavailable."""
        if self._ref is not None:
            return self._ref or None
        path = os.path.join(HERE, 'runs', 'refs', f'{self.slug}.jpg')
        try:
            if not os.path.exists(path) or time.time() - os.path.getmtime(path) > 7 * 86400:
                from curl_cffi import requests as rq
                if self.platform == 'twitch':
                    h = rq.get(f'https://www.twitch.tv/{self.slug}', impersonate='chrome', timeout=15).text
                    m = re.search(r'property="og:image" content="([^"]+)', h)
                    url = m.group(1) if m else None
                else:
                    url = ((rq.get(f'https://kick.com/api/v2/channels/{self.slug}', impersonate='chrome', timeout=15).json().get('user')) or {}).get('profile_pic')
                if url:
                    raw = rq.get(url, impersonate='chrome', timeout=20).content
                    from PIL import Image
                    import io
                    im = Image.open(io.BytesIO(raw)).convert('RGB'); im.thumbnail((320, 320))
                    os.makedirs(os.path.dirname(path), exist_ok=True); im.save(path, 'JPEG', quality=85)
            self._ref = open(path, 'rb').read() if os.path.exists(path) else b''
        except Exception:  # noqa: BLE001
            self._ref = b''
        return self._ref or None

    def kick_realtime(self):
        """Where Kick's chat lives now (2026-10-14: the old Pusher key was retired and every room read no chat at all, its
        socket closed at once with "App key ... not in this cluster", about 3,400 reconnects an hour). Kick's own page asks
        web.kick.com which realtime provider to use for a chatroom (Centrifugo today, Pusher kept as a fallback) and gets
        an anonymous connection token. -> (provider, url, token) or None"""
        import uuid
        from curl_cffi import requests as rq
        cid = str(uuid.uuid4())
        body = {'client': {'id': cid, 'type': 'web'}, 'capabilities': {'accepted_providers': [{'provider': 'pusher'}, {'provider': 'centrifugo'}]}}
        r = rq.post(f"{KICK_RT}/channels/{self.meta['chatroom']}/chat/connection", json=body, impersonate='chrome', timeout=20)
        for c in ((r.json().get('data') or {}).get('connections') or []) if r.ok else []:
            cr = c.get('credentials') or {}
            if c.get('provider') == 'centrifugo' and cr.get('url'):
                t = rq.post(f'{KICK_RT}/auth/connection', json={'client_id': cid}, impersonate='chrome', timeout=20)
                return 'centrifugo', cr['url'], ((t.json().get('data') or {}).get('token') if t.ok else None)
            if c.get('provider') == 'pusher' and (cr.get('appKey') or cr.get('app_key')):
                key, cl = cr.get('appKey') or cr.get('app_key'), cr.get('cluster') or 'us2'
                return 'pusher', f'wss://ws-{cl}.pusher.com/app/{key}?protocol=7&client=js&version=8.4.0&flash=false', None
        return None

    def kick_message(self, ev, raw):
        if str(ev or '').endswith('ChatMessageEvent'):
            x = json.loads(raw) if isinstance(raw, str) else (raw or {})
            self.heard(EMOTE.sub(r' \1 ', x.get('content') or '').strip())

    async def chat(self, http):
        if self.platform == 'twitch':
            return await self.twitch_chat(http)
        wait, said = 3, None
        while self.running:
            if not self.meta['chatroom']:
                await asyncio.sleep(5); continue
            t0 = time.time()
            try:
                rt = await asyncio.to_thread(self.kick_realtime)
                if not rt:
                    raise RuntimeError('Kick named no chat server')
                prov, url, tok = rt
                channel = f"chatrooms.{self.meta['chatroom']}.v2"
                async with http.ws_connect(url, heartbeat=25, headers={'Origin': 'https://kick.com', 'User-Agent': UA}) as ws:
                    if prov == 'centrifugo':
                        await ws.send_str(json.dumps({'id': 1, 'connect': {'name': 'js', **({'token': tok} if tok else {})}}))
                    else:
                        await ws.send_str(json.dumps({'event': 'pusher:subscribe', 'data': {'auth': '', 'channel': channel}}))
                    async for m in ws:
                        if not self.running:
                            break
                        if m.type != aiohttp.WSMsgType.TEXT:
                            continue
                        for line in m.data.split('\n') if prov == 'centrifugo' else [m.data]:
                            line = line.strip()
                            if not line:
                                continue
                            if prov == 'centrifugo' and line == '{}':
                                await ws.send_str('{}'); continue                       # the server's ping
                            d = json.loads(line)
                            if prov == 'centrifugo':
                                if d.get('id') == 1:
                                    if d.get('error'):
                                        raise RuntimeError(f"chat connect refused: {d['error']}")
                                    await ws.send_str(json.dumps({'id': 2, 'subscribe': {'channel': channel}}))
                                elif d.get('id') == 2:
                                    if d.get('error'):
                                        raise RuntimeError(f"chat subscribe refused: {d['error']}")
                                    if said != prov:
                                        self.emit('status', f"reading chat (room {self.meta['chatroom']})"); said = prov
                                elif 'push' in d:
                                    pub = ((d['push'].get('pub') or {}).get('data')) or {}
                                    self.kick_message(pub.get('event'), pub.get('data'))
                            else:
                                if d.get('event') == 'pusher:error':
                                    raise RuntimeError(f"chat refused: {str(d.get('data'))[:120]}")
                                if d.get('event') == 'pusher_internal:subscription_succeeded' and said != prov:
                                    self.emit('status', f"reading chat (room {self.meta['chatroom']})"); said = prov
                                self.kick_message(d.get('event'), d.get('data'))
                raise RuntimeError(f'chat socket closed ({ws.close_code})')
            except Exception as e:  # noqa: BLE001
                # back off when a connection doesn't last, so a refused socket can't spin (it reconnected ~1/s for days)
                wait = 3 if time.time() - t0 > 120 else min(120, wait * 2)
                said = None if time.time() - t0 > 120 else said
                self.emit('warn', f'chat: {str(e)[:120]}; retrying in {wait} s')
                await asyncio.sleep(wait)

    async def twitch_meta(self, http):
        """Title and display name for a Twitch room (yt-dlp reads the channel; viewer counts aren't exposed)."""
        while self.running:
            try:
                r = await asyncio.to_thread(subprocess.run, [YTDLP, '--no-warnings', '-J', '--skip-download', f'https://www.twitch.tv/{self.slug}'],
                                            capture_output=True, text=True, timeout=60)
                d = json.loads(r.stdout or 'null') or {}
                self.meta.update({'name': DISPLAY.get(self.slug) or d.get('uploader') or self.slug, 'live': bool(d.get('is_live')),
                                  'viewers': d.get('concurrent_view_count'), 'title': d.get('description') or d.get('title') or ''})
                self.meta['credit'] = f"{self.meta['name']} · twitch.tv/{self.slug}"
            except Exception as e:  # noqa: BLE001
                self.emit('warn', f'twitch channel info failed ({e})')
            await asyncio.sleep(60)

    async def kick_meta(self, http):
        if self.platform == 'twitch':
            return await self.twitch_meta(http)
        while self.running:
            try:
                async with http.get(f'https://kick.com/api/v2/channels/{self.slug}', headers={'User-Agent': UA, 'Accept': 'application/json'},
                                    timeout=aiohttp.ClientTimeout(total=15)) as r:
                    j = await r.json(content_type=None)
                ls = j.get('livestream') or {}
                self.meta.update({'name': DISPLAY.get(self.slug) or (j.get('user') or {}).get('username') or self.slug, 'live': bool(ls),
                                  'viewers': ls.get('viewer_count'), 'title': ls.get('session_title') or '',
                                  'category': ((ls.get('categories') or [{}])[0] or {}).get('name') or self.meta.get('category'),
                                  'chatroom': self.meta['chatroom'] or (j.get('chatroom') or {}).get('id')})
                self.meta['credit'] = f"{self.meta['name']} · kick.com/{self.slug}"
            except Exception as e:  # noqa: BLE001
                self.emit('warn', f'kick channel info failed ({e})')
            await asyncio.sleep(60)

    # ---------- listen ----------
    def asr_server(self, pcm):
        """Words from the shared GPU ASR (whisper-large-v3-turbo, ../asr/run.sh), or None if it isn't up."""
        try:
            req = urllib.request.Request(self.a.asr + '/transcribe', data=pcm, method='POST', headers={'Content-Type': 'application/octet-stream'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())['words']
        except Exception:  # noqa: BLE001
            return None

    def transcriber(self):
        """16 s chunks, words committed up to 2.5 s before the chunk end (the next chunk starts at the first uncommitted word).
        The GPU server (large-v3-turbo) hears accents far better; CPU faster-whisper small.en is the fallback."""
        model = None
        use_server = self.a.asr_live == 'gpu' and bool(self.a.asr) and self.asr_server(b'\0\0' * SR) is not None
        # the GPU hears a chunk in well under a second, so short chunks keep the transcript a few seconds behind live
        CH = self.a.asr_chunk or (6.0 if use_server else 16.0)
        if use_server:
            self.model_name = 'large-v3-turbo (GPU)'
        else:
            from faster_whisper import WhisperModel
            model = WhisperModel(self.model_name, device='cpu', compute_type='int8', cpu_threads=self.a.threads)
        self.emit('status', f'transcriber ready ({self.model_name}{"" if use_server else f", CPU, {self.a.threads} threads"})')
        prompt = f"{self.meta['name']} live stream. Chat, bro, bruh, yo."
        slow = 0
        down_since = fell_back = None; probed = 0.0
        while self.running:
            a = self.tx_until
            if self.now_T() - a < CH + 1:
                time.sleep(0.5); continue
            with self.lock:
                raw = bytes(self.pcm[int(a * SR) * 2: int((a + CH) * SR) * 2])
            t = time.time()
            ws = None
            if use_server:
                got = self.asr_server(raw)
                if got is None:
                    down_since = down_since or time.time()
                    if time.time() - down_since >= 45:   # gone for good (a restart takes ~20 s): the CPU model rather than stall the room
                        from faster_whisper import WhisperModel
                        use_server, self.model_name, fell_back = False, 'small.en', True
                        model = model or WhisperModel('small.en', device='cpu', compute_type='int8', cpu_threads=self.a.threads)
                        self.emit('status', 'GPU transcriber unreachable; switched to small.en on CPU')
                    else:
                        time.sleep(1); continue
                else:
                    down_since = None
                    ws = [{'w': w['w'], 's': round(a + w['s'], 2), 'e': round(a + w['e'], 2), 'p': 1.0} for w in got if w['w'].strip()]
            elif fell_back and time.time() - probed > 60:   # the GPU server came back: use it again
                probed = time.time()
                if self.asr_server(b'\0\0' * SR) is not None:
                    use_server, self.model_name, down_since = True, 'large-v3-turbo (GPU)', None
                    self.emit('status', 'GPU transcriber is back; switched to large-v3-turbo')
                    continue
            if ws is None:
                x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                segs, _ = model.transcribe(x, language='en', beam_size=1, vad_filter=True, word_timestamps=True,
                                           condition_on_previous_text=False, initial_prompt=prompt)
                ws = [{'w': w.word.strip(), 's': round(a + w.start, 2), 'e': round(a + w.end, 2), 'p': round(w.probability, 2)}
                      for sg in segs for w in (sg.words or []) if w.word.strip()]
            self.tx_rtf = round((time.time() - t) / CH, 2)
            cut = a + CH - 2.5
            keep = [w for w in ws if w['e'] <= cut and (not self.words or w['s'] >= self.words[-1]['e'] - 0.05)]
            rest = [w for w in ws if w['e'] > cut]
            self.tx_until = max(rest[0]['s'] if rest else cut, a + 3.0)
            # running behind the stream: drop to a smaller CPU model rather than fall further back
            slow = slow + 1 if self.tx_rtf > 0.85 else 0
            if slow >= 3 and self.model_name == 'small.en':
                from faster_whisper import WhisperModel
                self.model_name = 'base.en'
                model = WhisperModel('base.en', device='cpu', compute_type='int8', cpu_threads=self.a.threads)
                self.emit('status', f'transcriber was at {self.tx_rtf}x real time; switched to base.en to keep up')
            # lines: break after sentence ends and at pauses, like subtitles
            lines, cur = [], []
            for w in keep:
                if cur and (w['s'] - cur[-1]['e'] > 0.8 or len(cur) >= 18):
                    lines.append(cur); cur = []
                cur.append(w)
                if w['w'].endswith(('.', '?', '!')):
                    lines.append(cur); cur = []
            if cur:
                lines.append(cur)
            for lw in lines:
                ln = {'start': lw[0]['s'], 'end': lw[-1]['e'], 'text': ' '.join(w['w'] for w in lw), 'words': lw}
                self.words.extend(lw)
                self.lines.append({'start': ln['start'], 'end': ln['end'], 'text': ln['text']})
                with open(os.path.join(self.dir, 'transcript.jsonl'), 'a') as f:
                    f.write(json.dumps(ln) + '\n')
                self.emit('tx', start=ln['start'], end=ln['end'], text=ln['text'])

    # ---------- find ----------
    def per_second(self, span=1500):
        """Per-second signals for the last `span` seconds of stream time. Returns (offset, arrays...)."""
        n = int(self.now_T())
        if n < 30 or self.t0 is None or len(self.loud) < n * 2:
            return None
        o = max(0, n - span)
        L = np.array(self.loud[o * 2: n * 2]).reshape(-1, 2).mean(1)
        lag = self.chat_lag
        sec = lambda C: np.array([C.get(int(self.t0 + i + lag), 0) for i in range(o, n)], float)
        chat, laugh, clip = sec(self.chat_ps), sec(self.laugh_ps), sec(self.clip_ps)
        hype = np.zeros(n - o); wps = np.zeros(n - o)
        for w in self.words[-8000:]:
            i = int(w['s']) - o
            if 0 <= i < n - o:
                wps[i] += 1
                if HYPE.search(w['w']):
                    hype[i] += 1
        return o, L, chat, laugh, clip, hype, wps

    @staticmethod
    def robust_z(x, win=300, minh=60):
        """z against the trailing `win` seconds (median / MAD), so a loud stream isn't 'always a moment'.
        The first `minh` seconds have too little history to judge and score 0."""
        out = np.zeros_like(x)
        for i in range(minh, len(x)):
            ref = x[max(0, i - win): i + 1]
            med = np.median(ref); mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-3
            out[i] = (x[i] - med) / mad
        return out

    def lag_prior(self):
        """Where chat should sit: the live edge's delay, plus the viewer's player buffer, plus ~3 s to react and type.
        Low-latency streams (Kick: EXT-X-PREFETCH) buffer ~1.5 s; plain HLS players sit ~3 segments behind."""
        buffer = 1.5 if self.low_latency else 3 * self.target_dur
        return (self.edge_lag or 2.5) + buffer + 3.0

    def calibrate_lag(self):
        """Chat reacts after the moment (stream delay + typing). Fit the lag that lines chat bursts up with loud audio,
        searching near the prior so one noisy stretch can't throw it 15 s off."""
        n = int(self.now_T())
        prior = self.lag_prior()
        if n < 240 or self.t0 is None or len(self.loud) < n * 2:
            if self.chat_lag_fit is None:
                self.chat_lag = round(prior, 1)
            return
        L = np.array(self.loud[max(0, n - 600) * 2: n * 2]).reshape(-1, 2).mean(1)
        lz = np.clip(self.robust_z(L), 0, 6)
        base = n - len(L)
        best = None
        for lag in np.arange(max(2.0, prior - 8), prior + 8.01, 0.5):
            c = np.array([self.chat_ps.get(int(self.t0 + base + i + lag), 0) for i in range(len(L))], float)
            if c.std() < 1e-6:
                continue
            r = float(np.corrcoef(lz, c)[0, 1]) - 0.002 * (lag - prior) ** 2
            if best is None or r > best[1]:
                best = (float(lag), r)
        if not best or best[1] < 0.15:
            if self.chat_lag_fit is None:
                self.chat_lag = round(prior, 1)        # no usable correlation (common: crowd noise isn't chat): stay on the prior
            return
        new = best[0] if self.chat_lag_fit is None else 0.6 * self.chat_lag + 0.4 * best[0]
        if abs(new - self.chat_lag) >= 1:
            self.emit('status', f'chat reacts about {new:.1f} s after the moment (stream delay {self.edge_lag or 0:.1f} s + player buffer + typing); re-aligned')
        self.chat_lag, self.chat_lag_fit = round(new, 1), round(best[1], 2)

    def signals(self):
        P = self.per_second()
        if P is None:
            return None
        o, L, chat, laugh, clip, hype, wps = P
        sm = lambda a, k: np.convolve(a, np.ones(k) / k, 'same')
        lz = np.clip(self.robust_z(L), 0, 6)
        cz = np.clip(self.robust_z(sm(chat, 3)), 0, 8)
        lr = np.clip(self.robust_z(sm(laugh, 4)), 0, 8)
        speech = np.clip(np.convolve(wps, np.ones(20), 'same') / 20.0, 0.3, 1.0)
        score = 0.6 * sm(lz, 3) * speech + 1.0 * sm(cz, 4) + 0.7 * sm(lr, 4) + 0.9 * np.minimum(np.convolve(clip, np.ones(8), 'same'), 6) \
            + 0.5 * sm(hype, 6) * 3
        return {'o': o, 'score': score, 'lz': lz, 'cz': cz, 'lr': lr, 'chat': chat, 'laugh': laugh, 'clip': clip}

    def find(self):
        G = self.signals()
        if G is None:
            return
        o, score = G['o'], G['score']
        n = o + len(score)
        self.score = score
        # confirm peaks that chat has had time to react to and the transcript has caught up with
        # chat about second p only finishes arriving ~(chat_lag - edge_lag + 8) s of stream later: don't judge p before that
        settle = max(14.0, self.chat_lag - (self.edge_lag or 0) + 8)
        ready_to = int(min(n - settle, self.tx_until - 12))
        lo = int(max(self.scanned_to, o + 25))
        if ready_to <= lo:
            return
        hist = score[max(0, 90 - o):][-600:]           # last 10 min, skipping the warm-up minute and a half
        if len(hist) < 45:
            return                                       # still learning this stream's normal; peaks wait, nothing is skipped
        thr = max(self.a.min_score, float(np.percentile(hist, 97)))
        pending = sum(1 for c in self.clips if c['stage'] in ('queued', 'judging', 'cutting'))
        if pending >= 2:
            thr = max(thr, float(np.percentile(hist, 99.3)))
        for p in range(lo, ready_to):
            i = p - o
            if score[i] < thr or score[i] < score[max(0, i - 20): i + 21].max() or p - self.last_peak < self.a.cooldown:
                continue
            self.last_peak = p
            self.add_moment(p, G)
        self.scanned_to = ready_to

    def add_moment(self, p, G, forced=False):
        o = G['o']; i = p - o
        a0, b0 = p - 20, p + 8
        ws = [w for w in self.words if p - 30 <= w['s'] <= p + 16]
        # start on a sentence start (a pause before a word) near p-20; end on a pause after the payoff
        starts = [w['s'] for k, w in enumerate(ws) if p - 26 <= w['s'] <= p - 10 and (k == 0 or w['s'] - ws[k - 1]['e'] > 0.5)]
        a = min(starts, key=lambda s: abs(s - (p - 20))) - 0.25 if starts else a0
        ends = [w['e'] for k, w in enumerate(ws) if p + 3 <= w['e'] <= p + 14 and (k == len(ws) - 1 or ws[k + 1]['s'] - w['e'] > 0.5)]
        b = (ends[0] + 0.5) if ends else b0
        first = next(iter(self.segs.values()))['T'] if self.segs else 0
        a = max(a, first + 0.5)
        b = min(b, self.now_T() - 1, a + 50)
        mid = len(self.moments) + 1
        chat, laugh, clip = G['chat'], G['laugh'], G['clip']
        base_chat = float(np.median(chat[max(0, i - 300): i])) + 0.2 if i > 10 else 1.0
        sig = {'loud_z': round(float(G['lz'][i]), 1), 'chat_x': round(float(chat[max(0, i - 3): i + 8].mean()) / base_chat, 1),
               'laughs': int(laugh[max(0, i - 3): i + 10].sum()), 'clip_it': int(clip[max(0, i - 3): i + 10].sum()),
               'score': round(float(G['score'][i]), 1), 'forced': forced}
        # what chat said (text only) during the reaction window: repeated lines are the best hook material
        e0, e1 = self.t0 + p - 2 + self.chat_lag, self.t0 + p + 12 + self.chat_lag
        said = collections.Counter(re.sub(r'\s+', ' ', t.lower()).strip()[:60] for at, t in self.chat_recent if e0 <= at <= e1 and t.strip())
        lo_i, hi_i = max(0, (p - 5) * 2), min(len(self.loud), (p + 3) * 2)
        beat = (lo_i + int(np.argmax(self.loud[lo_i:hi_i]))) / 2 if hi_i > lo_i else p
        m = {'id': f'm{mid:02d}', 'peak': p, 'beat': beat, 'src': [round(a, 2), round(b, 2)], 'signals': sig, 'found_at': time.time(),
             'chat_top': [[t, c] for t, c in said.most_common(8)],
             'text': ' '.join(w['w'] for w in self.words if a <= w['s'] <= b)}
        self.moments.append(m)
        why = ', '.join(x for x in [f"chat ×{sig['chat_x']}" if sig['chat_x'] >= 1.5 else '', f"{sig['laughs']} laughs" if sig['laughs'] >= 3 else '',
                                     f"{sig['clip_it']}× \"clip it\"" if sig['clip_it'] else '', f"loud +{sig['loud_z']}σ" if sig['loud_z'] >= 2 else ''] if x) \
            or ('operator pressed clip' if forced else 'combined signals')
        self.emit('moment', f"moment {m['id']} at {hms(p)}: {why}", moment=m)
        clip = {'id': f'{self.slug}-{mid:02d}', 'moment': m['id'], 'src': m['src'], 'peak': p, 'beat': beat, 'stage': 'queued', 'times': {'found': time.time()},
                'hook': None, 'caption': None, 'layout': None, 'file': None, 'signals': sig}
        self.clips.append(clip)
        self.render_q.put_nowait(clip)
        self.emit('clip', clip=self.pub(clip))

    async def semantic(self):
        """Every minute: the editor reads the last ~2.5 minutes of transcript (and two frames) for moments that work on their
        own terms (a mood switch, a contradiction, a comeback, a relatable dynamic), whether or not chat reacted. A strong
        one becomes a moment like any chat-triggered one; if chat already flagged it, the reading is attached to it."""
        if self.a.semantic_every <= 0:
            return
        import semantic as SEM
        seen = []
        while self.running:
            await asyncio.sleep(self.a.semantic_every)
            if self.a.semantic_every <= 0 or self.t0 is None:
                continue
            now = self.tx_until
            win = [l for l in self.lines if now - 150 <= l['start'] < now]
            if len(win) < 4:
                continue
            try:
                frames = await asyncio.to_thread(self.frames_at, [now - 90, now - 30])
                found = await asyncio.to_thread(SEM.scan, win, frames, self.meta.get('name') or self.slug, '', self.meta.get('title') or '', self.a.judge)
            except Exception as e:  # noqa: BLE001
                self.emit('warn', f'semantic scan failed: {str(e)[:120]}'); continue
            for sm in found:
                if any(abs(sm['peak'] - x) < 15 for x in seen):
                    continue
                seen.append(sm['peak'])
                near = next((m for m in self.moments if abs(m['peak'] - sm['peak']) < 15), None)
                if near:                               # chat caught it too: give the editor the reading
                    near.setdefault('semantic', sm)
                    continue
                if sm['score'] >= self.a.semantic_min:
                    self.add_semantic_moment(sm)

    def add_semantic_moment(self, sm):
        a, b, p = sm['a'], min(sm['b'], self.now_T() - 1), int(sm['peak'])
        mid = len(self.moments) + 1
        e0, e1 = self.t0 + p - 2 + self.chat_lag, self.t0 + p + 12 + self.chat_lag
        said = collections.Counter(re.sub(r'\s+', ' ', t.lower()).strip()[:60] for at, t in self.chat_recent if e0 <= at <= e1 and t.strip())
        sig = {'loud_z': 0.0, 'chat_x': 1.0, 'laughs': 0, 'clip_it': 0, 'score': round(sm['score'], 1), 'forced': False, 'semantic': sm['kind']}
        m = {'id': f'm{mid:02d}', 'peak': p, 'beat': sm['peak'], 'src': [round(a, 2), round(b, 2)], 'signals': sig, 'found_at': time.time(),
             'chat_top': [[t, c] for t, c in said.most_common(8)], 'semantic': sm,
             'text': ' '.join(w['w'] for w in self.words if a <= w['s'] <= b)}
        self.moments.append(m)
        self.emit('moment', f"moment {m['id']} at {hms(p)}: the editor read it ({sm['kind']}: {sm['why'][:80]})", moment=m)
        clip = {'id': f'{self.slug}-{mid:02d}', 'moment': m['id'], 'src': m['src'], 'peak': p, 'beat': sm['peak'], 'stage': 'queued',
                'times': {'found': time.time()}, 'hook': None, 'caption': None, 'layout': None, 'file': None, 'signals': sig}
        self.clips.append(clip)
        self.render_q.put_nowait(clip)
        self.emit('clip', clip=self.pub(clip))

    async def finder(self):
        k = 0
        while self.running:
            await asyncio.sleep(2)
            try:
                await asyncio.to_thread(self.find)
                k += 1
                if k % (15 if self.chat_lag_fit is None else 60) == 0:
                    await asyncio.to_thread(self.calibrate_lag)
            except Exception as e:  # noqa: BLE001
                self.emit('warn', f'finder error: {e}')

    # ---------- judge + cut ----------
    def pub(self, c):
        return {k: v for k, v in c.items() if k not in ('words',)}

    def set_stage(self, c, stage, msg=None):
        c['stage'] = stage; c['times'][stage] = time.time()
        self.emit('clip', msg, clip=self.pub(c))

    async def renderer(self):
        while self.running:
            c = await self.render_q.get()
            try:
                await self.make(c)
            except Exception as e:  # noqa: BLE001
                c['error'] = str(e)[-300:]
                self.set_stage(c, 'failed', f"{c['id']} failed: {str(e)[-160:]}")

    async def make(self, c):
        m = next(x for x in self.moments if x['id'] == c['moment'])
        a, b = c['src']
        # wait for the transcript to cover the end of the clip
        while self.tx_until < b + 1 and self.running:
            await asyncio.sleep(1)
        lines = [ln for ln in self.lines if ln['end'] > a and ln['start'] < b]
        # re-hear just this window with the GPU model (large-v3-turbo): far better on accents than the live CPU pass,
        # and it's what the editor reads and the captions are made from
        hq = await asyncio.to_thread(self.rehear, a - 0.5, b + 0.5)
        if hq:
            c['words'] = hq
            c['asr'] = 'large-v3-turbo'
            lines = [{'start': hq[0]['s'], 'end': hq[-1]['e'], 'text': ' '.join(w['w'] for w in hq)}]
        c['heard'] = ' '.join(ln['text'] for ln in lines)[:400]
        self.set_stage(c, 'judging', f"{c['id']}: editor is reading the moment")
        tj = time.time()
        importlib.reload(J)   # editor rules can be tuned while the room is on air
        frames = await asyncio.to_thread(self.frames_at, [c['peak'] + d for d in (-8, -3, 1, 5)])
        c['frames'] = len(frames)
        # keep what the editor looked at, so the page can show its desk: the frames, the line it heard, its call
        fd = os.path.join(self.clip_dir, c['id']); os.makedirs(fd, exist_ok=True)
        c['seen'] = []
        for k, jpg in enumerate(frames):
            with open(os.path.join(fd, f'seen{k}.jpg'), 'wb') as f:
                f.write(jpg)
            c['seen'].append(f"clips/{c['id']}/seen{k}.jpg")
        self.emit('clip', clip=self.pub(c))
        # the streamer's own profile picture goes in first, so the editor can tell them from the friends and guests on screen
        # (a collab line by Clavicular's friend once went out captioned as his, 2026-10-17)
        ref = await asyncio.to_thread(self.ref_photo)
        # and each line says whose voice it is, by the streamer's voiceprint (pipeline/voiceprint.py)
        said = await asyncio.to_thread(self.voice_lines, c.get('words'), lines, a, b)
        c['voices'] = [x.get('voice') for x in said]
        # the minute before the window, so the editor can ask for the full-context cut when the setup is the good part
        before = [{'t': round(ln['start'] - a, 1), 'text': ln['text']} for ln in self.lines if ln['end'] > a - 60 and ln['start'] < a and ln.get('text')]
        v = await asyncio.to_thread(J.judge, {'streamer': self.meta['name'], 'title': self.meta['title'], 'slug': self.slug,
                                              'streamer_photo': bool(ref),
                                              'lines': said, **({'before': before} if before else {}),
                                              'chat_top': m['chat_top'], 'signals': m['signals'], 'duration': round(b - a, 1),
                                              **({'reading': {k: m['semantic'][k] for k in ('kind', 'why', 'framing')},
                                                  'reading_peak_t': round(m['semantic']['peak'] - a, 1)} if m.get('semantic') else {})},
                                    self.a.judge, ([ref] if ref else []) + frames)
        c.update({'hook': v['hook'], 'caption': v['caption'], 'why': v.get('why'), 'judge': v.get('by'), 'judge_s': round(time.time() - tj, 1), 'speaker': v.get('speaker'),
                  'evidence': v.get('evidence'), 'trim': v.get('trim'), 'payoff': v.get('payoff'), 'context': v.get('context'), 'score': quota.score_of(v.get('why')), 'big': v.get('big') or [],
                  **({'hook_original': v['hook_original']} if v.get('hook_original') else {})})
        if not v['keep'] and c.get('hook') and self.rescuable(c['score'], c['why']) and self.behind():
            # behind the daily minimum (or a new coin room's first hour): a decent moment beats no cut at all (safety drops stay dropped)
            v['keep'] = True; c['rescued'] = True
            c['why'] = f"{c['why']} · kept: " + ("a new room's first hour" if self.boosting() else f"the room cuts at least {quota.DAILY_MIN} a day")
        if not v['keep']:
            self.set_stage(c, 'dropped', f"{c['id']} dropped by the editor: {v.get('why', '')}")
            return
        await self.cut(c, a, b)

    def boosting(self):
        """A room just launched with a coin: its first --boost minutes on air cut more freely, so the person who launched it
        sees the agent cut within minutes: 3 cuts paced over the window on this session's own count (whatever the day's), and
        "cut anyway" from 5/10 instead of 6. Safety still applies to every cut."""
        return self.a.boost > 0 and self.now_T() < self.a.boost * 60

    def rescuable(self, score, why):
        if quota.rescuable(score, why):
            return True
        return self.boosting() and score is not None and score >= 5 and not quota.UNSAFE.search(why or '')

    def behind(self):
        """Behind pace for the daily minimum: on track for it within the first 3 hours of watching. In a new coin room's
        first hour: behind until it has 1 cut after 5 minutes, then 3 paced over the window."""
        if self.boosting() and self.now_T() > 300:
            made = sum(1 for c in self.clips if c['stage'] in ('cutting', 'ready', 'posting', 'posted', 'submitted', 'paid'))
            return made < max(1.0, 3 * self.now_T() / (self.a.boost * 60))
        span = min(self.a.minutes * 60, 3 * 3600)
        return quota.behind(self.slug, (time.time() - self.started) / max(60, span))

    async def rescue(self):
        """Every 2 minutes, when behind the daily minimum and nothing is being judged or cut, go back for the best moment
        the editor passed on (score >= quota.RESCUE_MIN, not dropped for safety, still on disk)."""
        while self.running:
            await asyncio.sleep(120)
            if not self.behind() or any(c['stage'] in ('queued', 'judging', 'cutting') for c in self.clips):
                continue
            on_disk = self.now_T() - (self.a.keep * 60 - 90)
            cands = [c for c in self.clips if c['stage'] == 'dropped' and not c.get('rescue_tried') and c.get('hook')
                     and self.rescuable(c.get('score'), c.get('why')) and c['src'][0] > on_disk]
            if not cands:
                continue
            c = max(cands, key=lambda x: (x['score'], x['peak']))
            c['rescue_tried'] = True; c['rescued'] = True
            c['why'] = f"{c['why']} · cut anyway: " + ("a new room's first hour" if self.boosting() else f"the room cuts at least {quota.DAILY_MIN} a day")
            self.emit('status', f"{c['id']}: behind the daily minimum, going back for the best moment it passed on ({c['score']:.0f}/10)")
            a, b = c['src']
            try:
                await self.cut(c, a, b)
            except Exception as e:  # noqa: BLE001
                c['error'] = str(e)[-300:]
                self.set_stage(c, 'failed', f"{c['id']} failed: {str(e)[-160:]}")

    async def cut(self, c, a, b):
        c['window'] = [round(a, 2), round(b, 2)]      # what the editor read, before its trim (the page draws the kept span inside it)
        a0 = a
        if c.get('trim'):
            ta, tb = c['trim']
            a, b = max(a, a + ta), min(b, a + tb) if tb else b
            c['src'] = [round(a, 2), round(b, 2)]
        # the shape top clips have (research/references/STYLE-GUIDE.md): the payoff late, a hard cut right after it, no
        # long walk-up. Our critic's most common note was dead setup and an ending that trails off.
        pk = (a0 + c['payoff']) if c.get('payoff') is not None else c.get('peak')   # the editor's payoff second, else the chat peak
        if c.get('context'):
            # the full-context cut (the user, 2026-10-19: "longer, full context when it's interesting"): start at a sentence
            # start up to a minute before the window, run through the payoff, at most 75 s; its words re-heard for captions
            first = next(iter(self.segs.values()))['T'] + 0.5 if self.segs else 0
            want = max(first, a0 - c['context'])
            ws = self.words
            starts = [w['s'] for k, w in enumerate(ws) if want - 4 <= w['s'] <= want + 4 and (k == 0 or w['s'] - ws[k - 1]['e'] > 0.5)]
            a = (min(starts, key=lambda x: abs(x - want)) - 0.25) if starts else want
            if pk and a < pk < b:
                b = min(b, pk + 4)
            b = min(b, a + 75)
            hq = await asyncio.to_thread(self.rehear, a - 0.5, b + 0.5)
            if hq:
                c['words'] = hq
            c['long'] = True; c['src'] = [round(a, 2), round(b, 2)]
        elif pk and a < pk < b:
            a2, b2 = max(a, pk - 25), min(b, pk + 4)
            if b2 - a2 >= 10:
                a, b = a2, b2
                c['src'] = [round(a, 2), round(b, 2)]
        import edit as E_                               # and the end on a finished line, never mid-sentence
        b_ = E_.snap_end(c.get('words') or self.words, max(a + 8, (pk or b) + 0.5), b, b + 3.5)
        if b_ != b:
            b = b_; c['src'] = [round(a, 2), round(b, 2)]
        if (c.get('score') or 0) >= self.a.edit_min and not c.get('rescued') and self.edits_started < self.a.studio_edits:
            c['edit_next'] = True; self.edits_started += 1      # the session's best moments also get a studio edit
        self.set_stage(c, 'cutting', f"{c['id']} {'cut anyway' if c.get('rescued') else 'kept'}: \"{c['hook']}\" — cutting {b - a:.0f} s")
        t = time.time()
        out = await asyncio.to_thread(self.render, c, a, b)
        c.update(out)
        c['render_s'] = round(time.time() - t, 1)
        c['after_moment_s'] = round(time.time() - (self.t0 + c['peak']), 1)
        # brand safety on the picture before anything posts (pipeline/safety.py): politics, a slur in a tweet on screen,
        # gambling... the transcript alone missed them once (xQc's Trump military-news cut, 2026-10-11)
        sv = await asyncio.to_thread(self.safe_cut, c)
        if not sv['safe']:
            c['why'] = f"held back: {sv['why']}"
            c.pop('edit_next', None)
            self.set_stage(c, 'hidden', f"{c['id']} held back: {sv['why'][:140]}")
            return
        self.set_stage(c, 'ready', f"{c['id']} ready: {c['duration']:.0f} s {c['layout']} cut, {hms(c['after_moment_s'])[3:]} after it happened")
        quota.add(self.slug)
        if c.pop('edit_next', None):
            asyncio.create_task(self.studio_edit_safe(c))
        await self.post(c)

    def safe_cut(self, c):
        import safety
        d = os.path.join(self.clip_dir, c['id'])
        try:
            W = json.load(open(os.path.join(d, 'words.json')))
            said = ' '.join(w['w'] for sg in W.get('segments', []) for w in sg.get('words', []))
        except Exception:  # noqa: BLE001
            said = ''
        return safety.check(os.path.join(self.dir, c['file']), said)

    async def studio_edit_safe(self, c):
        """studio_edit, but a crash marks the edit failed instead of leaving the clip on "editing" forever (a bad call
        crashed every live room's studio edit from 2026-10-09 to 10-21 without a word in the log but a traceback)."""
        try:
            await self.studio_edit(c)
        except Exception as e:  # noqa: BLE001
            c['edit'] = {'stage': 'failed', 'error': f'{type(e).__name__}: {str(e)[:200]}'}
            self.emit('clip', f"{c['id']}: studio edit failed ({type(e).__name__}: {str(e)[:120]})", clip=self.pub(c))

    async def studio_edit(self, c):
        """The editor plans a clip-page edit of this cut (tease, slow motion, the drop, a montage) and the studio renders it
        (pipeline/studio/hypecut.py), next to the plain cut. The site copy has no music; the TikTok copy does."""
        d = os.path.join(self.clip_dir, c['id']); eid = f"{c['id']}-edit"; out = os.path.join(d, 'edit')
        t0 = time.time()
        c['edit'] = {'stage': 'editing', 'started': t0}
        self.emit('clip', f"{c['id']}: the editor is planning a studio edit of it: the tease, the slow motion, where the music drops", clip=self.pub(c))
        handle = 'cutroom' + re.sub(r'[^a-z0-9]', '', (self.meta.get('name') or self.slug).lower())
        cmd = ['nice', '-n', '12', PY, os.path.join(PIPE, 'studio', 'hypecut.py'), os.path.join(d, 'src.mp4'), out, '--streamer', self.meta.get('name') or self.slug,
               '--credit', self.home, '--handle', handle, '--hint', c.get('hook') or '', '--id', eid, '--workers', '4']
        import gamecam  # noqa: E402
        if gamecam.is_game(self.meta.get('category'), self.meta.get('title') or ''):
            cmd.append('--game')
        try:
            r = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=1200)
            ok = r.returncode == 0
            err = (r.stderr or r.stdout)[-300:]
        except subprocess.TimeoutExpired:
            ok, err = False, 'timed out'
        for f in ('src.mp4', 'src.m4a'):
            if not self.a.keep_src and os.path.exists(os.path.join(d, f)):
                os.unlink(os.path.join(d, f))
        if not ok:
            c['edit'] = {'stage': 'failed', 'error': err}
            self.emit('clip', f"{c['id']}: studio edit failed", clip=self.pub(c)); return
        meta = json.load(open(os.path.join(out, f'{eid}.json')))
        # the sound, heard back from both copies (pipeline/safety.audio_gate): a slur the transcript missed, or one in the
        # TikTok copy's song, is bleeped; one still audible after that holds the edit. Fails closed
        held = None
        try:
            import safety
            for f_ in (f'{eid}.mp4', f'{eid}-site.mp4'):
                fp = os.path.join(out, f_)
                if os.path.exists(fp):
                    au = await asyncio.to_thread(safety.audio_gate, fp)
                    if not au['safe']:
                        held = au['why']; break
        except Exception as e:  # noqa: BLE001
            held = f'the sound check failed ({str(e)[:60]})'
        if held:
            c['edit'] = {'stage': 'failed', 'error': f'held back: {held}'}
            self.emit('clip', f"{c['id']}: studio edit held back: {held}", clip=self.pub(c)); return
        subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-y', '-ss', '3', '-i', os.path.join(out, f'{eid}.mp4'), '-frames:v', '1', '-vf', 'scale=540:-2',
                        os.path.join(out, f'{eid}.jpg')], capture_output=True)
        sheet = os.path.join(out, f'{eid}-sheet.webp')                    # the cut sheet: what the edit is made of
        await asyncio.to_thread(subprocess.run, ['nice', '-n', '12', PY, os.path.join(PIPE, 'studio', 'cutsheet.py'), os.path.join(out, 'edit.json'),
                                                 os.path.join(out, f'{eid}.mp4'), sheet, '--title', f"{meta['title']}: how the room cut it"], capture_output=True)
        c['edit'] = {'stage': 'ready', 'file': f"clips/{c['id']}/edit/{eid}-site.mp4", 'poster': f"clips/{c['id']}/edit/{eid}.jpg", 'title': meta['title'],
                     'duration': meta['duration'], 'took_s': round(time.time() - t0),
                     **({'sheet': f"clips/{c['id']}/edit/{eid}-sheet.webp"} if os.path.exists(sheet) else {})}
        self.emit('clip', f"{c['id']}: studio edit ready, \"{meta['title']}\" ({hms(time.time() - t0)[3:]} to plan and render)", clip=self.pub(c))

    def rehear(self, a, b):
        if not self.a.asr:
            return None
        with self.lock:
            raw = bytes(self.pcm[int(max(0, a) * SR) * 2: int(b * SR) * 2])
        got = self.asr_server(raw)
        if not got:
            return None
        return [{'w': w['w'], 's': round(max(0, a) + w['s'], 2), 'e': round(max(0, a) + w['e'], 2), 'p': 1.0} for w in got if w['w'].strip()]

    def frames_at(self, ts):
        """Small JPEG stills at stream times ts, for the editor to look at."""
        out = []
        for t in ts:
            s = next((x for x in self.segs.values() if x['T'] <= t < x['T'] + x['d']), None)
            if not s:
                continue
            try:
                # seek after -i: a lone 4 s segment can't be input-seeked (B-frame refs), decoding it from the start is cheap
                out.append(ff('-i', s['file'], '-ss', f"{t - s['T']:.2f}", '-frames:v', 1, '-vf', 'scale=480:-2', '-q:v', 6, '-f', 'image2pipe', '-c:v', 'mjpeg', '-'))
            except RuntimeError:
                pass
        return out

    def render(self, c, a, b):
        d = os.path.join(self.clip_dir, c['id']); os.makedirs(d, exist_ok=True)
        segs = [s for s in self.segs.values() if s['T'] + s['d'] > a - 1 and s['T'] < b + 1]
        if not segs:
            raise RuntimeError('segments already pruned')
        off = segs[0]['T']
        src_ts = os.path.join(d, 'src.ts')
        with open(src_ts, 'wb') as f:
            for s in segs:
                with open(s['file'], 'rb') as g:
                    shutil.copyfileobj(g, f)
        src = os.path.join(d, 'src.mp4'); aud = os.path.join(d, 'src.m4a')
        ff('-i', src_ts, '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy', '-bsf:a', 'aac_adtstoasc', src)
        ff('-i', src_ts, '-vn', '-c:a', 'copy', '-bsf:a', 'aac_adtstoasc', aud)
        os.unlink(src_ts)
        src_words = c.get('words') or self.words
        ws = [{**w, 's': round(w['s'] - off, 2), 'e': round(w['e'] - off, 2)} for w in src_words if a - 0.5 <= w['s'] <= b + 0.5]
        json.dump({'segments': [{'start': ws[0]['s'] if ws else 0, 'end': ws[-1]['e'] if ws else 0, 'text': ' '.join(w['w'] for w in ws), 'words': ws}]},
                  open(os.path.join(d, 'words.json'), 'w'))
        import edit as E, facecam  # noqa: E402  (cv2 is heavy; import on first render)
        # the fixed corner cam (xQc-style) must show in at least 2 of 3 samples: a dark corner of a night IRL shot can
        # look like one once
        votes = [E.side_at(src, (b - a) * t + (a - off)) for t in (0.3, 0.5, 0.7)]
        side = next((v for v in set(votes) if v and votes.count(v) >= 2), None)
        # a facecam box over a game, browser or chat; layouts switch mid-moment (a death screen, then the cams), so try the payoff half too
        cam = None if side else (facecam.detect(src, a - off, b - off) or facecam.detect(src, a - off + (b - a) * 0.45, b - off))
        import layouts  # noqa: E402
        layout, extra = layouts.choose(src, a - off, b - off, self.slug, ' '.join(w['w'] for w in ws), side, cam,
                                       self.meta.get('category'), self.meta.get('title') or '')
        spec = {'id': c['id'], 'video': 'src.mp4', 'audio': 'src.m4a', 'words': 'words.json', 'out_dir': '.', 'layout': layout,
                'segments': [[round(a - off, 2), round(b - off, 2)]], 'hook': c['hook'], 'punch': [round(c['beat'] - off, 2)] if c.get('beat') else [], 'big': c.get('big') or [], 'credit': self.meta['credit'] or self.home,
                'program': self.meta['program'], 'source_url': f'https://{self.home}', 'post_caption': c['caption'],
                'rules_check': {'source': f"live {'Twitch' if self.platform == 'twitch' else 'Kick'} stream", 'credit_on_screen': os.environ.get('CUTROOM_ONSCREEN_CREDIT') == '1', 'no_ai_video': True, 'no_botting': True}}
        if cam:
            spec['cam'] = {k: cam[k] for k in ('x', 'y', 'w', 'h', 'side')}
            spec['screen'] = cam['screen']
        spec.update(extra)
        json.dump(spec, open(os.path.join(d, 'spec.json'), 'w'), indent=1)
        r = subprocess.run(['nice', '-n', '10', PY, os.path.join(PIPE, 'edit.py'), os.path.join(d, 'spec.json')], capture_output=True, text=True,
                           env={**os.environ, 'CUTROOM_FF_THREADS': str(self.a.render_threads), 'CUTROOM_PRESET': 'veryfast'})
        if r.returncode:
            raise RuntimeError(r.stderr[-600:])
        sheet = json.load(open(os.path.join(d, c['id'] + '.json')))
        for f in ('src.mp4', 'src.m4a'):
            if not self.a.keep_src and not c.get('edit_next'):
                os.unlink(os.path.join(d, f))
        return {'layout': layout, 'duration': sheet['duration'], 'file': f"clips/{c['id']}/{c['id']}.mp4", 'poster': f"clips/{c['id']}/{c['id']}.jpg"}

    async def post(self, c):
        """Keep the rendered clip local; this release contains no social publisher."""
        c['post'] = {'status': 'ready locally; social publishing is not included'}
        self.emit('clip', clip=self.pub(c))

    def to_kit(self, c, why):
        """Keep the rendered clip local; this release contains no social publisher."""
        c['post'] = {'status': 'ready locally; social publishing is not included'}
        self.emit('clip', clip=self.pub(c))

    async def day_pick(self):
        """Social publishing belongs to the separate deployment integration."""
        return

    async def api_post(self, c):
        """Keep the rendered clip local; this release contains no social publisher."""
        c['post'] = {'status': 'ready locally; social publishing is not included'}
        self.emit('clip', clip=self.pub(c))

    # ---------- show ----------
    def state(self):
        n = int(self.now_T())
        L = np.array(self.loud[max(0, n - 300) * 2: n * 2]).reshape(-1, 2).mean(1).round(1).tolist() if len(self.loud) >= n * 2 and n else []
        chat = [self.chat_ps.get(int(self.t0 + i + self.chat_lag), 0) for i in range(max(0, n - 300), n)] if self.t0 else []
        return {'meta': self.meta, 'status': self.status, 'run': os.path.basename(self.dir), 'started': self.started, 'stop_at': self.stop_at,
                't0': self.t0,   # broadcast time (epoch) of T = 0: lets a delayed player line the transcript up with the picture
                'T': round(self.now_T(), 1), 'tx_until': round(self.tx_until, 1), 'tx_rtf': self.tx_rtf, 'model': self.model_name,
                'chat_total': self.chat_total, 'viewer_clips': self.viewer_clips, 'chat_lag': self.chat_lag, 'chat_lag_fit': self.chat_lag_fit, 'edge_lag': round(self.edge_lag or 0, 1),
                'words': len(self.words), 'lines': self.lines[-14:],
                'curve': {'from': max(0, n - 300), 'loud': L, 'chat': chat, 'score': self.score[-300:].round(2).tolist()},
                'moments': self.moments[-30:], 'clips': [self.pub(c) for c in self.clips[-30:]], 'events': list(self.events)[-80:],
                'threshold': self.a.min_score}

    async def h_state(self, req):
        return web.json_response(self.state(), headers={'Cache-Control': 'no-store'})

    async def h_events(self, req):
        resp = web.StreamResponse(headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        await resp.prepare(req)
        q = asyncio.Queue(); self.subs.add(q)
        try:
            await resp.write(f'event: hello\ndata: {json.dumps(self.state())}\n\n'.encode())
            while self.running:
                try:
                    line = await asyncio.wait_for(q.get(), 15)
                except asyncio.TimeoutError:
                    line = b': ping\n\n'
                await resp.write(line)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            self.subs.discard(q)
        return resp

    async def h_claim(self, req):
        """Operator steps that have no API: paste the post link, mark the program submission, record a payout.
        body: {id, action: posted|submitted|paid, url?, amount?}"""
        if not self.a.op_token or req.query.get('op') != self.a.op_token:
            raise web.HTTPForbidden()
        b = await req.json()
        c = next((x for x in self.clips if x['id'] == b.get('id')), None)
        if not c:
            raise web.HTTPNotFound()
        m = c.setdefault('money', {})
        act = b.get('action')
        if act == 'posted' and b.get('url'):
            c.setdefault('post', {})['url'] = b['url']
            self.set_stage(c, 'posted', f"{c['id']}: posted → {b['url']}")
        elif act == 'submitted':
            m['submitted_at'] = time.time()
            m['program'] = self.meta['program']
            self.set_stage(c, 'submitted', f"{c['id']}: submitted to {self.meta['program']}"
                           + (f" ({fmt_after(m['submitted_at'] - c['times'].get('posted', m['submitted_at']))} after posting)" if c['times'].get('posted') else ''))
        elif act == 'paid' and b.get('amount') is not None:
            m['paid_usd'] = float(b['amount']); m['paid_at'] = time.time()
            self.set_stage(c, 'paid', f"{c['id']}: program paid ${float(b['amount']):,.2f}")
        else:
            raise web.HTTPBadRequest(text='unknown action')
        return web.json_response({'ok': True, 'clip': self.pub(c)})

    async def views(self):
        """Public view counts for posted clips, every 5 min (yt-dlp reads the post page; no login)."""
        while self.running:
            await asyncio.sleep(300)
            for c in [x for x in self.clips if (x.get('post') or {}).get('url') and x['stage'] in ('posted', 'submitted', 'paid')]:
                r = await asyncio.to_thread(subprocess.run, [YTDLP, '-j', '--skip-download', c['post']['url']], capture_output=True, text=True, timeout=60)
                try:
                    v = int(json.loads(r.stdout).get('view_count') or 0)
                except (ValueError, TypeError):
                    continue
                m = c.setdefault('money', {})
                m['views'], m['views_at'] = v, time.time()
                if self.meta.get('rate'):
                    cap = self.a.cap or v
                    m['projected_usd'] = round(min(v, cap) / 100_000 * self.meta['rate'], 2)
                self.emit('clip', clip=self.pub(c))

    async def humans(self):
        """Kick viewers' own clips of this stream (every 90 s), matched to our cuts by broadcast time: the page can show that
        people clipped the same moment, and how long after it they had a raw 16:9 clip versus our finished cut."""
        from curl_cffi import requests as rq
        seen = {}
        while self.running:
            await asyncio.sleep(90)
            if self.t0 is None:
                continue
            if self.platform != 'kick':                  # viewers' clips of this stream: read from Kick's clip list
                continue
            try:
                r = await asyncio.to_thread(rq.get, f'https://kick.com/api/v2/channels/{self.slug}/clips?sort=date&time=day', impersonate='chrome', timeout=20)
                cl = (r.json().get('clips') or []) if r.status_code == 200 else []
            except Exception:  # noqa: BLE001
                continue
            for v in cl:
                try:
                    made = datetime.fromisoformat(v['created_at'].replace('Z', '+00:00')).timestamp()
                except (KeyError, ValueError):
                    continue
                if made < self.started - 120:
                    continue
                seen[v['id']] = {'id': v['id'], 'made': made, 'dur': v.get('duration') or 0, 'title': (v.get('title') or '')[:80],
                                 'views': v.get('view_count') or v.get('views') or 0, 'thumb': v.get('thumbnail_url'), 'video': v.get('video_url'), 'url': v.get('clip_url') or
                                 f"https://kick.com/{self.slug}/clips/{v['id']}"}
            self.viewer_clips = len(seen)
            for c in self.clips:
                if c['stage'] not in ('ready', 'posting', 'posted', 'submitted', 'paid'):
                    continue
                pw = self.t0 + c['peak']                # the moment, in broadcast time
                # a Kick clip ends about when it is made and runs back up to its length (plus the picker's slack)
                hit = sorted((h for h in seen.values() if h['made'] - h['dur'] - 45 <= pw <= h['made'] + 2), key=lambda h: h['made'])
                if not hit:
                    continue
                hs = [{**{k: h[k] for k in ('id', 'title', 'views', 'thumb', 'video', 'url', 'dur')}, 'after_s': round(h['made'] - pw, 1)} for h in hit[:5]]
                if hs != c.get('humans'):
                    c['humans'] = hs
                    ours = c.get('after_moment_s')
                    msg = f"{c['id']}: {len(hs)} viewer clip{'s' if len(hs) > 1 else ''} of the same moment, first {hms(hs[0]['after_s'])[3:]} after it"
                    if ours is not None:
                        msg += f" (ours: {hms(ours)[3:]})"
                    self.emit('clip', msg, clip=self.pub(c))
                    # viewers clipping it too makes a held cut strong: post it now, while the moment is fresh, not at the session's end
                    if self.a.post != 'off' and c['stage'] == 'ready' and (c.get('post') or {}).get('status', '').startswith('held'):
                        sys.path.insert(0, os.path.join(ROOT, 'posting'))
                        import accounts  # noqa: E402
                        if accounts.strong(c) and accounts.slots(self.slug) > 0:
                            asyncio.create_task(self.post(c))

    async def h_clip(self, req):
        """Operator 'clip it' button: forces a moment at the stream's current position minus the chat lag."""
        if not self.a.op_token or req.query.get('op') != self.a.op_token:
            raise web.HTTPForbidden()
        G = self.signals()
        if G is None:
            raise web.HTTPConflict(text='not enough stream yet')
        p = int(min(self.now_T() - 8, G['o'] + len(G['score']) - 1))
        self.last_peak = p
        self.add_moment(p, G, forced=True)
        return web.json_response({'ok': True, 'peak': p})

    def update_text(self):
        """A standalone X post for this point in the run (one post per update, never a thread)."""
        ready = [c for c in self.clips if c['stage'] in ('ready', 'posting', 'posted')]
        kept = [c for c in self.clips if c['stage'] not in ('dropped', 'failed', 'queued', 'judging')]
        lat = [c['after_moment_s'] for c in ready if c.get('after_moment_s')]
        mins = int(self.now_T() // 60)
        el = f'{mins // 60}h {mins % 60:02d}m' if mins >= 60 else f'{mins} min'
        fast = f'{int(min(lat) // 60)}:{int(min(lat) % 60):02d}' if lat else '—'
        return (f"Live room, {el} in, watching {self.home}.\n\n{len(self.moments)} moments flagged\n{len(kept)} kept by the editor\n"
                f"{len(ready)} clips cut\nfastest: ready {fast} after it happened\n\ncutroom.fun/live")

    async def ticker(self):
        every = self.a.update_every * 60
        next_update = time.time() + every
        while self.running:
            await asyncio.sleep(2)
            if every and time.time() >= next_update:
                next_update += every
                txt = self.update_text()
                with open(os.path.join(self.dir, 'updates.jsonl'), 'a') as f:
                    f.write(json.dumps({'at': time.time(), 'T': self.now_T(), 'text': txt}) + '\n')
                nm, nc = len(self.moments), sum(1 for c in self.clips if c['stage'] in ('ready', 'posting', 'posted'))
                self.emit('update', f"progress update: {nm} moment{'s' * (nm != 1)} flagged, {nc} clip{'s' * (nc != 1)} cut", text=txt)
            s = self.state()
            self.emit('tick', T=s['T'], tx_until=s['tx_until'], tx_rtf=s['tx_rtf'], chat_total=s['chat_total'], meta=s['meta'], curve=s['curve'],
                      status=self.status, chat_lag=self.chat_lag, model=self.model_name)
            # every minute, the room's state on disk too (atomic), not only at the end: the $CUTROOM card's since-launch numbers
            # (brand/x/launchday/since.py) count runs from state.json, and read 0 for a whole hour of live rooms on launch day
            if time.time() - getattr(self, '_saved_at', 0) > 60:
                self._saved_at = time.time()
                try:
                    tmp = os.path.join(self.dir, 'state.json.tmp')
                    json.dump(s, open(tmp, 'w')); os.replace(tmp, os.path.join(self.dir, 'state.json'))
                except Exception:  # noqa: BLE001
                    pass
            ended = (self.a.end_offline and self.was_live and self.offline_since
                     and time.time() - self.offline_since > self.a.end_offline * 60 and not any(c['stage'] in ('judging', 'cutting') for c in self.clips))
            # a room whose stream never came up (it ended just as the room opened) hands over too, instead of idling an hour
            ended = ended or bool(self.a.end_offline and not self.was_live and time.time() - self.started > (self.a.end_offline + 1) * 60)
            if time.time() > self.stop_at or ended:
                self.emit('status', f'stream ended (offline {self.a.end_offline:g} min); stopping' if ended else f'demo window over after {self.a.minutes} min; stopping', status='done')
                json.dump(self.state(), open(os.path.join(self.dir, 'state.json'), 'w'))
                self.running = False

    async def main(self):
        app = web.Application()
        app.router.add_get('/state', self.h_state)
        app.router.add_get('/events', self.h_events)
        app.router.add_post('/clip', self.h_clip)
        app.router.add_post('/claim', self.h_claim)
        app.router.add_static('/clips', self.clip_dir)
        app.router.add_get('/thumb.jpg', lambda r: web.FileResponse(os.path.join(self.dir, 'thumb.jpg'), headers={'Cache-Control': 'no-store'}))
        # `kill <pid>` stops cleanly: state.json and signals.json still get written
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, lambda: setattr(self, 'running', False))
        runner = web.AppRunner(app); await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', self.a.port).start()
        self.emit('status', f'cutroom live on :{self.a.port}, run {os.path.basename(self.dir)}', status='starting')
        async with aiohttp.ClientSession(headers={'User-Agent': UA}) as http:
            threading.Thread(target=self.transcriber, daemon=True).start()
            tasks = [asyncio.create_task(x) for x in (self.kick_meta(http), self.ingest(http), self.chat(http), self.finder(), self.renderer(), self.ticker(), self.views(), self.humans(), self.rescue(), self.semantic())]
            while self.running:
                await asyncio.sleep(1)
            for t in tasks:
                t.cancel()
            try:
                await self.day_pick()                   # a posting slot left today: the session's best held cut takes it
            except Exception as e:  # noqa: BLE001
                self.emit('status', f'day pick failed: {e}')
            # leave the last frame of the run on disk for the replay page, plus the raw signals for tuning
            json.dump(self.state(), open(os.path.join(self.dir, 'state.json'), 'w'))
            n = int(self.now_T())
            json.dump({'t0': self.t0, 'chat_lag': self.chat_lag,
                       'loud': np.array(self.loud[: n * 2]).reshape(-1, 2).mean(1).round(2).tolist() if len(self.loud) >= n * 2 else [],
                       'chat': [self.chat_ps.get(int(self.t0 + i), 0) for i in range(n)] if self.t0 else [],
                       'laugh': [self.laugh_ps.get(int(self.t0 + i), 0) for i in range(n)] if self.t0 else [],
                       'clip': [self.clip_ps.get(int(self.t0 + i), 0) for i in range(n)] if self.t0 else []},
                      open(os.path.join(self.dir, 'signals.json'), 'w'))
        await runner.cleanup()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('slug')
    ap.add_argument('--port', type=int, default=8820)
    ap.add_argument('--minutes', type=float, default=60)
    ap.add_argument('--boost', type=float, default=0, help='minutes: a new coin room cuts more freely at first (live.boosting)')
    ap.add_argument('--end-offline', type=float, default=0, help='stop once the stream has been offline this many minutes after being live (0 = never)')
    ap.add_argument('--fmt', default='4', help='yt-dlp format id; 4 = 1080p on Kick')
    ap.add_argument('--model', default='small.en', help='CPU fallback model when the GPU ASR server is down')
    ap.add_argument('--asr', default=os.environ.get('CUTROOM_ASR', 'http://127.0.0.1:8835'), help="shared GPU ASR server, used to re-hear each moment ('' = off)")
    ap.add_argument('--asr-live', default='cpu', choices=['cpu', 'gpu'], help='live transcription: cpu (small.en, 16 s chunks, scales to several rooms) or gpu (large-v3-turbo, 6 s chunks, one room)')
    ap.add_argument('--asr-chunk', type=float, default=0, help='live chunk seconds (0 = 16 on CPU, 6 on GPU)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--keep', type=float, default=15, help='minutes of video kept on disk')
    ap.add_argument('--keep-src', action='store_true')
    ap.add_argument('--semantic-every', type=float, default=60, help='seconds between semantic scans of the transcript (0 = off)')
    ap.add_argument('--semantic-min', type=float, default=7, help='a semantic reading this strong becomes a moment on its own')
    ap.add_argument('--studio-edits', type=int, default=0, help='this many of the best cuts also get a studio edit (hypecut); requires a configured music library; 0 = off')
    ap.add_argument('--edit-min', type=float, default=6, help='editor score a cut needs for a studio edit')
    ap.add_argument('--render-threads', type=int, default=8, help='encoder threads per render, so whisper keeps up')
    ap.add_argument('--chatroom', type=int, default=None)
    ap.add_argument('--platform', default='kick', choices=['kick', 'twitch'], help='twitch: rooms launched for Twitch streamers')
    ap.add_argument('--min-score', type=float, default=4.0)
    ap.add_argument('--cooldown', type=float, default=75)
    ap.add_argument('--judge', default='sonnet', choices=['sonnet', 'haiku', 'codex', 'claude', 'off'], help='the editor: sonnet (live demo), haiku, codex (kept warm), off = rules')
    ap.add_argument('--post', default='off', choices=['off'], help='local output only; social publishing is not included')
    ap.add_argument('--disclosure', default='', help='appended to every caption, e.g. the program\'s required disclosure')
    ap.add_argument('--program', default='Kick Clipping (clipping.net) bounty')
    ap.add_argument('--rate', type=float, default=None, help='$ per 100K views, for the dashboard')
    ap.add_argument('--cap', type=float, default=None, help='views the program pays up to, per clip')
    ap.add_argument('--min-views', type=float, default=None, help="program's minimum views before it pays")
    ap.add_argument('--program-url', default='https://clipping.net/c/kick-clipping', help='where the operator submits a posted clip')
    ap.add_argument('--op-token', default=os.environ.get('CUTROOM_OP'))
    ap.add_argument('--update-every', type=float, default=5, help='minutes between progress updates (X drafts in updates.jsonl); 0 = off')
    asyncio.run(Live(ap.parse_args()).main())
    os._exit(0)   # a worker thread stuck on a dead stream must not keep the room alive after it's done (it once ran 16 h)
