#!/usr/bin/env python3
"""When are the big streamers on? Every 10 minutes: who is live and with how many viewers, appended to runs/livelog.jsonl,
so the public launch can be timed for when the most of them are live at once (the user, 2026-10-17).

usage: livelog.py [--every 600]          livelog.py report      (the best hours so far, local time)
"""
import calendar, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'runs', 'livelog.jsonl')
YTDLP = os.path.join(os.path.dirname(os.path.dirname(HERE)), '.venv', 'bin', 'yt-dlp')
KICK = ['clavicular', 'adinross', 'n3on', 'xqc', 'deenthegreat', 'lospollostv', 'jasontheween', 'stableronaldo', 'edmatthews', 'konvy', 'asmongold']
TWITCH = ['jasontheween', 'stableronaldo', 'kaicenat', 'ishowspeed', 'zackrawrr', 'jynxzi', 'caseoh_', 'lacy', 'marlon']


def kick(slug):
    from curl_cffi import requests as rq
    j = rq.get(f'https://kick.com/api/v2/channels/{slug}', impersonate='chrome', timeout=15).json()
    ls = j.get('livestream') or {}
    return bool(ls), ls.get('viewer_count') or 0


def twitch(login):
    r = subprocess.run([YTDLP, '--no-warnings', '-J', '--skip-download', f'https://www.twitch.tv/{login}'], capture_output=True, text=True, timeout=60)
    d = json.loads(r.stdout or 'null') or {}
    return bool(d.get('is_live')), d.get('concurrent_view_count') or 0


GQL_ID = 'kimne78kx3ncx6brgo4mv6wki5h1ko'   # the public client id twitch.tv's own web player sends (yt-dlp uses it too)


def iso(t):
    return calendar.timegm(time.strptime(t[:19], '%Y-%m-%dT%H:%M:%S')) if t else 0


def twitch_gql(logins):
    """Every login in one request to Twitch's public GraphQL: live, real viewer count (yt-dlp's comes back 0), title, the live
    preview image, the profile picture, and when their last broadcast ended (newest VOD's end, or the last stream's start)."""
    from curl_cffi import requests as rq
    q = ' '.join(f'u{i}: user(login:"{n}"){{ login displayName profileImageURL(width:150) lastBroadcast{{ startedAt }} '
                 f'videos(first:1, type:ARCHIVE, sort:TIME){{ edges{{ node{{ createdAt lengthSeconds }} }} }} '
                 f'stream{{ viewersCount title previewImageURL(width:640,height:360) game{{ name }} }} }}' for i, n in enumerate(logins))
    d = rq.post('https://gql.twitch.tv/gql', json={'query': '{ ' + q + ' }'}, headers={'Client-Id': GQL_ID}, impersonate='chrome', timeout=15).json()['data']
    out = {}
    for u in d.values():
        if not u:
            continue
        s = u.get('stream') or {}
        v = ((u.get('videos') or {}).get('edges') or [{}])[0].get('node') or {}
        out[u['login']] = {'name': u.get('displayName') or u['login'], 'live': bool(s), 'viewers': s.get('viewersCount') or 0, 'title': s.get('title') or '',
                           'thumb': s.get('previewImageURL') or '', 'category': (s.get('game') or {}).get('name'), 'avatar': u.get('profileImageURL') or '',
                           'vod_end': max(iso(v.get('createdAt')) + (v.get('lengthSeconds') or 0) if v else 0, iso((u.get('lastBroadcast') or {}).get('startedAt')))}
    return out


def tick():
    row = {'at': round(time.time()), 'live': {}}
    try:
        for n, t in twitch_gql(TWITCH).items():
            if t['live']:
                row['live'][f'twitch:{n}'] = t['viewers']
        done = ('kick',)
    except Exception:  # noqa: BLE001
        done = ()                                        # GraphQL down: one yt-dlp call per login instead
    for plat, names, f in (('kick', KICK, kick), ('twitch', TWITCH, twitch)):
        if plat != 'kick' and done:
            continue
        for n in names:
            try:
                on, v = f(n)
            except Exception:  # noqa: BLE001
                continue
            if on:
                row['live'][f'{plat}:{n}'] = v
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(row) + '\n')


def report():
    rows = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    by = {}
    for r in rows:
        t = time.localtime(r['at'])
        k = (time.strftime('%a', t), t.tm_hour)
        by.setdefault(k, []).append(r['live'])
    out = []
    for (d, h), xs in by.items():
        names = {n for x in xs for n in x}
        out.append((sum(len(x) for x in xs) / len(xs), sum(sum(x.values()) for x in xs) / len(xs), d, h, sorted(names)))
    for n, v, d, h, names in sorted(out, reverse=True)[:12]:
        print(f'{d} {h:02d}:00  {n:.1f} live on average, {v:,.0f} viewers  {", ".join(names)}')
    print(f'{len(rows)} checks since {time.strftime("%b %d %H:%M", time.localtime(rows[0]["at"]))}' if rows else 'no checks yet')


if __name__ == '__main__':
    if sys.argv[1:2] == ['report']:
        report()
    else:
        every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 600
        n = 0
        while True:
            try:
                tick()
            except Exception as e:  # noqa: BLE001
                print('tick failed', e, flush=True)
            if n % max(1, 86400 // every) == 0 and n:     # once a day: who streams where (platforms.py; the gateway rereads it)
                try:
                    import platforms
                    platforms.main()
                except Exception as e:  # noqa: BLE001
                    print('platforms failed', e, flush=True)
            n += 1
            time.sleep(every)
