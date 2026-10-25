#!/usr/bin/env python3
"""Where each streamer streams, decided up front (the user, 2026-10-18: "figure it out before hand ... some streamers are on
both ... lospollos we just prefer kick because that's his main one"). Writes runs/platforms.json:

  {slug: {"kick": "<kick slug>" | null, "twitch": "<twitch login>" | null, "main": "kick" | "twitch", "why": "..."}}

An account counts only if it looks like theirs (verified, or at least 10K followers), so a same-name fan or squatter account
on the other platform is ignored. "main" is where they actually streamed most in the last 30 days (hours of past broadcasts
on each), else where they streamed last, else the bigger following; MAIN below overrides it. The gateway
then shows a streamer on whichever platform they're live on, the main one when they're live on both or offline.

usage: platforms.py            (re-run when a streamer is added; the gateway rereads the file)
"""
import calendar, json, os, sys, time

from curl_cffi import requests as rq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'platforms.json')
sys.path.insert(0, HERE)
import livelog  # noqa: E402

MAIN = {'lospollostv': 'kick'}            # the user's calls: their main platform, whatever the follower counts say
HANDLES = {'asmongold': {'kick': 'asmongold', 'twitch': 'zackrawrr'}}   # slug -> its handles, when they differ between platforms
# one streamer, one slug everywhere (the user, 2026-10-21: Asmongold's Kick and his Twitch zackrawrr are one room): an
# alias's runs, live-log rows, agent and room cards all count as the canonical streamer's
ALIAS = {'zackrawrr': 'asmongold'}


def canon(slug):
    return ALIAS.get(slug, slug)
MIN_FOLLOWERS = 10_000


def candidates():
    s = set(livelog.KICK) | set(livelog.TWITCH)
    try:
        import gateway
        s |= {r['kick'] for r in json.load(open(gateway.RATES))['rates']}
    except (OSError, ValueError, KeyError):
        pass
    try:
        import gateway
        s |= gateway.room_slugs()
    except Exception:  # noqa: BLE001
        pass
    return sorted({canon(x) for x in s if x})


DAYS = 30


def kick(slug):
    try:
        j = rq.get(f'https://kick.com/api/v2/channels/{slug}', impersonate='chrome', timeout=15).json()
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(j, dict) or not j.get('id'):
        return None
    hours, last = 0.0, 1e12 if j.get('livestream') else 0
    try:
        time.sleep(0.4)
        for v in rq.get(f'https://kick.com/api/v2/channels/{slug}/videos', impersonate='chrome', timeout=15).json() or []:
            t = calendar.timegm(time.strptime(v['created_at'][:19], '%Y-%m-%d %H:%M:%S'))
            last = max(last, t)
            if time.time() - t < DAYS * 86400:
                hours += (v.get('duration') or 0) / 3.6e6
    except Exception:  # noqa: BLE001
        pass
    return {'followers': int(j.get('followers_count') or 0), 'verified': bool(j.get('verified')), 'banned': bool(j.get('is_banned')), 'hours': hours, 'last': last}


def twitch(logins):
    q = ' '.join(f'u{i}: user(login:"{n}"){{ login followers{{ totalCount }} roles{{ isPartner }} lastBroadcast{{ startedAt }} stream{{ id }} '
                 f'videos(first:40, type:ARCHIVE, sort:TIME){{ edges{{ node{{ createdAt lengthSeconds }} }} }} }}' for i, n in enumerate(logins))
    r = rq.post('https://gql.twitch.tv/gql', json={'query': '{ ' + q + ' }'}, headers={'Client-Id': livelog.GQL_ID}, impersonate='chrome', timeout=20).json()
    if 'data' not in r:
        raise RuntimeError(str(r)[:300])
    d = r['data']
    out = {}
    for u in d.values():
        if not u:
            continue
        vids = [e['node'] for e in (u.get('videos') or {}).get('edges') or []]
        last = max([livelog.iso(v['createdAt']) for v in vids] + [livelog.iso((u.get('lastBroadcast') or {}).get('startedAt')), 1e12 if u.get('stream') else 0])
        out[u['login']] = {'followers': (u.get('followers') or {}).get('totalCount') or 0, 'verified': bool((u.get('roles') or {}).get('isPartner')), 'banned': False,
                           'hours': sum((v.get('lengthSeconds') or 0) / 3600 for v in vids if time.time() - livelog.iso(v['createdAt']) < DAYS * 86400), 'last': last}
    return out


def theirs(acct):
    return bool(acct) and not acct.get('banned') and (acct['verified'] or acct['followers'] >= MIN_FOLLOWERS)


def main():
    slugs = candidates()
    try:
        import gateway
        bounty = {r['kick'] for r in json.load(open(gateway.RATES))['rates']}
    except (OSError, ValueError, KeyError):
        bounty = set()
    tw = {}
    for i in range(0, len(slugs), 8):
        tw.update(twitch([HANDLES.get(s, {}).get('twitch', s) for s in slugs[i:i + 8]]))
    out = {}
    for s in slugs:
        kh, th = HANDLES.get(s, {}).get('kick', s), HANDLES.get(s, {}).get('twitch', s)
        k, t = kick(kh), tw.get(th)
        time.sleep(0.4)                                   # Kick answers sequential calls; parallel ones come back bogus
        on = {p: a for p, a in (('kick', k), ('twitch', t)) if theirs(a)}
        if not on:
            continue
        m = MAIN.get(s) if MAIN.get(s) in on else max(on, key=lambda p: (round(on[p]['hours'], 1), on[p]['last'], on[p]['followers']))
        if m == 'twitch' and s in bounty and 'kick' in on and on['kick']['hours'] >= 0.5 * on['twitch']['hours'] and s not in MAIN:
            m = 'kick'                                    # simulcasts (xQc): the Kick Clipping bounty pays on Kick
        why = ', '.join(f"{p} {a['hours']:.0f} h in {DAYS} d, {a['followers']:,} followers" for p, a in on.items())
        out[s] = {'kick': kh if 'kick' in on else None, 'twitch': th if 'twitch' in on else None, 'main': m,
                  'why': why + (' · main set by the user' if s in MAIN else ' · Kick bounty streamer' if s in bounty and m == 'kick' and 'twitch' in on else '')}
        print(f"{s:16} main={m:6} {why}")
    tmp = OUT + '.tmp'
    json.dump({'at': round(time.time()), 'streamers': out}, open(tmp, 'w'), indent=1); os.replace(tmp, OUT)
    print(f'{len(out)} streamers -> {OUT}')


if __name__ == '__main__':
    main()
