#!/usr/bin/env python3
"""A streamer's moment catalog: the deep pool rankings choose from.

Gathers the streamer's most-viewed viewer clips (Kick: this month + all time, paged; Twitch: last 30 days + all time),
downloads each at 360p (cheap), re-hears it on the GPU ASR, and has the editor score them in batches (one call per
--batch moments, transcript + one frame each): score 1-10 as a standalone short, tags (funny, wild, heated, emotional,
wholesome, hype, fail, skill, awkward, chaotic), a 2-5 word label, the 8-15 s that carry it, and hard-no reasons
(slurs, sexual, gambling on screen, minors, allegations). Labels must come from what was said or shown.

A ranking then takes only the best: --rank funny picks the 5 highest-scored moments tagged funny (score >= --min),
at most --per-stream from one stream, downloads just those in 1080p, and renders the countdown (rankcut.py) with the
catalog's labels and spans.

usage: catalog.py <slug> --name xQc [--platform kick|twitch] [--pool 60] [--judge sonnet] [--batch 8]
       catalog.py <slug> --rank funny [--title "XQC'S FUNNIEST MOMENTS THIS MONTH"] [--min 7] [--n 5]
Writes portfolio/catalog/<slug>/catalog.json (+ low-res files), and the ranking under portfolio/clips/<slug>-<theme>-<date>/.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'live'))
import clipcut as C  # noqa: E402

CAT = os.path.join(ROOT, 'portfolio', 'catalog')
GAMBLING = {'slots & casino', 'gambling', 'slots', 'casino', 'poker'}
TAGS = ['funny', 'wild', 'heated', 'emotional', 'wholesome', 'hype', 'fail', 'skill', 'awkward', 'chaotic']
THEMES = {'funny': "FUNNIEST MOMENTS", 'wild': "WILDEST MOMENTS", 'heated': "MOST HEATED MOMENTS", 'emotional': "MOST EMOTIONAL MOMENTS",
          'wholesome': "MOST WHOLESOME MOMENTS", 'hype': "MOST HYPE MOMENTS", 'fail': "BIGGEST FAILS", 'skill': "CRAZIEST PLAYS",
          'awkward': "MOST AWKWARD MOMENTS", 'chaotic': "MOST CHAOTIC MOMENTS"}

RULES = """You are the editor of a short-form clipping page building a catalog of a streamer's moments, so the best ones can
go into rankings ("TOP 5 FUNNIEST MOMENTS"). Each moment below is a clip the streamer's viewers made, with its transcript
(seconds from the clip start) and one frame. Judge each as a standalone short for strangers scrolling TikTok.

For each moment return:
- score 1-10: 9-10 instantly funny/wild with a clear payoff anyone gets; 7-8 strong; 5-6 fine but needs context; 1-4 weak.
- tags: 1-3 of funny, wild, heated, emotional, wholesome, hype, fail, skill, awkward, chaotic
- label: 2-5 words naming the moment for a countdown list, from words said or things the frame shows (no invented facts)
- span: [start, end] seconds, the 8-15 s that carry it (setup + payoff)
- hard_no: "" or the reason (a slur that IS the moment: the joke, or said at someone; sexual content, gambling on screen:
  slots/casino/poker/cards and chips/betting sites, minors, real allegations, tragedies, humiliating a member of the
  public). Political topics are NOT a hard no (the user, 2026-10-21). A hard no gets score 1. A slur said in passing is NOT a hard no: our cut mutes the word and masks it in the
  captions, the way the big clip pages do, so score the moment as if that word weren't there.
- evidence: the words said (or "frame: ...") the label comes from

Reply with JSON only: {"moments": [{"i": <index>, "score": n, "tags": [...], "label": "...", "span": [a, b], "hard_no": "", "evidence": "..."}]}

STREAMER: %s
MOMENTS:
"""


def kick_pool(slug, pool):
    from curl_cffi import requests as rq
    out = {}
    # this week's clips first (fresh moments: the user prefers new clips, and a 24/7 stream's newest never reach an all-time
    # top 80, so a launched room's dig ran dry: "read 0 past moments" all day, 2026-10-21), then the month's and all time's
    for tm, pages in (('week', 3), ('month', 4), ('all', max(2, -(-pool // 20)))):
        cur = None
        for _ in range(pages):
            u = f'https://kick.com/api/v2/channels/{slug}/clips?sort=view&time={tm}' + (f'&cursor={urllib.parse.quote(cur)}' if cur else '')
            try:
                j = rq.get(u, impersonate='chrome', timeout=20).json()
            except Exception:  # noqa: BLE001
                break
            for c in j.get('clips') or []:
                cat = ((c.get('category') or {}).get('name') or '').lower()
                if cat in GAMBLING or not 10 <= (c.get('duration') or 0) <= 150:
                    continue
                out.setdefault(c['id'], {'id': c['id'], 'url': c['video_url'], 'source_url': f"https://kick.com/{slug}/clips/{c['id']}", 'title': c['title'], 'created': c.get('created_at'),
                                         'views': c['view_count'], 'duration': c['duration'], 'date': (c.get('vod_starts_at') or c.get('created_at') or '')[:10],
                                         'stream': c.get('livestream_id'), 'category': (c.get('category') or {}).get('name'), 'window': tm})
            cur = j.get('nextCursor')
            time.sleep(1)
            if not cur:
                break
    ranked = sorted(out.values(), key=lambda c: -c['views'])
    return ([c for c in ranked if c['window'] == 'week'] + [c for c in ranked if c['window'] != 'week'])[:pool]


def twitch_pool(user, pool):
    out = {}
    for rng in ('7d', '30d', 'all'):                    # this week's first, as kick_pool (fresh moments; the agents ran dry)
        r = subprocess.run([C.YTDLP, '--flat-playlist', '--playlist-end', str(pool), '-j', f'https://www.twitch.tv/{user}/clips?filter=clips&range={rng}'],
                           capture_output=True, text=True, timeout=300)
        for ln in r.stdout.splitlines():
            j = json.loads(ln)
            if not 10 <= (j.get('duration') or 0) <= 150:
                continue
            cid = j['url'].rsplit('/', 1)[-1]
            out.setdefault(cid, {'id': cid, 'url': j['url'], 'source_url': j['url'], 'title': j.get('title'), 'views': j.get('view_count') or 0,
                                 'duration': j.get('duration'), 'date': None, 'stream': None, 'category': None, 'window': 'week' if rng == '7d' else rng})
    ranked = sorted(out.values(), key=lambda c: -c['views'])
    return ([c for c in ranked if c['window'] == 'week'] + [c for c in ranked if c['window'] != 'week'])[:pool]


def low(m, d):
    """360p copy for hearing and a frame; the ranking fetches the winners in 1080p later."""
    f = os.path.join(d, f"{m['id']}.mp4")
    if not os.path.exists(f):
        subprocess.run([C.YTDLP, '-q', '--no-warnings', '--ffmpeg-location', C.FF, '-f', 'b[height<=480]/bv*[height<=480]+ba/w', '--merge-output-format', 'mp4', '-o', f, m['url']],
                       capture_output=True, text=True, timeout=300)
    return f if os.path.exists(f) else None


def build(slug, name, platform, pool, judge_mode, batch, limit=None, progress=None):
    """limit: add at most this many new moments this call (the fleet grows a catalog a batch at a time while offline).
    progress(event, data): the room's work board (fleet.py) follows the dig: 'batch' (the moments it's fetching), 'heard'
    (one moment heard, its frame written next to its 360p copy as .jpg), 'scoring' (ids), 'scored' (the editor's calls)."""
    say = progress or (lambda *_: None)
    import rehear, judge as J
    d = os.path.join(CAT, slug); os.makedirs(os.path.join(d, 'low'), exist_ok=True)
    cf = os.path.join(d, 'catalog.json')
    cat = json.load(open(cf)) if os.path.exists(cf) else {'slug': slug, 'name': name, 'platform': platform, 'moments': {}}
    cands = kick_pool(slug, pool) if platform == 'kick' else twitch_pool(slug, pool)
    print(f'{slug}: {len(cands)} candidate moments', flush=True)
    # downloads are network-bound (Kick clips come in one 1080p rendition): fetch three at a time ahead of the ASR
    from concurrent.futures import ThreadPoolExecutor
    need = [m for m in cands if not (m['id'] in cat['moments'] and cat['moments'][m['id']].get('words') is not None)][:limit]
    keep = {m['id'] for m in need} | {k for k, m in cat['moments'].items() if m.get('words') is not None}
    cands = [m for m in cands if m['id'] in keep]
    say('batch', [{'id': m['id'], 'title': m.get('title'), 'views': m.get('views'), 'date': m.get('date')} for m in need])
    pool_x = ThreadPoolExecutor(3)
    futs = {m['id']: pool_x.submit(low, m, os.path.join(d, 'low')) for m in need}
    for m in cands:
        if m['id'] in cat['moments'] and cat['moments'][m['id']].get('words') is not None:
            cat['moments'][m['id']].update(views=m['views'], created=m.get('created'), stream=m.get('stream')); continue
        try:
            f = futs[m['id']].result()
        except Exception:  # noqa: BLE001
            f = None
        if not f:
            continue
        dur = C.duration(f)
        aud = f[:-4] + '.wav'
        C.ff('-i', f, '-vn', '-ac', '1', '-ar', 16000, aud)
        ws = []
        for t in range(0, int(dur) + 1, 30):
            if dur - t > 0.5:
                ws += rehear.hear_safe(aud, t, min(dur, t + 30))
        os.unlink(aud)
        try:                                             # a frame for the work board
            C.ff('-ss', f'{dur * 0.5:.2f}', '-i', f, '-frames:v', 1, '-vf', 'scale=320:-2', '-q:v', 5, f[:-4] + '.jpg')
        except RuntimeError:
            pass
        cat['moments'][m['id']] = {**m, 'low': f, 'dur': round(dur, 1), 'words': [[w['w'], w['s']] for w in ws]}
        json.dump(cat, open(cf, 'w'), indent=0, ensure_ascii=False)
        say('heard', {'id': m['id'], 'words': len(ws)})
    todo = [m for m in cat['moments'].values() if m.get('words') is not None and 'score' not in m]
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        items, frames = [], []
        for i, m in enumerate(chunk):
            text = ' '.join(f"[{s:.0f}] {w}" if j % 12 == 0 else w for j, (w, s) in enumerate(m['words']))
            items.append({'i': i, 'viewer_clip_title': m['title'], 'views': m['views'], 'duration': m['dur'], 'transcript': text[:1400]})
            try:
                frames.append(C.ff('-ss', f"{m['dur'] * 0.6:.2f}", '-i', m['low'], '-frames:v', 1, '-vf', 'scale=360:-2', '-q:v', 6, '-f', 'image2pipe', '-c:v', 'mjpeg', '-'))
            except RuntimeError:
                frames.append(b'')
        frames = [f for f in frames if f]
        import lore
        say('scoring', [m['id'] for m in chunk])
        prompt = (RULES % name) + lore.prompt_block(slug) + '\n' + json.dumps(items, ensure_ascii=False) + '\n(One frame per moment, in order, taken 60% into each.)'
        try:
            raw = J._ask(prompt, frames, 120, J.CLAUDE_MODELS.get(judge_mode, 'sonnet'))
            v = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
        except Exception as e:  # noqa: BLE001
            print(f'  batch {k // batch}: editor failed ({str(e)[:80]})', flush=True); continue
        for r in v.get('moments', []):
            if not isinstance(r.get('i'), int) or r['i'] >= len(chunk):
                continue
            m = chunk[r['i']]
            m.update({'score': r.get('score'), 'tags': [t for t in r.get('tags') or [] if t in TAGS], 'label': J.mask(str(r.get('label') or '')),
                      'span': r.get('span'), 'hard_no': r.get('hard_no') or '', 'evidence': r.get('evidence') or ''})
        json.dump(cat, open(cf, 'w'), indent=0, ensure_ascii=False)
        say('scored', [{'id': m['id'], 'score': m.get('score'), 'label': m.get('label'), 'hard_no': m.get('hard_no')} for m in chunk if 'score' in m])
        print(f'  scored batch {k // batch + 1}/{(len(todo) + batch - 1) // batch}', flush=True)
    sc = sorted([m for m in cat['moments'].values() if m.get('score')], key=lambda m: -m['score'])
    print(f"{slug}: {len(sc)} scored; top: " + ' | '.join(f"{m['score']} {m['label']} {m['tags']}" for m in sc[:8]))
    return cat


def themes_ready(slug, min_score=7, n=5, used=()):
    """Tags with at least n strong, safe moments not used in an earlier ranking: what a ranking could be about now."""
    cf = os.path.join(CAT, slug, 'catalog.json')
    if not os.path.exists(cf):
        return []
    ms = [m for m in json.load(open(cf))['moments'].values() if (m.get('score') or 0) >= min_score and not m.get('hard_no') and m['id'] not in used]
    count = {t: sum(t in (m.get('tags') or []) for m in ms) for t in TAGS}
    return sorted([t for t, c in count.items() if c >= n], key=lambda t: -count[t])


EXCLUDE = ('gun', 'gunpoint', 'police', 'cops', 'arrest', 'fight', 'slap', 'punch', 'blood', 'swat')   # not for a fun countdown


def _ts(m):
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat(str(m.get('created')).replace('Z', '+00:00')).timestamp()
    except (TypeError, ValueError):
        return None


def rank(slug, theme, title, min_score, n, per_stream, judge_mode):
    import rankcut
    cat = json.load(open(os.path.join(CAT, slug, 'catalog.json')))
    pool = [m for m in cat['moments'].values() if (m.get('score') or 0) >= min_score and not m.get('hard_no') and theme in (m.get('tags') or [])
            and not any(w in f"{m.get('label', '')} {m.get('title', '')}".lower() for w in EXCLUDE)]
    pool.sort(key=lambda m: (-m['score'], -m['views']))
    pick, by_stream = [], {}
    for m in pool:
        s = m.get('stream') or m['id']
        if by_stream.get(s, 0) >= per_stream:
            continue
        # two viewer clips of one moment (same stream, made within 5 minutes) are one moment
        if any(p.get('stream') and p.get('stream') == m.get('stream') and _ts(p) and _ts(m) and abs(_ts(p) - _ts(m)) < 300 for p in pick):
            continue
        by_stream[s] = by_stream.get(s, 0) + 1; pick.append(m)
        if len(pick) == n:
            break
    if len(pick) < 3:
        raise SystemExit(f'only {len(pick)} {theme} moments score {min_score}+: grow the catalog first')
    rid = f"{slug}-{theme}-{time.strftime('%Y%m%d')}"
    # "this month" only when every moment really is from the last 35 days (the catalog mixes in all-time clips)
    import datetime as _dt
    recent = all(m.get('date') and (_dt.date.today() - _dt.date.fromisoformat(m['date'])).days <= 35 for m in pick)
    when = ' THIS MONTH' if recent else ''
    # best last: #1 is the highest score
    parts, items = [], []
    for k, m in enumerate(sorted(pick, key=lambda m: (m['score'], m['views']))):
        parts.append({'id': f"{rid}-{k}", 'streamer': cat['name'], 'slug': slug, 'platform': cat['platform'], 'url': m['url'], 'title': m['title'],
                      'views': m['views'], 'source_url': m['source_url'], 'credit': f"{'kick.com' if cat['platform'] == 'kick' else 'twitch.tv'}/{slug}", 'date': m.get('date')})
        items.append({'part': k, 'rank': len(pick) - k, 'label': m['label'], 'span': m['span'], 'evidence': m['evidence']})
    spec = {'id': rid, 'streamer': cat['name'], 'slug': slug, 'platform': cat['platform'], 'credit': parts[0]['credit'],
            'program': 'Kick Clipping (clipping.net) bounty' if cat['platform'] == 'kick' else None, 'parts': parts,
            'preset': {'title': title or f"{cat['name'].upper()}'S {THEMES[theme]}{when}", 'items': items,
                       'caption': f"{cat['name']}'s {THEMES[theme].lower()}{when.lower()}, ranked #{re.sub(r'[^a-z0-9]', '', cat['name'].lower())} #streamer #fyp"}}
    sf = os.path.join(C.OUT, f'{rid}.json'); json.dump(spec, open(sf, 'w'), indent=1, ensure_ascii=False)
    rankcut.main(sf, judge_mode)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('slug')
    ap.add_argument('--name')
    ap.add_argument('--platform', default='kick', choices=['kick', 'twitch'])
    ap.add_argument('--pool', type=int, default=60)
    ap.add_argument('--judge', default='sonnet')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--rank', choices=TAGS)
    ap.add_argument('--title')
    ap.add_argument('--min', type=float, default=7)
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--per-stream', type=int, default=2)
    a = ap.parse_args()
    if a.rank:
        rank(a.slug, a.rank, a.title, a.min, a.n, a.per_stream, a.judge)
    else:
        build(a.slug, a.name or a.slug, a.platform, a.pool, a.judge, a.batch)
