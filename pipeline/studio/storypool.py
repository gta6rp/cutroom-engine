#!/usr/bin/env python3
"""storypool: footage for one storyline -> STORY_DIR/src/*.mp4 + pool.json, for storyedit.py.

Searches the streamer's own clips for the storyline's keywords (Twitch: the channel's all-time clips through yt-dlp;
Kick: the clips API, most viewed, all time), ranks them by keyword hits and views, downloads the best at up to 1080p,
and adds any long videos given with --video (a YouTube recap, the stream VOD).

usage: storypool.py STORY_DIR --platform twitch --channel kaicenat --keywords "million,1m,mafiathon" [--max 14]
                    [--video URL --video-note "..."]...
"""
import argparse, json, math, os, re, subprocess, sys, time, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
YTDLP = os.path.join(ROOT, '.venv', 'bin', 'yt-dlp')
FF = os.path.join(ROOT, '.venv', 'bin', 'ffmpeg')


def twitch_clips(channel):
    r = subprocess.run([YTDLP, '--flat-playlist', '-J', f'https://www.twitch.tv/{channel}/clips?filter=clips&range=all'],
                       capture_output=True, text=True, timeout=300)
    d = json.loads(r.stdout or '{}')
    return [{'title': e.get('title') or '', 'views': e.get('view_count') or 0, 'dur': e.get('duration') or 0, 'url': e.get('url'), 'date': None}
            for e in d.get('entries') or []]


def kick_clips(channel, pages=15):
    from curl_cffi import requests as rq
    out, cur = [], None
    for _ in range(pages):
        u = f'https://kick.com/api/v2/channels/{channel}/clips?sort=view&time=all' + (f'&cursor={urllib.parse.quote(cur)}' if cur else '')
        r = rq.get(u, impersonate='chrome', timeout=20)
        if r.status_code != 200:
            break
        j = r.json()
        out += [{'title': c.get('title') or '', 'views': c.get('view_count') or 0, 'dur': c.get('duration') or 0,
                 'url': f"https://kick.com/{channel}/clips/{c['id']}", 'date': (c.get('created_at') or '')[:10]} for c in j.get('clips') or []]
        cur = j.get('nextCursor')
        if not cur:
            break
        time.sleep(0.6)
    return out


def slug(t, used):
    s = re.sub(r'[^a-z0-9]+', '-', t.lower()).strip('-')[:28] or 'clip'
    k, n = s, 2
    while k in used:
        k = f'{s}-{n}'; n += 1
    used.add(k)
    return k


def fetch(url, out):
    subprocess.run([YTDLP, '-q', '--no-warnings', '--ffmpeg-location', FF, '-f', 'bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]/b',
                    '--merge-output-format', 'mp4', '-o', out, url], capture_output=True, timeout=1800)
    return os.path.exists(out)


def upload_date(url):
    r = subprocess.run([YTDLP, '-J', '--skip-download', url], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(r.stdout)
        ud = d.get('upload_date') or (time.strftime('%Y%m%d', time.gmtime(d['timestamp'])) if d.get('timestamp') else '')
        return f'{ud[:4]}-{ud[4:6]}-{ud[6:8]}' if ud else None
    except (ValueError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('story'); ap.add_argument('--platform', choices=['twitch', 'kick'], required=True); ap.add_argument('--channel', required=True)
    ap.add_argument('--keywords', required=True); ap.add_argument('--exclude', default='')
    ap.add_argument('--max', type=int, default=14); ap.add_argument('--min-dur', type=float, default=6)
    ap.add_argument('--video', action='append', default=[]); ap.add_argument('--video-note', action='append', default=[])
    ap.add_argument('--since', default=''); ap.add_argument('--until', default='', help='keep clips dated in this window (YYYY-MM-DD)')
    ap.add_argument('--yt-search', default='', help='also take the top YouTube recaps for this search (dated in the window)')
    ap.add_argument('--yt-max', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.story, 'src'), exist_ok=True)
    kws = [k.strip().lower() for k in a.keywords.split(',') if k.strip()]
    bad = [k.strip().lower() for k in a.exclude.split(',') if k.strip()]
    clips = twitch_clips(a.channel) if a.platform == 'twitch' else kick_clips(a.channel)
    hits = []
    for c in clips:
        t = c['title'].lower()
        n = sum(1 for k in kws if re.search(r'\b' + re.escape(k) + r'\b', t))
        if n and not any(b in t for b in bad) and c['dur'] >= a.min_dur:
            hits.append((n + math.log10(max(10, c['views'])) / 2, c))
    hits.sort(key=lambda x: -x[0])
    print(f'{len(clips)} clips, {len(hits)} match the storyline')
    pf = os.path.join(a.story, 'pool.json')
    pool = json.load(open(pf)) if os.path.exists(pf) else []
    used = {p['id'] for p in pool}
    have = {p.get('url') for p in pool}
    inwin = lambda d: bool(d) and (not a.since or d >= a.since) and (not a.until or d <= a.until)
    if a.since or a.until:                               # date the best candidates (Twitch listings carry no dates)
        kept = []
        for sc, c in hits[: a.max * 4]:
            c['date'] = c['date'] or upload_date(c['url'])
            if inwin(c['date']):
                kept.append((sc, c))
        hits = kept
        print(f'{len(hits)} in {a.since}..{a.until}')
    if a.yt_search:
        r = subprocess.run([YTDLP, '--flat-playlist', '-J', f'ytsearch12:{a.yt_search}'], capture_output=True, text=True, timeout=120)
        n = 0
        for e in (json.loads(r.stdout or '{}').get('entries') or []):
            if n >= a.yt_max or not (60 <= (e.get('duration') or 0) <= 1800):
                continue
            u = e.get('url') or f"https://www.youtube.com/watch?v={e['id']}"
            d = upload_date(u)
            if (a.since or a.until) and not inwin(d):
                continue
            a.video.append(u); a.video_note.append(f"YouTube: '{e.get('title')}' ({e.get('view_count') or 0:,} views)"); n += 1
    for _, c in hits[: a.max]:
        if c['url'] in have:
            continue
        cid = slug(c['title'], used)
        out = os.path.join(a.story, 'src', f'{cid}.mp4')
        if fetch(c['url'], out):
            date = c['date'] or upload_date(c['url'])
            pool.append({'id': cid, 'file': f'src/{cid}.mp4', 'date': date, 'url': c['url'], 'views': c['views'], 'note': f"{a.platform} clip '{c['title']}' ({c['views']:,} views)"})
            print(f"  + {cid}  {c['views']:,} views  {date}  {c['title'][:60]}")
    for k, v in enumerate(a.video):
        if v in have:
            continue
        cid = slug(f'video-{k}', used)
        out = os.path.join(a.story, 'src', f'{cid}.mp4')
        if fetch(v, out):
            pool.append({'id': cid, 'file': f'src/{cid}.mp4', 'date': upload_date(v), 'url': v, 'note': (a.video_note[k] if k < len(a.video_note) else 'video')})
            print(f'  + {cid}  {v}')
    json.dump(pool, open(pf, 'w'), indent=1, ensure_ascii=False)
    print(f'{len(pool)} sources in {pf}')


if __name__ == '__main__':
    main()
