#!/usr/bin/env python3
"""Cut a vertical clip from a moment a streamer's own viewers already clipped (Kick or Twitch clips, sorted by views).

The moment is picked by the audience (the platform's most-viewed clips); everything after that is the room's own
pipeline: GPU re-transcription (asr/run.sh), the same editor as the live room (live/judge.py: score, hook, trim, grounded
in what was said), and edit.py (virtual camera or facecam split, TikTok hook box, captions, credit).

usage: clipcut.py jobs.json [--only id1,id2] [--judge sonnet] [--force]
  jobs.json: [{"id", "streamer", "slug", "platform": "kick"|"twitch", "url", "title", "views", "category",
               "program", "source_url", "credit"}]
Writes portfolio/clips/<id>/{<id>.mp4, <id>.jpg, <id>.json, src.mp4, words.json, spec.json, editor.json}
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, '.venv', 'bin', 'python')
YTDLP = os.path.join(ROOT, '.venv', 'bin', 'yt-dlp')
FF = os.path.join(ROOT, '.venv', 'bin', 'ffmpeg')
OUT = os.path.join(ROOT, 'portfolio', 'clips')
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'live'))


def ff(*a):
    r = subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', *map(str, a)], capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='ignore')[-300:])
    return r.stdout


def duration(path):
    r = subprocess.run([FF, '-i', path], capture_output=True, text=True)
    import re
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def fetch(job, d):
    src = os.path.join(d, 'src.mp4')
    if not os.path.exists(src) and job.get('local') and os.path.exists(job['local']):
        os.link(job['local'], src)                      # already downloaded for another cut
    if not os.path.exists(src):
        r = subprocess.run([YTDLP, '-q', '--no-warnings', '--ffmpeg-location', FF, '-f', 'bv*[height<=1080]+ba/b[height<=1080]/b', '--merge-output-format', 'mp4',
                            '-o', src, job['url']], capture_output=True, text=True, timeout=600)
        if r.returncode or not os.path.exists(src):
            raise RuntimeError(f'download failed: {r.stderr[-300:]}')
    aud = os.path.join(d, 'src.m4a')
    if not os.path.exists(aud):
        ff('-i', src, '-vn', '-c:a', 'aac', '-b:a', '192k', aud)
    return src, aud


def cut(job, judge_mode='sonnet', force=False, rejudge=False):
    import rehear, judge as J
    d = os.path.join(OUT, job['id']); os.makedirs(d, exist_ok=True)
    final = os.path.join(d, job['id'] + '.mp4')
    if os.path.exists(final) and not force:
        print(f"{job['id']}: already cut"); return json.load(open(os.path.join(d, 'editor.json')))
    src, aud = fetch(job, d)
    dur = duration(src)
    ej = os.path.join(d, 'editor.json')
    if os.path.exists(ej) and not rejudge:
        v = json.load(open(ej))                       # the editor already read this moment: reuse its call
        return render(job, d, src, dur, v) if v['keep'] or force else v
    words = []
    for t in range(0, int(dur) + 1, 30):                # the ASR server takes ~30 s at a time
        if dur - t > 0.5:
            words += rehear.hear_safe(aud, t, min(dur, t + 30))
    json.dump({'segments': [{'start': words[0]['s'] if words else 0, 'end': words[-1]['e'] if words else 0,
                             'text': ' '.join(w['w'] for w in words), 'words': words}]}, open(os.path.join(d, 'words.json'), 'w'))
    # lines of ~8 s for the editor, timed from the clip start
    lines, cur = [], []
    for w in words:
        if cur and (w['s'] - cur[0]['s'] > 8 or w['s'] - cur[-1]['e'] > 1.2):
            lines.append({'t': round(cur[0]['s'], 1), 'text': ' '.join(x['w'] for x in cur)}); cur = []
        cur.append(w)
    if cur:
        lines.append({'t': round(cur[0]['s'], 1), 'text': ' '.join(x['w'] for x in cur)})
    frames = []
    for f in (0.15, 0.4, 0.65, 0.88):
        try:
            frames.append(ff('-ss', f'{dur * f:.2f}', '-i', src, '-frames:v', 1, '-vf', 'scale=480:-2', '-q:v', 6, '-f', 'image2pipe', '-c:v', 'mjpeg', '-'))
        except RuntimeError:
            pass
    payload = {'streamer': job['streamer'], 'title': job.get('title') or '', 'slug': job['slug'], 'lines': lines,
               'chat_top': [], 'signals': {'source': f"{job['platform'].title()} clip made by a viewer, {job.get('views', 0):,} views",
                                           'score': 8, 'clip_it': 3, 'laughs': 0, 'chat_x': 0},
               'duration': round(dur, 1)}
    v = J.judge(payload, judge_mode, frames)
    v['heard'] = ' '.join(ln['text'] for ln in lines)[:600]
    json.dump(v, open(os.path.join(d, 'editor.json'), 'w'), indent=1, ensure_ascii=False)
    print(f"{job['id']}: {v.get('why')} | hook: {v.get('hook')}", flush=True)
    if not v['keep'] and not force:
        return v
    return render(job, d, src, dur, v)


def render(job, d, src, dur, v):
    a, b = 0.0, dur
    if v.get('trim'):
        a, b = max(0.0, float(v['trim'][0])), min(dur, float(v['trim'][1]) or dur)
    elif dur > 50:
        a = dur - 42   # viewers press clip right after the payoff: keep the end
    po = v.get('payoff')                           # the editor's payoff second: ≤25 s of build-up, a hard cut ~4 s after it
    if po is not None and a < po < b:
        a2, b2 = max(a, po - 25), min(b, po + 4)
        if b2 - a2 >= 10:
            a, b = a2, b2
    import edit as E_                                   # the end on a finished line, never mid-sentence
    try:
        ws_ = [w for sg in json.load(open(os.path.join(d, 'words.json'))).get('segments', []) for w in sg.get('words', [])]
        b = min(dur, E_.snap_end(ws_, max(a + 8, (po if po is not None else b) + 0.5), b, min(dur, b + 3.5)))
    except (OSError, ValueError):
        pass
    a = max(a, 0.4)    # platform clips open on a black or half-decoded frame
    import edit as E, facecam
    # the fixed corner cam (xQc-style) must show in at least 2 of 3 samples: a dark corner of a night IRL shot can
    # look like one once
    votes = [E.side_at(src, a + (b - a) * t) for t in (0.3, 0.5, 0.7)]
    side = next((v for v in set(votes) if v and votes.count(v) >= 2), None)
    cam = None if side else (facecam.detect(src, a, b) or facecam.detect(src, a + (b - a) * 0.45, b))   # a facecam box; also just in the payoff half
    import layouts
    try:
        words_text = ' '.join(w['w'] for sg in json.load(open(os.path.join(d, 'words.json'))).get('segments', []) for w in sg.get('words', []) if a <= w['s'] <= b)
    except (OSError, ValueError):
        words_text = ''
    layout, extra = layouts.choose(src, a, b, job.get('slug'), words_text, side, cam, job.get('category'), job.get('title') or '', job['id'])
    layout = job.get('layout') or layout
    beat = None
    spec = {'id': job['id'], 'video': 'src.mp4', 'audio': 'src.m4a', 'words': 'words.json', 'out_dir': '.', 'layout': layout,
            'segments': [[round(a, 2), round(b, 2)]], 'hook': v['hook'], 'punch': [beat] if beat else [],
            'credit': job.get('credit') or f"{job['platform']}.com/{job['slug']}", 'program': job.get('program'),
            'source_url': job.get('source_url'), 'post_caption': v.get('caption'),
            'rules_check': {'source': f"{job['platform']} clip of a live stream", 'credit_on_screen': True, 'no_ai_video': True}}
    if cam and layout == 'hooksplit':
        spec['cam'] = {k: cam[k] for k in ('x', 'y', 'w', 'h', 'side')}
        spec['screen'] = cam['screen']
    spec.update(extra)
    json.dump(spec, open(os.path.join(d, 'spec.json'), 'w'), indent=1, ensure_ascii=False)
    r = subprocess.run(['nice', '-n', '10', PY, os.path.join(HERE, 'edit.py'), os.path.join(d, 'spec.json')], capture_output=True, text=True,
                       env={**os.environ, 'CUTROOM_FF_THREADS': '4'})
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    v.update({'layout': layout, 'src': [a, b]})
    json.dump(v, open(os.path.join(d, 'editor.json'), 'w'), indent=1, ensure_ascii=False)
    print(f"{job['id']}: cut {b - a:.0f} s {layout}", flush=True)
    return v


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs')
    ap.add_argument('--only', default='')
    ap.add_argument('--judge', default='sonnet')
    ap.add_argument('--force', action='store_true', help='re-render (and keep moments the editor dropped)')
    ap.add_argument('--rejudge', action='store_true', help='ask the editor again instead of reusing editor.json')
    a = ap.parse_args()
    for job in json.load(open(a.jobs)):
        if a.only and job['id'] not in a.only.split(','):
            continue
        try:
            cut(job, a.judge, a.force, a.rejudge)
        except Exception as e:  # noqa: BLE001
            print(f"{job['id']}: FAILED {str(e)[:300]}", flush=True)
