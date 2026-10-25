#!/usr/bin/env python3
"""Find where something was said in a long video (a VOD, a full-stream upload): transcribe it once on the GPU ASR
(cached next to the file as <name>.words.json), then print the windows where the keywords come up.

Used to locate a story beat inside an hour-long upload before downloading just that section in 1080p.

usage: vodfind.py <url-or-file> <keyword> [<keyword> ...] [--pad 20]
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clipcut as C  # noqa: E402

VOD = os.path.join(os.path.dirname(HERE), 'portfolio', 'vod')


def local(src):
    if os.path.exists(src):
        return src
    os.makedirs(VOD, exist_ok=True)
    vid = re.sub(r'[^A-Za-z0-9_-]', '', src.rstrip('/').rsplit('/', 1)[-1].split('=')[-1])[:40]
    f = os.path.join(VOD, f'{vid}.360.mp4')
    if not os.path.exists(f):
        subprocess.run([C.YTDLP, '-q', '--no-warnings', '--ffmpeg-location', C.FF, '-f', 'b[height<=360]/bv*[height<=360]+ba/w',
                        '--merge-output-format', 'mp4', '-o', f, src], check=True, timeout=1800)
    return f


def words(f):
    import rehear
    wf = f.rsplit('.', 1)[0] + '.words.json'
    if os.path.exists(wf):
        return json.load(open(wf))
    wav = f.rsplit('.', 1)[0] + '.wav'
    if not os.path.exists(wav):
        C.ff('-i', f, '-vn', '-ac', 1, '-ar', 16000, wav)
    dur = C.duration(wav)
    ws = []
    for t in range(0, int(dur) + 1, 30):
        if dur - t > 0.5:
            ws += rehear.hear_safe(wav, t, min(dur, t + 30))
    json.dump(ws, open(wf, 'w'))
    os.unlink(wav)
    return ws


def find(ws, keys, pad=20):
    hits = [w['s'] for w in ws if any(k.lower() in w['w'].lower() for k in keys)]
    wins = []
    for t in hits:
        if wins and t - wins[-1][1] < pad:
            wins[-1][1] = t; wins[-1][2] += 1
        else:
            wins.append([t, t, 1])
    out = []
    for a, b, n in sorted(wins, key=lambda w: -w[2]):
        text = ' '.join(w['w'] for w in ws if a - 6 <= w['s'] <= b + 6)
        out.append({'start': round(max(0, a - pad), 1), 'end': round(b + pad, 1), 'hits': n, 'text': text[:300]})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('keys', nargs='+')
    ap.add_argument('--pad', type=float, default=20)
    a = ap.parse_args()
    f = local(a.src)
    for w in find(words(f), a.keys, a.pad)[:8]:
        print(f"{w['start']:7.1f}-{w['end']:7.1f}  x{w['hits']}  {w['text']}")
