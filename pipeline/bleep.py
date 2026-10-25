#!/usr/bin/env python3
"""bleep: put a bleep over every slur in a finished cut, using the cut's own words.json and spec.json (edit.py output).

edit.py now bleeps while it renders; this fixes cuts made before that, in place. A cut with nothing to bleep is untouched.
usage: bleep.py CUT_DIR [CUT_DIR...]      (each dir has <id>.mp4, spec.json, words.json)
       bleep.py --scan ROOT               every cut dir under ROOT
"""
import json, os, re, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from edit import SLUR_AUDIO, audio_chain  # noqa: E402

FF = os.path.join(HERE, '..', '.venv', 'bin', 'ffmpeg')


def edit_times(d):
    """Slur spans on the cut's own timeline: the spec's segments laid end to end (cards and intros shift them)."""
    S = json.load(open(os.path.join(d, 'spec.json')))
    W = json.load(open(os.path.join(d, 'words.json')))
    words = [w for sg in W.get('segments', []) for w in sg.get('words', [])] if isinstance(W, dict) else W
    rep = os.path.join(d, S['id'] + '.json')
    timeline, tight = [], False
    if os.path.exists(rep):
        R = json.load(open(rep))
        timeline = [t for t in R.get('timeline', []) if 'src' in t]
        tight = bool((R.get('render') or {}).get('tighten'))     # jump cuts: a word belongs to the piece it starts in
    if not timeline:
        at = 0.0
        for a, b in S.get('segments') or [[0, 1e9]]:
            timeline.append({'src': [a, b], 'at': at}); at += b - a
    out = []
    for tl in timeline:
        a, b = tl['src']
        for w in words:
            if (a - 0.05 <= w['s'] < b if tight else a - 0.05 <= w['s'] and w['e'] <= b + 0.2) and SLUR_AUDIO.search(w['w']):
                out.append((max(0, w['s'] - a + tl['at'] - 0.04), min(b, w['e']) - a + tl['at'] + 0.06))
    return S['id'], out


def fix(d):
    try:
        cid, spans = edit_times(d)
    except (OSError, ValueError, KeyError):
        return None
    mp4 = os.path.join(d, cid + '.mp4')
    if not spans or not os.path.exists(mp4):
        return 0
    marker = os.path.join(d, '.bleeped')
    if os.path.exists(marker):
        return 0
    td = tempfile.mkdtemp(prefix='bleep-'); out = os.path.join(td, 'o.mp4')
    r = subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-i', mp4, '-filter_complex', audio_chain(spans, 600).replace('loudnorm=I=-14:TP=-1.5:LRA=11', 'anull'),
                        '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], capture_output=True, text=True)
    if r.returncode:
        print(d, 'failed', r.stderr[-200:]); return None
    shutil.move(out, mp4); shutil.rmtree(td, ignore_errors=True)
    open(marker, 'w').write(json.dumps(spans))
    return len(spans)


if __name__ == '__main__':
    dirs = sys.argv[1:]
    if dirs[:1] == ['--scan']:
        dirs = [os.path.dirname(p) for root in dirs[1:] for p, _, _ in [(os.path.join(r, 'spec.json'), 0, 0) for r, _, fs in os.walk(root) if 'spec.json' in fs]]
    total = 0
    for d in dirs:
        n = fix(d)
        if n:
            print(f'{d}: bleeped {n}'); total += n
    print(f'{total} bleeps')
