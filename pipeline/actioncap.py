#!/usr/bin/env python3
"""Action captions: the stage directions the biggest YouTube Shorts pages put on a clip ("*NOTICES*", "*GOES TO EMILY*",
"*RAKAI'S PS5*", "*GRABS*", "*RUNS*", "*FALLS*"; 94-164M views each, research/references/STYLE-GUIDE.md). They narrate what
happens so the clip works with the sound off, like a video game or a roleplay; the key spoken line goes in caps too.

labels(video, a, b, words, hook) -> [{"t0", "t1", "text", "kind": "action"|"line"}]  (seconds from a)
The editor watches the stretch at 2 frames a second (sheets of 8 frames, each stamped with its time) and the transcript.
usage: actioncap.py VIDEO A B [words.json]
"""
import json, os, re, sys
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'live'))

RULES = """You caption a short clip the way the biggest YouTube Shorts pages do (Succesful Celebrity, Blessed, KaiNFriends:
90-160M views): short STAGE DIRECTIONS in asterisks that narrate what happens, so a viewer follows it with the sound off,
like a video game or a roleplay: *NOTICES*, *WALKS IN*, *GRABS PHONE*, *RUNS*, *FALLS*, *LAUGHING*, *AHH!*, *RAKAI'S PS5*.
Also the one or two spoken lines that carry the joke, in caps, without asterisks (NO CRYING, WHOSE ARM IS THIS?).
You get the frames in order, each stamped with its time in seconds, and the transcript with word times.
Rules:
- 6-14 labels across the clip, 1-4 words each, in time order, each on screen while it's happening (t0-t1, 0.6-2.5 s).
- Only what is plainly visible in the frames or said in the transcript. Never invent, never guess a name you can't see or
  hear, never describe bodies or appearance. Verbs in present tense (*GRABS*, not *GRABBED*).
- Leave quiet stretches without a label rather than filling them.
Answer with JSON only: {"labels": [{"t0": s, "t1": s, "text": "*GRABS*", "kind": "action"}, {"t0": s, "t1": s, "text": "NO CRYING", "kind": "line"}]}"""


def sheets(video, a, b, fps=2.0, per=8, w=360):
    c = cv2.VideoCapture(video); out, cur = [], []
    t = a
    while t < b:
        c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = c.read()
        if not ok:
            break
        h0, w0 = f.shape[:2]
        f = cv2.resize(f, (w, int(w * h0 / w0)))
        cv2.putText(f, f'{t - a:.1f}s', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(f, f'{t - a:.1f}s', (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cur.append(f)
        if len(cur) == per:
            out.append(cur); cur = []
        t += 1 / fps
    if cur:
        out.append(cur)
    imgs = []
    for g in out:
        while len(g) % 4:
            g.append(np.zeros_like(g[0]))
        rows = [np.hstack(g[i:i + 4]) for i in range(0, len(g), 4)]
        imgs.append(cv2.imencode('.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    return imgs


def labels(video, a, b, words=(), hook=''):
    import judge as J
    ws = [w for w in words if a - 0.2 <= w['s'] <= b]
    tx = ' '.join(f"{w['w']}@{w['s'] - a:.1f}" for w in ws)[:2500] or '(no speech)'
    prompt = RULES + f"\n\nTHE CLIP'S HOOK: {hook}\nLENGTH: {b - a:.1f} s\nTRANSCRIPT (word@seconds):\n{tx}"
    imgs = sheets(video, a, b)[:6]
    for extra in ('', '\n\nYour last answer was not valid JSON. Answer again, JSON only.'):
        raw = J._ask(prompt + extra, imgs, 150)
        m = re.search(r'\{.*\}', raw, re.S)
        try:
            out = json.loads(m.group(0)).get('labels') if m else None
        except ValueError:
            out = None
        if out:
            break
    good = []
    for x in out or []:
        try:
            t0, t1 = float(x['t0']), float(x['t1'])
        except (KeyError, TypeError, ValueError):
            continue
        txt = re.sub(r'\s+', ' ', str(x.get('text') or '')).strip().upper()
        if not txt or len(txt.split()) > 5 or not (0 <= t0 < t1 <= b - a + 0.5):
            continue
        good.append({'t0': round(t0, 2), 't1': round(min(t1, t0 + 2.8), 2), 'text': txt, 'kind': 'line' if x.get('kind') == 'line' else 'action'})
    good.sort(key=lambda x: x['t0'])
    for p, q in zip(good, good[1:]):                      # one label at a time
        p['t1'] = min(p['t1'], q['t0'])
    return [x for x in good if x['t1'] - x['t0'] >= 0.3]


if __name__ == '__main__':
    v, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    W = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else {}
    ws = [w for s in W.get('segments', []) for w in s.get('words', [])] if isinstance(W, dict) else W
    for x in labels(v, a, b, ws):
        print(x)
