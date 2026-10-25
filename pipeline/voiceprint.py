#!/usr/bin/env python3
"""The streamer's voice, learned from their own past sessions, so the editor knows who is talking.

A stream is not one person: Clavicular's collabs put his friends on mic, and a friend's line once went out captioned as his
(2026-10-17). The transcript doesn't say who talks. This learns each streamer's voiceprint from the live room's own clips:
every transcribed line (1.5-12 s) is embedded with Resemblyzer (a small speaker-embedding model, CPU), and the streamer is
the voice that comes back across the most SESSIONS (friends and guests come and go; the streamer is in all of them).

build(slug)  -> runs/voiceprints/<slug>.npz: centroid, threshold, how many lines/sessions it rests on
label(slug, pcm16k, spans) -> ['streamer' | 'other' | '?' per span]   (live.py tags each line it sends to the editor)

usage: voiceprint.py build SLUG [SLUG ...]      voiceprint.py check SLUG CLIP_DIR
"""
import glob, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(HERE, 'live', 'runs')
OUT = os.path.join(RUNS, 'voiceprints')
FF = os.path.join(ROOT, '.venv', 'bin', 'ffmpeg')
SR = 16000
_enc = None
_vp = {}


def encoder():
    global _enc
    if _enc is None:
        from resemblyzer import VoiceEncoder
        _enc = VoiceEncoder('cpu', verbose=False)
    return _enc


def embed(x):
    """One utterance (float32 16 kHz mono) -> unit 256-d embedding, or None when there's too little voiced audio."""
    from resemblyzer import preprocess_wav
    try:
        w = preprocess_wav(x, source_sr=SR)                  # volume-normalised, long silences trimmed (webrtcvad)
    except Exception:  # noqa: BLE001
        return None
    if len(w) < SR * 1.0:
        return None
    e = encoder().embed_utterance(w)
    return e / (np.linalg.norm(e) + 1e-9)


def audio(path):
    raw = subprocess.run([FF, '-nostdin', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR), '-f', 's16le', '-'],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def run_of(d):
    """runs/<run>/clips/<id> -> <run> (the project folder is itself called clips, so no string splitting)"""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.normpath(d))))


def lines(words, gap=0.45, longest=8.0):
    """Words -> spoken lines: a break at a sentence end, a pause of `gap` s, or when a line gets longer than `longest` s
    (the live transcript keeps a whole cut as one segment, so lines are made here)."""
    out, cur = [], []
    for w in sorted(words, key=lambda w: w['s']):
        if cur and (str(cur[-1]['w']).rstrip().endswith(('.', '?', '!')) or w['s'] - cur[-1]['e'] > gap or w['e'] - cur[0]['s'] > longest):
            out.append(cur); cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def utterances(slug, max_clips=160, per_clip=6):
    """(session, embedding) for transcribed lines in the room's finished cuts (their raw source audio isn't kept): whisper
    segments of 4+ words, mapped from source time into the cut through its sheet's timeline, 1.5-12 s of it."""
    dirs = []
    for d in sorted(glob.glob(os.path.join(RUNS, f'{slug}-2*', 'clips', '*'))):
        cid = os.path.basename(d)
        if all(os.path.exists(os.path.join(d, f)) for f in (f'{cid}.mp4', 'words.json', f'{cid}.json')):
            dirs.append(d)
    by_run = {}                                              # spread over sessions: round robin, newest first
    for d in dirs:
        by_run.setdefault(run_of(d), []).append(d)
    order, k = [], 0
    while len(order) < max_clips and any(len(v) > k for v in by_run.values()):
        order += [v[-1 - k] for v in by_run.values() if len(v) > k]; k += 1
    out = []
    for d in order[:max_clips]:
        cid = os.path.basename(d)
        try:
            W = json.load(open(os.path.join(d, 'words.json')))
            tl = [t for t in json.load(open(os.path.join(d, f'{cid}.json'))).get('timeline', []) if 'src' in t]
            x = audio(os.path.join(d, f'{cid}.mp4'))
        except Exception:  # noqa: BLE001
            continue
        n = 0
        for ws in lines([w for sg in W.get('segments', []) for w in sg.get('words') or []]):
            if len(ws) < 4 or n >= per_clip:
                continue
            a, b = ws[0]['s'], ws[-1]['e']
            piece = next((t for t in tl if t['src'][0] - 0.05 <= a < t['src'][1]), None)
            if not piece:
                continue
            e0 = piece['at'] + max(0.0, a - piece['src'][0]); e1 = piece['at'] + min(b, piece['src'][1]) - piece['src'][0]
            if not 1.5 <= e1 - e0 <= 12:
                continue
            e = embed(x[int(e0 * SR):int(e1 * SR)])
            if e is not None:
                out.append((run_of(d), e)); n += 1
    return out


def build(slug):
    U = utterances(slug)
    if len(U) < 20:
        print(f'{slug}: only {len(U)} usable lines, no voiceprint'); return None
    sess = np.array([u[0] for u in U]); E = np.stack([u[1] for u in U])
    S = E @ E.T
    n_sess = len(set(sess))
    # each line's support: in how many OTHER sessions a close voice (cos >= 0.78) speaks
    support = np.array([len({sess[j] for j in np.flatnonzero(S[i] >= 0.78) if sess[j] != sess[i]}) for i in range(len(U))])
    seed = np.argsort(-support)[:max(8, len(U) // 5)]
    c = E[seed].mean(0); c /= np.linalg.norm(c)
    for _ in range(4):                                       # refine: the members are the lines close to the centroid
        sim = E @ c
        mem = sim >= np.percentile(sim, 40)
        c = E[mem].mean(0); c /= np.linalg.norm(c)
    sim = E @ c
    # threshold between the voice's own lines and everyone else's (2-means on the similarities)
    lo, hi = np.percentile(sim, 20), np.percentile(sim, 80)
    for _ in range(20):
        t = (lo + hi) / 2
        lo, hi = sim[sim < t].mean() if (sim < t).any() else lo, sim[sim >= t].mean() if (sim >= t).any() else hi
    thr = float(np.clip((lo + hi) / 2, 0.62, 0.86))
    own = sim >= thr
    own_sess = len(set(sess[own]))
    info = {'slug': slug, 'lines': int(len(U)), 'sessions': int(n_sess), 'own_lines': int(own.sum()), 'own_sessions': own_sess,
            'share': round(float(own.mean()), 2), 'threshold': round(thr, 3), 'gap': round(float(hi - lo), 3)}
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'{slug}.npz'), centroid=c, threshold=thr, info=json.dumps(info))
    print(info)
    return info


def load(slug):
    if slug not in _vp:
        f = os.path.join(OUT, f'{slug}.npz')
        if os.path.exists(f):
            z = np.load(f)
            _vp[slug] = (z['centroid'], float(z['threshold']))
        else:
            _vp[slug] = None
    return _vp[slug]


# the regular cast's own voices, learned from THEIR channels (a co-host's lines once went out as the streamer's: Hexumlite's
# "best conversation of the night" was captioned as Clavicular's, 2026-10-19). cast.json: {streamer slug: {name: print}}
CAST = os.path.join(OUT, 'cast')


def build_cast(name, wavs, win=3.0):
    """A cast member's voiceprint from audio of their own channel (their clips): 3 s windows, and their voice is the one
    that recurs across the most clips (guests come and go)."""
    E, src = [], []
    for i, f in enumerate(wavs):
        x = audio(f)
        for t in np.arange(0, max(0, len(x) / SR - win), win):
            e = embed(x[int(t * SR):int((t + win) * SR)])
            if e is not None:
                E.append(e); src.append(i)
    E, src = np.array(E), np.array(src)
    S = E @ E.T
    support = np.array([len({src[j] for j in np.flatnonzero(S[i] >= 0.78) if src[j] != src[i]}) for i in range(len(E))])
    own = support >= max(2, np.percentile(support, 60))
    c = E[own].mean(0); c /= np.linalg.norm(c)
    sims = E @ c
    lo, hi = float(np.median(sims[~own])) if (~own).any() else 0.5, float(np.median(sims[own]))
    thr = float(np.clip((lo + hi) / 2, 0.62, 0.86))
    os.makedirs(CAST, exist_ok=True)
    info = {'name': name, 'windows': int(len(E)), 'own': int(own.sum()), 'clips': int(len(wavs)), 'threshold': round(thr, 3), 'gap': round(hi - lo, 3)}
    np.savez(os.path.join(CAST, f'{name.lower()}.npz'), centroid=c, threshold=thr, info=json.dumps(info))
    print(info)
    return info


def cast_prints(slug):
    """{display name: (centroid, threshold)} for the streamer's regulars that have a voiceprint."""
    key = ('cast', slug)
    if key not in _vp:
        out = {}
        try:
            for nm, f in (json.load(open(os.path.join(CAST, 'cast.json'))).get(slug) or {}).items():
                z = np.load(os.path.join(CAST, f'{f}.npz'))
                out[nm] = (z['centroid'], float(z['threshold']))
        except (OSError, ValueError):
            pass
        _vp[key] = out
    return _vp[key]


def label(slug, x, spans, margin=0.06, lead=0.03):
    """x: float32 16 kHz mono; spans: [(a, b)] seconds into x. -> per span: 'streamer', a cast member's name (their own
    voiceprint fits better), 'other', or '?' (too short, no voiceprint, or too close to call)."""
    vp, cast = load(slug), cast_prints(slug)
    # a co-host whose print sits this close to the streamer's (Hexumlite vs Clavicular: 0.95 / 0.91 with Resemblyzer) can't
    # be told apart by voice: then the voice never vouches for the streamer, it only says "unclear" (or "other" for neither)
    twin = bool(vp) and any(float(c @ vp[0]) > 0.88 for c, _ in cast.values())
    out = []
    for a, b in spans:
        if not vp or b - a < 1.2:
            out.append('?'); continue
        e = embed(x[int(max(0, a) * SR):int(b * SR)])
        if e is None:
            out.append('?'); continue
        s = float(e @ vp[0])
        best = max(((float(e @ c), t, nm) for nm, (c, t) in cast.items()), default=(0.0, 1.0, None))
        if best[2] and best[0] >= best[1] and best[0] > s + lead:
            out.append(best[2])                          # the co-host's own voice fits better than the streamer's
        elif s >= vp[1] and s > best[0] + lead:
            out.append('?' if twin else 'streamer')
        elif s < vp[1] - margin and (not best[2] or best[0] < best[1] - margin):
            out.append('other')
        else:
            out.append('?')
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        for s in sys.argv[2:]:
            build(s)
    elif sys.argv[1] == 'check':
        slug = sys.argv[2]
        for d in sys.argv[3:]:
            cid = os.path.basename(os.path.normpath(d))
            W = json.load(open(os.path.join(d, 'words.json')))
            tl = [t for t in json.load(open(os.path.join(d, f'{cid}.json'))).get('timeline', []) if 'src' in t]
            x = audio(os.path.join(d, f'{cid}.mp4'))
            print('==', cid)
            for ws in lines([w for sg in W['segments'] for w in sg.get('words') or []]):
                a, b = ws[0]['s'], ws[-1]['e']
                piece = next((t for t in tl if t['src'][0] - 0.05 <= a < t['src'][1]), None)
                if not piece:
                    continue
                e0 = piece['at'] + max(0.0, a - piece['src'][0]); e1 = piece['at'] + min(b, piece['src'][1]) - piece['src'][0]
                print(f"  {label(slug, x, [(e0, e1)])[0]:9s} {' '.join(w['w'] for w in ws)[:100]}")
