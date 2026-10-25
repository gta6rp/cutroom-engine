#!/usr/bin/env python3
"""Jump-cut tightening for talk clips: cut the pauses and breaths out of a finished cut, keeping every word.

Ported from the fx session's talkpolish lab (clips/video/fx/lab/talkpolish/tighten.py, 2026-10-11), where it took the
time with nobody talking from 6.6-10 s to 1.1-2.1 s on the room's own clips. The lab owns its copy. This one is
pipeline's, used by edit.py on the live room's cuts:
- the truth is the audio, not the transcript: pauses are found on a band-passed (200-3500 Hz) energy envelope in 10 ms
  frames. The pause/speech split is Otsu on the clip's own dB histogram (a room with a fan sits 15 dB over the floor
  between words), clamped to [floor + 9, ref - 12] dB, so every room gets its own threshold
- the words only veto cuts: a word's core (its start to start + 0.6 s, trimmed to where the energy ends) is speech,
  since whisper's word end often runs on through the pause after the word
- breaths (unvoiced islands quieter than speech, between two silences) count as silence
- a cut leaves 0.07 s after the word and 0.05 s before the next; `keep` holds a pause at a set length (the beat
  before a punchline)
- the voice is rebuilt as one wav with 12 ms raised-cosine crossfades, all inside silence

    spans(src, a, b, words=[...], keep=[[t0, t1, gap]]) -> {'spans': [[s, e], ...], 'cuts': [...], 'stats': {...}}
    voice_wav(src, spans, out)
usage: tighten.py SRC a b [--words words.json]
"""
import argparse, json, os, subprocess, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FF = os.path.join(HERE, '..', '.venv', 'bin', 'ffmpeg')
HOP = 0.01


def pcm(src, a, b, sr=16000, ch=1):
    raw = subprocess.run([FF, '-nostdin', '-v', 'error', '-ss', f'{max(0.0, a):.3f}', '-t', f'{b - max(0.0, a):.3f}', '-i', src, '-vn', '-ac', str(ch),
                          '-ar', str(sr), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).copy()
    return x.reshape(-1, ch) if ch > 1 else x


def envelope(x, sr=16000):
    """Band-passed (200-3500 Hz) energy in dB per 10 ms frame (25 ms window)."""
    import scipy.signal as sg
    y = sg.sosfilt(sg.butter(4, [200, 3500], 'bandpass', fs=sr, output='sos'), x)
    hop, win = int(HOP * sr), int(0.025 * sr)
    n = max(1, (len(y) - win) // hop + 1)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10((y[idx] ** 2).mean(1) + 1e-10)


def voiced(x, i0, i1, sr=16000):
    """Voiced frames (pyin, 70-500 Hz) in frames [i0, i1): only short islands are asked, so a whole clip stays cheap."""
    import librosa
    seg = x[max(0, int(i0 * HOP * sr) - 512): int(i1 * HOP * sr) + 512]
    if len(seg) < 1024:
        return 0
    _, vf, _ = librosa.pyin(seg, fmin=70, fmax=500, sr=sr, frame_length=1024, hop_length=int(HOP * sr), center=False)
    return int(np.nansum(vf))


def otsu(e, bins=100):
    h, ed = np.histogram(e, bins)
    c = (ed[:-1] + ed[1:]) / 2
    best, thr = -1.0, float(np.median(e))
    for i in range(5, bins - 5):
        w0, w1 = h[:i].sum(), h[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (h[:i] * c[:i]).sum() / w0, (h[i:] * c[i:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best:
            best, thr = v, float(c[i])
    return thr


def runs(mask):
    """[(i0, i1)] runs of True (i1 exclusive)."""
    d = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def spans(src, a, b, words=(), keep=(), min_gap=0.3, pad_out=0.07, pad_in=0.05, lead=None, core_max=0.6):
    """The kept pieces of [a, b] (seconds of src). lead: trim dead air before the first speech down to this many seconds
    (None: the start stays where the caller put it)."""
    x = pcm(src, a, b)
    e = envelope(x)
    n = len(e)
    floor, ref = np.percentile(e, 8), np.percentile(e, 95)
    thr = max(floor + 9.0, min(otsu(e), ref - 12.0))
    loud = e > thr
    for i0, i1 in runs(~loud):                          # a dip < 60 ms inside a word (a stop, a plosive) is still speech
        if i1 - i0 < 6 and i0 > 0 and i1 < n:
            loud[i0:i1] = True
    core = np.zeros(n, bool)
    for w in words:
        s0 = max(0, int((w['s'] - a) / HOP))
        s1 = max(0, min(n, int((min(w['e'], w['s'] + core_max) - a) / HOP)))
        hit = np.flatnonzero(loud[s0:s1])
        if len(hit):                                    # the energy shows where the word really ends
            s1 = min(s1, s0 + hit[-1] + 7)
        core[s0:s1] = True
    loud |= core
    breaths = []
    for i0, i1 in runs(loud):
        if (i1 - i0) * HOP > 0.8 or core[i0:i1].any() or e[i0:i1].max() >= ref - 10:
            continue
        if (i0 > 0 and loud[max(0, i0 - 8):i0].any()) or (i1 < n and loud[i1:i1 + 8].any()):
            continue
        if voiced(x, max(0, i0 - 3), min(n, i1 + 3)) <= 2:
            loud[i0:i1] = False
            breaths.append((round(a + i0 * HOP, 2), round(a + i1 * HOP, 2)))
    cuts = []
    for i0, i1 in runs(~loud):
        t0, t1 = a + i0 * HOP, a + i1 * HOP
        if i1 >= n:                                     # the end is the caller's (snap_end already put it after a line)
            continue
        if i0 == 0:
            if lead is not None and t1 - t0 > lead + 0.2:
                cuts.append((a - 0.01, t1 - lead))
            continue
        kg = next((g for k0, k1, g in keep if t0 < k1 and t1 > k0), None)
        if kg is not None:
            if t1 - t0 <= kg + 0.05:
                continue
            c0, c1 = t0 + kg / 2, t1 - kg / 2
        else:
            if t1 - t0 < min_gap:
                continue
            c0, c1 = t0 + pad_out, t1 - pad_in
        inside = [w for w in words if c0 + 0.08 < w['s'] < c1 - 0.08]     # a quiet word whisper heard in the "silence"
        if inside:
            cur = c0
            for w in inside:
                ws0, ws1 = w['s'] - 0.06, min(w['e'], w['s'] + 0.45) + 0.04
                if ws0 - cur >= 0.15:
                    cuts.append((cur, ws0))
                cur = max(cur, ws1)
            if c1 - cur >= 0.15:
                cuts.append((cur, c1))
        elif c1 - c0 > 0.02:
            cuts.append((c0, c1))
    merged = []
    for c in sorted(cuts):
        if merged and c[0] <= merged[-1][1] + 0.02:
            merged[-1] = (merged[-1][0], max(merged[-1][1], c[1]))
        else:
            merged.append(c)
    out, cur = [], a
    for c0, c1 in merged:
        if c0 > cur + 0.02:
            out.append([round(cur, 3), round(c0, 3)])
        cur = max(cur, c1)
    if b > cur + 0.02:
        out.append([round(cur, 3), round(b, 3)])
    kept = sum(s1 - s0 for s0, s1 in out)

    def dead(spl):
        return sum((~loud[int((s0 - a) / HOP):int((s1 - a) / HOP)]).sum() * HOP for s0, s1 in spl)
    stats = {'src': round(b - a, 2), 'kept': round(kept, 2), 'removed': round(b - a - kept, 2), 'cuts': len(merged), 'breaths': len(breaths),
             'thr_db': round(float(thr), 1), 'floor_db': round(float(floor), 1), 'ref_db': round(float(ref), 1),
             'silence_before': round(dead([[a, b]]), 2), 'silence_after': round(dead(out), 2)}
    return {'spans': out, 'cuts': [[round(c0, 3), round(c1, 3)] for c0, c1 in merged], 'breaths': breaths, 'stats': stats}


def voice_wav(src, sp, out, xfade=0.012, sr=48000):
    """The pieces back to back as one 48 kHz stereo wav, joined with raised-cosine crossfades centred on each cut. Its
    length is exactly the pieces' total, so it stays in sync with a picture cut on the same pieces."""
    pieces = []
    for k, (s0, s1) in enumerate(sp):
        a0 = s0 - (xfade / 2 if k else 0.0)
        a1 = s1 + (xfade / 2 if k < len(sp) - 1 else 0.0)
        p = pcm(src, a0, a1, sr, 2)
        want = int(round((a1 - max(0.0, a0)) * sr))
        p = p[:want] if len(p) >= want else np.concatenate([p, np.zeros((want - len(p), 2), np.float32)])
        pieces.append(p)
    nx = int(xfade * sr)
    w = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nx)))[:, None].astype(np.float32)
    buf = pieces[0]
    for p in pieces[1:]:
        if len(buf) >= nx and len(p) >= nx:
            buf = np.concatenate([buf[:-nx], buf[-nx:] * (1 - w) + p[:nx] * w, p[nx:]])
        else:
            buf = np.concatenate([buf, p])
    with wave.open(out, 'wb') as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((np.clip(buf, -1, 1) * 32767).astype(np.int16).tobytes())
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('a', type=float); ap.add_argument('b', type=float)
    ap.add_argument('--words')
    o = ap.parse_args()
    ws = []
    if o.words:
        W = json.load(open(o.words))
        ws = [w for s in W['segments'] for w in s['words'] if o.a - 0.5 <= w['s'] <= o.b + 0.5]
    r = spans(o.src, o.a, o.b, ws)
    for c in r['cuts']:
        print(f'  cut {c[0]:7.2f}-{c[1]:7.2f}  ({c[1] - c[0]:.2f} s)')
    print(r['stats'])
