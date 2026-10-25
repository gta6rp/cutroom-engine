#!/usr/bin/env python3
"""Stage FIND, part 2: score every second of a VOD and propose clip windows.

Signals (per second): loudness (RMS, z-scored against the stream's own baseline), loudness jumps, speech rate,
hype words / exclamations / money calls in the transcript, and laughter-ish bursts (loud + few words).
Windows snap to transcript segment boundaries: start a beat before the setup, end after the payoff.

usage: moments.py <audio> [top_n]   (needs <audio>.words.json)  ->  <audio>.moments.json
"""
import json, os, re, subprocess, sys
import numpy as np

FFMPEG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.venv', 'bin', 'ffmpeg')
HYPE = re.compile(r"\b(no way|let'?s go+|oh my god|omg|what the|holy|bro+|dude|insane|crazy|wtf|are you kidding|i can'?t believe|"
                  r"oh no|nooo+|yes+|boom|lfg|send it|we'?re rich|million|thousand|grand|\$\d|rugged|rug|dump(ed)?|pump(ed)?|"
                  r"liquidat|stop|shut up|why|fuck|shit|damn|hell)\b", re.I)


def loudness(path, hop=0.5):
    raw = subprocess.run([FFMPEG, '-nostdin', '-loglevel', 'error', '-i', path, '-f', 's16le', '-ac', '1', '-ar', '16000', '-'],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    n = int(16000 * hop)
    frames = x[: len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-5)
    return db, hop


def main():
    audio = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    T = json.load(open(audio + '.words.json'))
    segs = T['segments']
    db, hop = loudness(audio)
    secs = int(len(db) * hop) + 1
    per = lambda arr: np.array([arr[int(i / hop): int((i + 1) / hop)].mean() if int(i / hop) < len(arr) else arr[-1] for i in range(secs)])
    loud = per(db)
    voiced = loud > np.percentile(loud, 30)
    base, spread = np.median(loud[voiced]), np.std(loud[voiced]) + 1e-6
    z = np.clip((loud - base) / spread, -3, 6)
    jump = np.clip(np.r_[0, np.diff(loud)] / spread, 0, 6)
    words_ps = np.zeros(secs)
    hype = np.zeros(secs)
    for s in segs:
        for w in s['words']:
            i = int(w['s'])
            if i < secs:
                words_ps[i] += 1
        for m in HYPE.finditer(s['text']):
            i = int(s['start'] + (s['end'] - s['start']) * m.start() / max(1, len(s['text'])))
            if i < secs:
                hype[i] += 1
        if s['text'].strip().endswith(('!', '?!')):
            hype[min(secs - 1, int(s['end']))] += 0.5
    rate = np.convolve(words_ps, np.ones(5) / 5, 'same')
    rate_z = (rate - rate.mean()) / (rate.std() + 1e-6)
    laugh = ((z > 1.2) & (words_ps < 1)).astype(float)
    sm = lambda a, k: np.convolve(a, np.ones(k) / k, 'same')
    # speech presence: windows with almost no words are music / ambience (concerts, IRL crowds), not moments
    speech = np.convolve(words_ps, np.ones(20), 'same')            # words in a 20 s window
    speech_w = np.clip(speech / 25.0, 0.15, 1.0)
    score = (1.0 * sm(np.clip(z, 0, None), 3) + 0.6 * sm(jump, 3) + 0.4 * sm(laugh, 4)) * speech_w \
        + 0.9 * sm(hype, 6) * 3 + 0.4 * sm(np.clip(rate_z, 0, None), 5)
    # windows: take peaks with spacing, snap to segments
    order = np.argsort(-score)
    taken, out = [], []
    seg_starts = np.array([s['start'] for s in segs])
    seg_ends = np.array([s['end'] for s in segs])
    for p in order:
        if len(out) >= top_n:
            break
        if any(abs(p - q) < 45 for q in taken):
            continue
        a = p - 14
        b = p + 10
        si = max(0, int(np.searchsorted(seg_starts, a, 'right') - 1))
        ei = min(len(segs) - 1, int(np.searchsorted(seg_ends, b)))
        start, end = segs[si]['start'], segs[ei]['end']
        if end - start > 58:
            end = start + 58
        if end - start < 14:
            end = min(seg_ends[-1], start + 18)
        if any(min(end, o['end']) - max(start, o['start']) > 0.3 * (end - start) for o in out):
            continue   # overlaps a window already taken
        taken.append(p)
        text = ' '.join(s['text'] for s in segs if s['end'] > start and s['start'] < end)
        out.append({'peak': int(p), 'start': round(float(start), 2), 'end': round(float(end), 2), 'score': round(float(score[p]), 2),
                    'signals': {'loud_z': round(float(z[p]), 2), 'jump': round(float(jump[p]), 2), 'hype': int(hype[max(0, p - 6): p + 6].sum()),
                                'laugh': int(laugh[max(0, p - 4): p + 4].sum())},
                    'text': text})
    json.dump({'audio': audio, 'duration': T['duration'], 'curve': [round(float(v), 2) for v in score], 'moments': out}, open(audio + '.moments.json', 'w'))
    for i, m in enumerate(out):
        mm = lambda t: f"{int(t // 3600):02d}:{int(t // 60) % 60:02d}:{int(t % 60):02d}"
        print(f"#{i:02d} {mm(m['start'])}-{mm(m['end'])} ({m['end'] - m['start']:.0f}s) score {m['score']} {m['signals']}\n    {m['text'][:420]}\n")


if __name__ == '__main__':
    main()
