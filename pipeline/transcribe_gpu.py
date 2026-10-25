#!/usr/bin/env python3
"""GPU transcription (transformers Whisper, default large-v3-turbo) with word timings, in the same format as
transcribe.py. Processes the audio in 10-minute blocks and checkpoints after each one.

usage: transcribe_gpu.py <audio> [model] [start-end|all] [out]
"""
import json, os, re, subprocess, sys, time
import numpy as np
import torch
from transformers import pipeline

FFMPEG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.venv', 'bin', 'ffmpeg')
SR = 16000
BLOCK = 600.0


def load_audio(path, a=None, b=None):
    cmd = [FFMPEG, '-nostdin', '-loglevel', 'error']
    if a is not None:
        cmd += ['-ss', str(a), '-to', str(b)]
    cmd += ['-i', path, '-f', 's16le', '-ac', '1', '-ar', str(SR), '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def to_segments(words):
    """Group word chunks into sentence-ish segments (break on punctuation or a pause > 0.7 s)."""
    segs, cur = [], []
    for w in words:
        if cur and (w['s'] - cur[-1]['e'] > 0.7 or len(cur) >= 30):
            segs.append(cur); cur = []
        cur.append(w)
        if re.search(r'[.!?]$', w['w']):
            segs.append(cur); cur = []
    if cur:
        segs.append(cur)
    return [{'start': s[0]['s'], 'end': s[-1]['e'], 'text': ' '.join(x['w'] for x in s), 'words': s} for s in segs]


def main():
    audio = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else 'openai/whisper-large-v3-turbo'
    span = sys.argv[3] if len(sys.argv) > 3 else 'all'
    out_path = sys.argv[4] if len(sys.argv) > 4 else audio + '.words.json'
    t0 = time.time()
    asr = pipeline('automatic-speech-recognition', model=name, torch_dtype=torch.float16, device='cuda:0')
    x = load_audio(audio)
    total = len(x) / SR
    a, b = (0.0, total) if span in ('all', '-') else (float(v) for v in span.split('-'))
    words = []
    t = a
    while t < b:
        e = min(b, t + BLOCK)
        chunk = x[int(t * SR): int(e * SR)]
        r = asr({'raw': chunk, 'sampling_rate': SR}, return_timestamps='word', chunk_length_s=30, batch_size=16,
                generate_kwargs={'language': 'en', 'task': 'transcribe'})
        for c in r.get('chunks', []):
            s_, e_ = c['timestamp']
            if s_ is None:
                continue
            e_ = e_ if e_ is not None else s_ + 0.3
            words.append({'w': c['text'].strip(), 's': round(t + s_, 2), 'e': round(t + e_, 2), 'p': 1.0})
        segs = to_segments([w for w in words if w['w']])
        json.dump({'audio': audio, 'model': name, 'span': [a, b], 'duration': total, 'complete': e >= b, 'segments': segs}, open(out_path, 'w'))
        print(f'{e / 60:.1f} / {b / 60:.1f} min, {time.time() - t0:.0f}s elapsed', flush=True)
        t = e
    print(f'done: {len(segs)} segments in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
