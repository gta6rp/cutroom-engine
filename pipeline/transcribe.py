#!/usr/bin/env python3
"""Stage FIND, part 1: word-level transcript of a VOD's audio.

usage: transcribe.py <audio> [model] [start-end] [out]
  model      faster-whisper model (default small.en; use base.en for a fast first pass over long VODs)
  start-end  only transcribe this span (seconds), e.g. 3600-3720; times in the output stay VOD-absolute
  out        output path (default <audio>.words.json)
The transcript is checkpointed every 25 segments, so a killed run keeps what it had.
"""
import json, os, subprocess, sys, time
import numpy as np
from faster_whisper import WhisperModel

FFMPEG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.venv', 'bin', 'ffmpeg')
SR = 16000


def load_audio(path, a=None, b=None):
    cmd = [FFMPEG, '-nostdin', '-loglevel', 'error']
    if a is not None:
        cmd += ['-ss', str(a), '-to', str(b)]
    cmd += ['-i', path, '-f', 's16le', '-ac', '1', '-ar', str(SR), '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def main():
    audio = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else 'small.en'
    a = b = None
    if len(sys.argv) > 3 and sys.argv[3] not in ('-', 'all'):
        a, b = (float(x) for x in sys.argv[3].split('-'))
    out_path = sys.argv[4] if len(sys.argv) > 4 else audio + '.words.json'
    t0 = time.time()
    model = None
    for device, ctype in (('cuda', 'float16'), ('cpu', 'int8')):
        try:
            model = WhisperModel(name, device=device, compute_type=ctype, cpu_threads=16)
            print('device', device, flush=True)
            break
        except Exception as e:  # no CUDA build of ctranslate2 on this machine -> CPU
            print('fallback from', device, '-', str(e)[:100], flush=True)
    off = a or 0.0
    segs, info = model.transcribe(load_audio(audio, a, b), word_timestamps=True, vad_filter=True, beam_size=1, condition_on_previous_text=False)
    out = []
    dump = lambda done: json.dump({'audio': audio, 'model': name, 'span': [a, b] if a is not None else None, 'duration': info.duration + off,
                                   'complete': done, 'segments': out}, open(out_path, 'w'))
    for s in segs:
        out.append({'start': round(s.start + off, 2), 'end': round(s.end + off, 2), 'text': s.text.strip(),
                    'words': [{'w': w.word.strip(), 's': round(w.start + off, 2), 'e': round(w.end + off, 2), 'p': round(w.probability, 2)} for w in (s.words or [])]})
        if len(out) % 25 == 0:
            dump(False)
            print(f'{(s.end + off) / 60:.1f} min transcribed, {time.time() - t0:.0f}s elapsed', flush=True)
    dump(True)
    print(f'done: {len(out)} segments in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
