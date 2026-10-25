#!/usr/bin/env python3
"""Re-hear spans of a recording with the shared GPU ASR (whisper-large-v3-turbo, asr/run.sh) and write a words file
edit.py can use: {"segments": [{"start", "end", "text", "words": [{w, s, e, p}]}]} with times on the recording's clock.

usage: rehear.py <audio> <out.json> <start-end> [<start-end> ...]
"""
import json, os, subprocess, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FF = os.path.join(HERE, '..', '.venv', 'bin', 'ffmpeg')
ASR = os.environ.get('CUTROOM_ASR', 'http://127.0.0.1:8835')


def hear(audio, a, b):
    pcm = subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', audio,
                          '-f', 's16le', '-ac', '1', '-ar', '16000', '-'], capture_output=True, check=True).stdout
    req = urllib.request.Request(ASR + '/transcribe', data=pcm, method='POST', headers={'Content-Type': 'application/octet-stream'})
    words = json.loads(urllib.request.urlopen(req, timeout=120).read())['words']
    return [{'w': w['w'], 's': round(a + w['s'], 2), 'e': round(a + w['e'], 2), 'p': 1.0} for w in words if w['w'].strip()]


def hear_safe(audio, a, b, depth=0):
    """Long or tricky spans can trip the server's word-timestamp alignment: split in half and retry. If the server is
    restarting (connection refused), wait for it, up to about 2 minutes."""
    import time
    for wait in (5, 10, 20, 40, 60):
        try:
            return hear(audio, a, b)
        except urllib.error.HTTPError:
            break
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            print(f'  ASR unreachable ({str(e)[:60]}); retrying in {wait} s', file=sys.stderr)
            time.sleep(wait)
    try:
        return hear(audio, a, b)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors='ignore')[:200]
        if depth >= 2 or b - a < 8:
            print(f'  span {a:.0f}-{b:.0f} failed: {msg}', file=sys.stderr)
            return []
        m = (a + b) / 2
        return hear_safe(audio, a, m, depth + 1) + hear_safe(audio, m, b, depth + 1)


def main(audio, out, spans):
    segs = []
    for sp in spans:
        a, b = map(float, sp.split('-'))
        ws = hear_safe(audio, a, b)
        if ws:
            segs.append({'start': ws[0]['s'], 'end': ws[-1]['e'], 'text': ' '.join(w['w'] for w in ws), 'words': ws})
    json.dump({'audio': audio, 'model': 'whisper-large-v3-turbo', 'segments': segs}, open(out, 'w'))
    print(f'{out}: {sum(len(s["words"]) for s in segs)} words in {len(segs)} spans')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
