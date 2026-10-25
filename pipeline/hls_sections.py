#!/usr/bin/env python3
"""Stage WATCH helper: download only the parts of a VOD a room needs.

Fetches the HLS playlist of one format (via yt-dlp -g), keeps the segments that overlap the requested spans
(+ padding), downloads them with Python (no ffmpeg networking), and joins each span locally into an .mp4 whose
time 0 = the span's first segment start (offset written to <out>.json).

usage: hls_sections.py <vod url> <format id> <out_prefix> <start-end> [<start-end> ...]   (seconds or h:mm:ss)
"""
import json, os, re, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
YTDLP = os.path.join(HERE, '..', '.venv', 'bin', 'yt-dlp')
FF = os.path.join(HERE, '..', '.venv', 'bin', 'ffmpeg')
UA = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) Chrome/130'}


def secs(x):
    p = [float(v) for v in str(x).split(':')]
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def main():
    url, fmt, prefix, *spans = sys.argv[1:]
    m3u8 = subprocess.run([YTDLP, '-f', fmt, '-g', url], capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    pl = fetch(m3u8).decode()
    base = m3u8.rsplit('/', 1)[0] + '/'
    segs, t, dur = [], 0.0, None
    for line in pl.splitlines():
        if line.startswith('#EXTINF:'):
            dur = float(line[8:].split(',')[0])
        elif line and not line.startswith('#'):
            segs.append((t, t + dur, line if line.startswith('http') else base + line)); t += dur
    print(f'{len(segs)} segments, {t / 60:.1f} min')
    out = []
    for i, sp in enumerate(spans):
        a, b = (secs(v) for v in sp.split('-'))
        keep = [s for s in segs if s[1] > a - 4 and s[0] < b + 4]
        files = []
        with ThreadPoolExecutor(8) as ex:
            datas = list(ex.map(lambda s: fetch(s[2]), keep))
        ts_path = f'{prefix}_{i}.ts'
        with open(ts_path, 'wb') as f:
            for d in datas:
                f.write(d)
        mp4 = f'{prefix}_{i}.mp4'
        subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-y', '-i', ts_path, '-c', 'copy', mp4], check=True)
        os.unlink(ts_path)
        info = {'span': [a, b], 'offset': keep[0][0], 'file': os.path.basename(mp4)}
        json.dump(info, open(mp4 + '.json', 'w'))
        out.append(info)
        print(f'span {i}: {a:.1f}-{b:.1f} -> {mp4} (offset {keep[0][0]:.2f}s, {len(keep)} segments)')
    return out


if __name__ == '__main__':
    main()
