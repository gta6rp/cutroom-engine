#!/usr/bin/env python3
"""cutsheet: what an edit is made of, as one picture (for the site): the shots on a timeline with a frame of each, the
song with its beats and the drop, the voice lines with what they say, the transitions and hits. Drawn from the EDL
(edit.json) and the rendered video.

usage: cutsheet.py edit.json rendered.mp4 out.webp [--title "..."]
"""
import argparse, json, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import studio  # noqa: E402

BG, INK, MUTE, LINE, Y, RED, PINK = (13, 13, 16), (244, 241, 234), (150, 146, 140), (40, 40, 46), (255, 214, 10), (255, 59, 48), (255, 140, 200)


def font(sz, bold=True):
    return ImageFont.truetype(os.path.join(studio.FONTS, 'TikTokSans-Bold.ttf' if bold else 'TikTokSans-Medium.ttf'), sz)


def frame(video, t, h):
    import cv2
    c = cv2.VideoCapture(video); c.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000); ok, fr = c.read()
    if not ok:
        return None
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    return im.resize((int(im.width * h / im.height), h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('edl'); ap.add_argument('video'); ap.add_argument('out'); ap.add_argument('--title', default='')
    a = ap.parse_args()
    E = json.load(open(a.edl))
    a.video, a.out = os.path.abspath(a.video), os.path.abspath(a.out)
    os.chdir(os.path.dirname(os.path.abspath(a.edl)))
    D = E['duration']; Wd = 2400; L, R = 40, 40; pps = (Wd - L - R) / D
    X = lambda t: L + t * pps
    H = 700
    im = Image.new('RGB', (Wd, H), BG); d = ImageDraw.Draw(im)
    shots = E['shots']; voice = E.get('voice') or []; trans = E.get('transitions') or []
    beats = []
    m = E.get('music')
    if m:
        try:
            lib = json.load(open(os.path.join(studio.PIPE, '..', 'portfolio', 'music', 'library.json')))
            s = next(x for x in lib if os.path.basename(m['src']) == x['file'])
            bt = json.load(open(os.path.join(studio.PIPE, '..', 'portfolio', 'music', s['beats'])))
            beats = [b - m.get('start', 0) + m.get('at', 0) for b in bt if 0 <= b - m.get('start', 0) + m.get('at', 0) <= D]
            song = s['credit']
        except (StopIteration, OSError, ValueError):
            song = os.path.basename(m['src'])
    on_beat = sum(1 for x in shots[1:] if beats and min(abs(x['at'] - b) for b in beats) < 0.07)
    # header
    d.text((L, 22), a.title or 'The cut', font=font(34), fill=INK)
    facts = f"{D:.0f} s  ·  {len(shots)} shots, {on_beat} cut on the beat  ·  {len(voice)} lines of their own  ·  {len(trans)} transitions" + (f"  ·  {song}" if m else '')
    d.text((L, 66), facts, font=font(22, False), fill=MUTE)
    lx = Wd - R - 760                                    # legend
    d.rectangle((lx, 30, lx + 26, 40), fill=(90, 220, 255)); d.text((lx + 34, 24), 'slow motion', font=font(16, False), fill=MUTE)
    d.rectangle((lx + 150, 30, lx + 176, 36), fill=(200, 200, 200)); d.text((lx + 184, 24), 'black & white', font=font(16, False), fill=MUTE)
    d.text((lx + 320, 24), 'transitions: Z zoom · W whip · F flash · D dissolve · B black · G glitch', font=font(16, False), fill=MUTE)
    # 1. shots
    y0, sh_h = 120, 220
    d.text((L, y0 - 4), 'SHOTS', font=font(16), fill=Y)
    y0 += 22
    for x in shots:
        x0, x1 = int(X(x['at'])), int(X(x['at'] + x['dur']))
        if x1 - x0 < 2:
            continue
        if not x.get('src'):
            d.rectangle((x0, y0, x1 - 1, y0 + sh_h), fill=(0, 0, 0)); continue
        w = x1 - x0
        fw = int(sh_h * 9 / 16)
        k = max(1, -(-w // fw))                          # a long shot gets several frames across it
        for j in range(k):
            fr = frame(a.video, x['at'] + x['dur'] * (j + 0.5) / k, sh_h)
            if fr is None:
                continue
            seg = min(fw, w - j * fw)
            cx = fr.width // 2
            im.paste(fr.crop((max(0, cx - seg // 2), 0, max(0, cx - seg // 2) + seg, sh_h)), (x0 + j * fw, y0))
        slow = isinstance(x.get('speed'), list) or (isinstance(x.get('speed'), (int, float)) and x['speed'] < 0.9)
        if slow:
            d.rectangle((x0, y0 + sh_h - 16, x1 - 1, y0 + sh_h), fill=(90, 220, 255))
        if 'bw' in (x.get('look') or []):
            d.rectangle((x0, y0, x1 - 1, y0 + 5), fill=(200, 200, 200))
        d.line((x0, y0, x0, y0 + sh_h), fill=BG, width=2)
    # transitions
    for tr in trans:
        x0 = X(tr['at']); lab = {'zoom': 'Z', 'whip': 'W', 'flash': 'F', 'dissolve': 'D', 'black': 'B', 'glitch': 'G'}.get(tr['type'], '·')
        d.ellipse((x0 - 9, y0 + sh_h + 6, x0 + 9, y0 + sh_h + 24), fill=Y); d.text((x0 - 5, y0 + sh_h + 7), lab, font=font(13), fill=BG)
    # 2. music
    y1 = y0 + sh_h + 44
    d.text((L, y1 - 4), 'MUSIC' + ('  (muffled until the drop)' if m and m.get('muffle_until') else ''), font=font(16), fill=Y)
    y1 += 22
    if m:
        pcm = subprocess.run([studio.FF, '-nostdin', '-loglevel', 'error', '-ss', f"{m.get('start', 0):.3f}", '-t', f"{D:.3f}", '-i', m['src'], '-f', 's16le', '-ac', '1', '-ar', '8000', '-'],
                             capture_output=True).stdout
        w = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        n = int((Wd - L - R)); off = m.get('at', 0)
        peak = max(1e-4, float(np.sqrt((w ** 2).mean())) * 2.2)
        for i in range(n):
            t = i / pps - off
            if t < 0:
                continue
            seg = w[int(t * 8000): int((t + 1 / pps) * 8000)]
            if not len(seg):
                continue
            amp = min(34.0, float(np.sqrt((seg ** 2).mean())) / peak * 34)
            mu = m.get('muffle_until') and (i / pps) < m['muffle_until']
            d.line((L + i, y1 + 40 - amp, L + i, y1 + 40 + amp), fill=(110, 100, 60) if mu else (200, 170, 40))
        for b in beats:
            d.line((X(b), y1 + 84, X(b), y1 + 92), fill=MUTE)
        if m.get('muffle_until'):
            xd = X(m['muffle_until']); d.line((xd, y1 - 6, xd, y1 + 92), fill=RED, width=3); d.text((xd + 6, y1 - 6), 'DROP', font=font(16), fill=RED)
    # 3. voice lines
    y2 = y1 + 116
    d.text((L, y2 - 4), 'THEIR OWN WORDS', font=font(16), fill=Y)
    y2 += 22
    lyr = [t for t in E.get('texts') or [] if t.get('style') in ('lyric', 'hype') and t.get('anim') != 'fade']
    for k, v in enumerate(voice):
        x0 = X(v['at']); x1 = X(v['at'] + (v['out'] - v['in']) / (v.get('rate') or 1))
        row = k % 2
        col = PINK if v.get('rate') else INK
        d.rounded_rectangle((x0, y2 + row * 34, max(x0 + 4, x1), y2 + row * 34 + 26), radius=6, fill=(48, 48, 56))
        said = next((t['text'] for t in lyr if abs(t['t0'] - v['at']) < 0.8), '')
        if said and x1 - x0 > 40:
            txt = said
            while txt and d.textlength(txt, font=font(15, False)) > x1 - x0 - 10:
                txt = txt[:-2]
            d.text((x0 + 6, y2 + row * 34 + 4), txt, font=font(15, False), fill=col)
    # time ruler
    yr = H - 26
    for s in range(0, int(D) + 1, 5):
        d.line((X(s), yr - 4, X(s), yr), fill=MUTE); d.text((X(s) + 3, yr - 2), f'{s}s', font=font(13, False), fill=MUTE)
    im.save(a.out, quality=82)
    print(a.out, im.size)


if __name__ == '__main__':
    main()
