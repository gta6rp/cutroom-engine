#!/usr/bin/env python3
"""Stages CUT + CAPTION: turn chosen VOD spans into a finished 9:16 clip.

  spec.json -> portfolio/<program>/<id>.mp4 (+ .json post sheet)

Layout "hooksplit": the streamer's facecam on top (found by face detection, so it follows the cam between scenes),
the stream screen below, a hook box on the seam, word-by-word captions timed from the transcript, the source
credited as plain text. Segments can be joined with cards ("2 MINUTES LATER"). Audio is the stream's own,
loudness-normalised to -14 LUFS. No AI anywhere in the output (some programs forbid it).
A talk cut on a full frame (speaker, letterbox, a centred cam card; >= 1.5 words/s) gets jump cuts (tighten_cut, from the
fx session's talkpolish lab): pauses and breaths go, every word stays, the voice is one crossfaded track, cuts alternate
1.0x / 1.12x, and its captions are punch captions (1-3 words, one huge word on 1-3 beats). spec "tighten": false turns it off.

usage: edit.py <spec.json>
"""
import json, os, re, shutil, subprocess, sys, tempfile, wave
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FF = os.path.join(HERE, '..', '.venv', 'bin', 'ffmpeg')
FONTS = os.path.join(HERE, 'fonts')
W, H = 1080, 1920
RANK_COLORS = ['&H0AD6FF&', '&HD3D3D3&', '&H3E8BCD&', '&HFF7BB7&', '&H5AFF39&']   # gold, silver, bronze, violet, green (ASS BGR)


THREADS = os.environ.get('CUTROOM_FF_THREADS')   # the live room caps encoder threads so transcription keeps up


def ff(*args):
    args = [str(a) for a in args]
    if THREADS:
        args[-1:-1] = ['-threads', THREADS]           # output option: goes right before the output path
    r = subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-y', *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])


def grab(video, t):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        ff('-ss', f'{t:.2f}', '-i', video, '-frames:v', 1, f.name)
        img = cv2.imread(f.name)
    os.unlink(f.name)
    return img


FACE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))


def find_cam(video, a, b, cam_w=678, cam_h=382):
    """Facecam rectangle for a span: detect faces on a few frames, snap a cam-sized box to the nearest bottom corner."""
    votes = []
    for t in np.linspace(a + 0.5, b - 0.5, 5):
        img = grab(video, t)
        if img is None:
            continue
        sh, sw = img.shape[:2]
        g = cv2.cvtColor(cv2.resize(img, (sw // 2, sh // 2)), cv2.COLOR_BGR2GRAY)
        faces = FACE.detectMultiScale(g, 1.1, 6, minSize=(40, 40))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            votes.append(((x + fw / 2) * 2, (y + fh / 2) * 2, sw, sh))
    if not votes:
        return None
    cx = float(np.median([v[0] for v in votes])); sw, sh = votes[0][2], votes[0][3]
    x0 = 0 if cx < sw / 2 else sw - cam_w
    return {'x': int(x0), 'y': int(sh - cam_h), 'w': cam_w, 'h': cam_h, 'side': 'left' if x0 == 0 else 'right', 'face_votes': len(votes)}


def side_at(video, t, cam_w=678, cam_h=382):
    """Which bottom corner holds the facecam: the cam shows a lit room, the trading UI around it is near-black.
    Returns 'left' / 'right', or None mid-transition (neither corner is a camera image)."""
    img = grab(video, t)
    if img is None:
        return None
    sh, sw = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    def camlike(x0):
        r = g[sh - cam_h + 8: sh - 8, x0 + 8: x0 + cam_w - 8]
        if not (r.mean() > 70 and r.std() > 25):   # bright and textured
            return False
        # settled: the strips just outside the box (above it, and on its inner side) are dark UI
        above = g[sh - cam_h - 34: sh - cam_h - 6, x0 + 20: x0 + cam_w - 20]
        side = g[sh - cam_h + 20: sh - 20, (cam_w + 6 if x0 == 0 else x0 - 40): (cam_w + 40 if x0 == 0 else x0 - 6)]
        return above.mean() < 60 and side.mean() < 60
    l, rr = camlike(0), camlike(sw - cam_w)
    if l and not rr:
        return 'left'
    if rr and not l:
        return 'right'
    return None


def refine_switch(video, lo, hi, side_lo, iters=4):
    """Bisect the moment the cam leaves side_lo (to ~60 ms); unknown frames count as 'switched' (mid-transition)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if side_at(video, mid) == side_lo:
            lo = mid
        else:
            hi = mid
    return hi


def cam_runs(video, a, b, step=0.5, cam_w=678, cam_h=382):
    """Split [a, b] by facecam corner. Streamers switch scene layouts mid-moment, often with an animated transition
    where the cam is in neither corner: those spans get the 'full' layout (the whole stream frame on top)."""
    ts = list(np.arange(a + 0.1, b, step))
    sides = [side_at(video, t) for t in ts]
    if not any(sides):
        return []
    # detector flicker: a short gap between two samples of the same side keeps that side
    for i in range(1, len(sides) - 1):
        if sides[i] is None and sides[i - 1] and sides[i - 1] == sides[i + 1]:
            sides[i] = sides[i - 1]
    if sides[0] is None:
        sides[0] = next((v for v in sides if v), None)
    if sides[-1] is None:
        sides[-1] = next((v for v in reversed(sides) if v), None)
    lab = [v or 'full' for v in sides]
    runs, start = [], a
    for i in range(1, len(lab)):
        if lab[i] != lab[i - 1]:
            lo, hi = ts[i - 1], ts[i]
            for _ in range(4):   # bisect to ~30 ms
                mid = (lo + hi) / 2
                if (side_at(video, mid) or 'full') == lab[i - 1]:
                    lo = mid
                else:
                    hi = mid
            runs.append((start, hi, lab[i - 1])); start = hi
    runs.append((start, b, lab[-1]))
    out = []
    for r0, r1, side in runs:
        if side == 'full':
            out.append((r0, r1, {'x': 0, 'y': 0, 'w': 1920, 'h': 1080, 'side': 'full'}))
        else:
            out.append((r0, r1, {'x': 0 if side == 'left' else 1920 - cam_w, 'y': 1080 - cam_h, 'w': cam_w, 'h': cam_h, 'side': side}))
    return out


def speaker_shots(video, a, b, fps=4, crop_w=608):
    """Split [a, b] into camera shots (frame-difference cuts) and centre a 9:16 crop on the main face of each shot."""
    with tempfile.TemporaryDirectory() as d:
        ff('-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', video, '-vf', f'fps={fps},scale=480:-2', os.path.join(d, 'f%05d.png'))
        files = sorted(os.listdir(d))
        frames = [cv2.imread(os.path.join(d, f)) for f in files]
    if not frames:
        return [(a, b, 960)]
    sh, sw = frames[0].shape[:2]
    scale = 1920 / sw
    hists, faces = [], []
    for im in frames:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hists.append(cv2.normalize(cv2.calcHist([g], [0], None, [32], [0, 256]), None).flatten())
        fs = FACE.detectMultiScale(g, 1.1, 5, minSize=(28, 28))
        faces.append(None if not len(fs) else (lambda f: (f[0] + f[2] / 2) * scale)(max(fs, key=lambda f: f[2] * f[3])))
    cuts = [0] + [i for i in range(1, len(frames)) if cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA) > 0.32] + [len(frames)]
    # merge shots shorter than 0.5 s into the previous one (flash frames, detector noise)
    spans = []
    for s0, s1 in zip(cuts, cuts[1:]):
        if spans and (s1 - s0) / fps < 0.5:
            spans[-1] = (spans[-1][0], s1)
        else:
            spans.append((s0, s1))
    shots = []
    for s0, s1 in spans:
        xs = [x for x in faces[s0:s1] if x is not None]
        cx = float(np.median(xs)) if xs else 960.0
        x0 = int(min(1920 - crop_w, max(0, cx - crop_w / 2)))
        shots.append((a + s0 / fps, a + s1 / fps if s1 < len(frames) else b, x0, follow(faces[s0:s1], fps, crop_w)))
    return shots


def follow(xs, fps, crop_w, still=50):
    """Handheld IRL cameras pan without cutting, so a fixed crop loses people. Track the main face through a shot:
    hold through misses, drop one-frame jumps to another face, smooth both ways, cap the speed.
    Returns [(t, x0)] relative to the shot start, or None when the face barely moves (a fixed crop is calmer)."""
    known = [x for x in xs if x is not None]
    if len(known) < 3:
        return None
    v, last = [], known[0]
    for x in xs:
        last = x if x is not None else last
        v.append(last)
    v = np.array(v, float)
    med = np.array([np.median(v[max(0, i - 2): i + 3]) for i in range(len(v))])       # kills single-frame jumps
    for _ in range(2):                                                                  # zero-phase EMA
        for i in range(1, len(med)):
            med[i] = 0.6 * med[i - 1] + 0.4 * med[i]
        med = med[::-1]
    cap = 900 / fps                                                                     # at most 900 px/s on the 1920 frame
    for i in range(1, len(med)):
        med[i] = med[i - 1] + float(np.clip(med[i] - med[i - 1], -cap, cap))
    x0 = np.clip(med - crop_w / 2, 0, 1920 - crop_w).round().astype(int)
    if x0.max() - x0.min() < still:
        return None
    return [(i / fps, int(x)) for i, x in enumerate(x0)]


def crop_cmds(track, t0, t1, name, dx=0):
    """sendcmd lines that move crop@name along a face track, for the piece [t0, t1] (shot-relative seconds)."""
    pts = [(t - t0, x + dx) for t, x in track if t0 - 0.25 <= t <= t1]
    return ';'.join(f'{max(0.0, t):.3f} crop@{name} x {x}' for t, x in pts)


TOP_H, CENTER_H, CENTER_Y = 608, 980, 500
GAME_H = 1010
PRESET = os.environ.get('CUTROOM_PRESET', 'slow')     # live rooms use 'veryfast': the cut is ready sooner, phones can't tell
SLUR_AUDIO = re.compile(r'n+i+g+g|\bf+a+g+(?:s|got|gots|gy)?\b|r+e+t+a+r+d', re.I)
CENSOR = [(r'f+u+c+k', 'F**K'), (r'sh+i+t', 'SH*T'), (r'b+i+t+c+h', 'B*TCH'), (r'd+i+c+k', 'D*CK'), (r'p+u+s+s+y', 'P***Y'), (r'c+u+n+t', 'C**T'),
          (r'n+i+g+g', 'N***'), (r'f+a+g+', 'F**'), (r'r+e+t+a+r+d', 'R****D')]


def exposure_lift(video, n=10):
    """A dark cut (a night room, a hallway with the lights off) lifted so faces read on a phone: '' when it's fine, else
    an ffmpeg filter that raises the shadows and mids (not the highlights, which would wash out) by how dark it is."""
    import cv2, numpy as np
    c = cv2.VideoCapture(video); N = c.get(7); ys = []
    for k in range(n):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + 0.5) / n)); ok, f = c.read()
        if ok:
            ys.append(float(cv2.cvtColor(cv2.resize(f, (192, 342)), cv2.COLOR_BGR2GRAY).mean()))
    if not ys:
        return ''
    m = float(np.median(ys))
    if m >= 72:
        return ''
    g = min(1.6, 1.0 + (72 - m) / 70)                    # median luma 40 -> gamma 1.46
    return f"eq=gamma={g:.2f}:contrast=1.04:saturation=1.06,"


def headline_png(text, out, W, size=84):
    """The news-headline hook (Pubity "ISHOWSPEED JUST RACED A HORSE" 3.3M, House of Highlights 3.8M): all caps, heavy,
    white with a thick black outline, the name, CAPS punch words and numbers in yellow, colour emoji. No box."""
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, os.path.join(HERE, 'studio'))
    import studio as ST
    font = ImageFont.truetype(os.path.join(FONTS, 'TikTokSansWide-Black.ttf'), size)
    words = text.split(' ')
    first = next((i for i, w in enumerate(words) if re.sub(r'[^A-Za-z]', '', w)), 0)
    def key(i, w):
        t = re.sub(r"[^A-Za-z0-9']", '', w)
        return bool(t) and (i == first or any(ch.isdigit() for ch in t) or (len(t) >= 3 and t.isupper()))
    toks = [(w.upper() if not any(e for e, _ in ST.runs(w)) else w, key(i, w)) for i, w in enumerate(words)]
    d0 = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    space = d0.textlength(' ', font=font)
    tw_ = lambda w: sum(d0.textlength(t, font=font) if not e else ST.emoji_img(t, size).width + 6 for e, t in ST.runs(w))
    lines, cur, cw = [], [], 0.0
    for tk in toks:
        w_ = tw_(tk[0])
        if cur and cw + space + w_ > W * 0.86:
            lines.append(cur); cur, cw = [], 0.0
        cw += (space if cur else 0) + w_; cur.append(tk)
    lines.append(cur)
    asc, desc = font.getmetrics(); lh = int((asc + desc) * 0.92)
    widths = [sum(tw_(t) for t, _ in l) + space * (len(l) - 1) for l in lines]
    tw = int(max(widths)) + 40; th = lh * len(lines) + 30
    im = Image.new('RGBA', (tw, th), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    y = 12
    for l, lw in zip(lines, widths):
        x = (tw - lw) / 2
        for t, k in l:
            for e, part in ST.runs(t):
                if e:
                    em = ST.emoji_img(part, size); im.alpha_composite(em, (int(x + 3), int(y + (lh - em.height) / 2))); x += em.width + 6
                else:
                    dr.text((x, y), part, font=font, fill=(255, 214, 10, 255) if k else (255, 255, 255, 255), stroke_width=7, stroke_fill=(0, 0, 0, 255))
                    x += d0.textlength(part, font=font)
            x += space
        y += lh
    im.save(out)
    return tw, th


def plain_text_png(text, out, W, size=54):
    """TikTok's plain caption: white TikTok Sans with a thin dark outline and colour emoji, wrapped, centred."""
    from PIL import Image, ImageDraw, ImageFont
    sys.path.insert(0, os.path.join(HERE, 'studio'))
    import studio as ST
    font = ImageFont.truetype(os.path.join(FONTS, 'TikTokSans-SemiBold.ttf'), size)
    d0 = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
    width = lambda line: sum(d0.textlength(t, font=font) if not e else ST.emoji_img(t, size).width + 6 for e, t in ST.runs(line))
    lines, cur = [], ''
    for wd in text.split(' '):
        trial = (cur + ' ' + wd).strip()
        if cur and width(trial) > W * 0.84:
            lines.append(cur); cur = wd
        else:
            cur = trial
    lines.append(cur)
    asc, desc = font.getmetrics(); lh = asc + desc
    tw = int(max(width(l) for l in lines)) + 24; th = lh * len(lines) + 8 * (len(lines) - 1) + 16
    im = Image.new('RGBA', (tw, th), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    y = 8
    for l in lines:
        x = (tw - width(l)) / 2
        for e, t in ST.runs(l):
            if e:
                em = ST.emoji_img(t, size); im.alpha_composite(em, (int(x + 3), int(y + (lh - em.height) / 2))); x += em.width + 6
            else:
                dr.text((x, y), t, font=font, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 200)); x += d0.textlength(t, font=font)
        y += lh + 8
    im.save(out)
    return tw, th


def snap_end(words, earliest, target, latest):
    """A cut's end moved onto a finished line: the word end (a . ? ! or a pause of 0.45 s+ after it) nearest the target
    within [earliest, latest], plus a breath. The payoff + 4 s rail alone cut xQc mid-sentence ("... YOU SAY") twice in
    one session. Without a candidate, at least the word the target falls in is finished."""
    ws = sorted([w for w in words if 'e' in w], key=lambda w: w['s'])
    cands = []
    for i, w in enumerate(ws):
        if not (earliest <= w['e'] <= latest):
            continue
        gap = (ws[i + 1]['s'] - w['e']) if i + 1 < len(ws) else 9.0
        if str(w['w']).rstrip()[-1:] in '.?!' or gap >= 0.45:
            cands.append(w['e'])
    if cands:
        best = min(cands, key=lambda e: abs(e - target) + (0.5 if e > target + 1.5 else 0))
        return round(best + 0.3, 2)
    inside = next((w for w in ws if w['s'] < target < w['e']), None)
    return round(inside['e'] + 0.25, 2) if inside else target


def measured_gain(path, target=-14.0):
    """dB that brings the file to the target integrated loudness (None when it can't be read)."""
    r = subprocess.run([FF, '-nostdin', '-hide_banner', '-i', path, '-vn', '-af', 'ebur128', '-f', 'null', '-'], capture_output=True, text=True)
    v = re.findall(r'I:\s+(-?[\d.]+) LUFS', r.stderr)
    return max(-20.0, min(30.0, target - float(v[-1]))) if v and float(v[-1]) > -70 else None


def audio_chain(bleeps, total, gain=None):
    """[0:a] -> [a]: loudness to -14 LUFS, with a 1 kHz bleep over each slur (the word itself muted). gain (dB, measured on
    the joined cut): one static gain and a peak limiter, so the payoff's scream stays louder than the setup (single-pass
    loudnorm rode the level like a slow AGC and flattened it); without it, loudnorm as before."""
    lvl = f'volume={gain:.2f}dB,alimiter=limit=0.84:attack=3:release=60:level=0' if gain is not None else 'loudnorm=I=-14:TP=-1.5:LRA=11'
    if not bleeps:
        return f'[0:a]{lvl}[a]'
    on = '+'.join(f'between(t,{a:.3f},{b:.3f})' for a, b in bleeps)
    return (f"[0:a]volume=0:enable='{on}'[mu];sine=f=1000:sample_rate=48000:duration={total + 1:.2f},volume=0.2,volume=0:enable='not({on})',"
            f"aformat=channel_layouts=stereo[bp];[mu][bp]amix=inputs=2:normalize=0:duration=first,{lvl}[a]")


def censor(w):
    """Mask profanity and slurs inside any word form (fucked, bullshit, motherfucker), keeping the rest of the word."""
    out = w
    for pat, rep in CENSOR:
        out = re.sub(pat, rep, out, flags=re.I)
    out = re.sub(r'f\W*\*+\W*', 'F**', out, flags=re.I) if out == w else out   # already-starred input from ASR ("f***ing")
    if re.sub(r'[^a-z]', '', out.lower()) == 'ass':
        out = 'A**'
    return out


def ass_time(t):
    t = max(0, t)
    return f"{int(t // 3600)}:{int(t // 60) % 60:02d}:{t % 60:05.2f}"


def ass_header():
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Anton,118,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,1,0,1,9,4,5,40,40,0,1
Style: Hook,Archivo Black,50,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,3,22,0,5,90,90,0,1
Style: Credit,Archivo Black,30,&H33FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,1,0,1,3,0,2,40,40,70,1
Style: Title,Archivo Black,62,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,70,70,0,1
Style: Card,Anton,150,&H000AD6FF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,2,0,1,10,0,5,40,40,0,1
Style: Pop,Anton,150,&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,0,0,0,0,100,100,1,0,1,12,5,5,30,30,0,1
Style: Plain,TikTok Sans SemiBold,54,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,1.5,5,80,80,0,1
Style: Subs,TikTok Sans Bold,58,&H00FFFFFF,&H00FFFFFF,&H58000000,&H00000000,0,0,0,0,100,100,0,0,3,14,0,5,70,70,0,1
Style: Mark,Archivo Black,34,&H0018FC53,&H00FFFFFF,&H00000000,&H78000000,0,0,0,0,100,100,1,0,1,4,2,2,40,40,0,1
Style: Rank,Archivo Black,46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,1,5,3,7,56,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


STOP_EMPH = set("""the a an and or but so to of in on at for with from by is are was were be been it its this that these those he she
they them his her their him we us our you your i me my mine just like then than when what's that's there here have has
had do does did not don't can't won't im i'm it's he's she's they're you're gonna wanna got get gets going really very
about into out up down over after before again all some any more most much one two who how why where which while because
yeah yes okay ok oh uh um""".split())
SHOUT = {'WAIT', 'NO', 'NAH', 'BRO', 'WHAT', 'STOP', 'HELP', 'RUN', 'GO', 'YO', 'WHY', 'NEVER', 'CRAZY', 'INSANE', 'DEAD', 'BRUH'}
EMPH_COLORS = ['&H0AD6FF&', '&HF53DFF&']               # yellow, then magenta: only on the words that carry the moment


def laughing(audio, t0, t1):
    """Is the stretch [t0, t1] of the audio laughter? Loud, and its loudness pulses 3-7 times a second (ha-ha-ha), unlike a
    sustained scream or music. Only says yes when it's clear."""
    import numpy as np
    r = subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-ss', f'{t0:.3f}', '-t', f'{t1 - t0:.3f}', '-i', audio, '-vn', '-ac', '1', '-ar', '8000',
                        '-f', 's16le', '-'], capture_output=True)
    x = np.frombuffer(r.stdout, np.int16).astype(np.float32)
    if len(x) < 8000 * 0.7:
        return False
    env = np.sqrt(np.array([np.mean(x[k:k + 160] ** 2) for k in range(0, len(x) - 160, 160)]) + 1)   # 50 Hz envelope
    if 20 * np.log10(env.mean()) < 62:                 # quiet: a pause, not a laugh
        return False
    e = env - env.mean()
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e))))
    f = np.fft.rfftfreq(len(e), 1 / 50)
    band = spec[(f >= 3) & (f <= 7)].sum(); tot = spec[(f >= 0.5) & (f <= 20)].sum() + 1e-6
    return band / tot > 0.38 and env.std() / env.mean() > 0.35


def emphasis(hook, words):
    """Which spoken words get colour (top clip pages colour by meaning, never at random): words of the hook, names (a
    capital letter mid-sentence), numbers, and shouted interjections. Returns a predicate on a word dict."""
    hook_words = {re.sub(r"[^a-z0-9']", '', w.lower())[:5] for w in (hook or '').split()}
    hook_words = {w for w in hook_words if len(w) >= 4 and w not in STOP_EMPH}
    def f(w, prev=None):
        t = re.sub(r"[^A-Za-z0-9']", '', w['w'])
        if not t:
            return False
        low = t.lower()
        if any(ch.isdigit() for ch in t) or t.upper() in SHOUT:
            return True
        if low not in STOP_EMPH and len(low) >= 4 and low[:5] in hook_words:
            return True
        return t[0].isupper() and prev is not None and not prev['w'].endswith(('.', '?', '!')) and low not in STOP_EMPH and t != 'I'
    return f


def chunks(words, max_words=2, max_gap=0.45):
    """Group words into on-screen chunks of 1–2 words (word-pop pacing)."""
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w['s'] - cur[-1]['e'] > max_gap or cur[-1]['w'].endswith(('.', '?', '!', ','))):
            out.append(cur); cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


FUNC = {'the', 'my', 'a', 'an', 'to', 'of', 'and', 'your', 'his', 'her', 'their', 'our', 'in', 'on', 'at', 'for', 'with', 'is', 'was', 'i', "i'm", 'like'}
TIGHT_LAYOUTS = ('speaker', 'letterbox')             # talk on a full frame; a split's screen panel or a game must not jump


def face_anchor(video, t):
    """Where the biggest face is in the 9:16 frame at t (its centre), for a jump cut's punch-in; None without one."""
    import facecam
    c = cv2.VideoCapture(video); c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = c.read(); c.release()
    if not ok:
        return None
    fs = [f for f in facecam.faces(img, 0.5) if f[2] > img.shape[1] * 0.06]
    if not fs:
        return None
    x, y, w, h = max(fs, key=lambda f: f[2] * f[3])
    return x + w / 2, y + h / 2


def tighten_cut(joined, timeline, words_all, S, tmp, zoom=True):
    """Jump cuts on a talk cut (pipeline/tighten.py, the talkpolish lab's method): the pauses and breaths inside every
    segment cut down to 0.12 s, the voice rebuilt as one track with 12 ms crossfades, jump cuts alternating 1.0x / 1.12x
    (pushed in around the face, so the seam reads as a style), dead air before the first word trimmed to 0.12 s. The pause
    right before the editor's beat is held at 0.22 s: it is part of the joke. Cards are left alone.
    Returns (new joined file, new timeline, report), or None when there's nothing worth cutting or the audio isn't talk
    (more than 40% would go)."""
    import tighten
    c = cv2.VideoCapture(joined); fps = c.get(cv2.CAP_PROP_FPS) or 30.0; c.release()
    snap = lambda t: round(t * fps) / fps
    beats = S.get('punch') or []
    pieces, stats, first = [], [], True
    for tl in timeline:
        if 'src' not in tl:
            pieces.append([snap(tl['at']), snap(tl['at'] + tl['dur']), tl, 0])
            continue
        a, b = tl['src']; at = tl['at']
        ws = [{'w': w['w'], 's': w['s'] - a + at, 'e': min(w['e'], b) - a + at} for w in words_all if a - 0.05 <= w['s'] < b]
        keep = [[p - a + at - 1.5, p - a + at + 0.1, 0.22] for p in beats if a <= p <= b]
        r = tighten.spans(joined, at, at + b - a, ws, keep, lead=0.12 if first else None)
        first = False
        stats.append(r['stats'])
        for k, (s0, s1) in enumerate(r['spans']):
            if snap(s1) > snap(s0):
                pieces.append([snap(s0), snap(s1), tl, k])
    src_len = sum(s['src'] for s in stats); removed = sum(s['removed'] for s in stats)
    rep = {'removed': round(removed, 2), 'cuts': sum(s['cuts'] for s in stats), 'per_segment': stats}
    if removed < 0.5 or removed > 0.4 * src_len:
        print(f'tighten: skipped ({removed:.1f} of {src_len:.1f} s would go)', file=sys.stderr)
        return None
    # punch-ins: the framing toggles at each jump cut inside a segment; a piece under 0.4 s keeps the one before it (a
    # flicker), and nothing is pushed around the beat, where the speaker camera already pushes in 1.15x
    beat_e = [tl['at'] + p - tl['src'][0] for p in beats for tl in timeline if 'src' in tl and tl['src'][0] <= p <= tl['src'][1]]
    z, prev_tl = False, None
    for p in pieces:
        if 'src' not in p[2]:
            p.append(False); z = False; continue
        if p[2] is not prev_tl:
            z = False
        elif p[1] - p[0] >= 0.4:
            z = not z
        prev_tl = p[2]
        p.append(bool(zoom and z and not any(p[0] - 1.5 < be < p[1] + 0.2 for be in beat_e)))
    ZW, ZH = int(W * 1.12) // 2 * 2, int(H * 1.12) // 2 * 2
    n = len(pieces)
    fc = f"[0:v]split={n}" + ''.join(f'[s{k}]' for k in range(n)) + ';'
    for k, (e0, e1, tl, j, zz) in enumerate(pieces):
        zf = ''
        if zz:
            fx, fy = face_anchor(joined, (e0 + e1) / 2) or (W / 2, H * 0.38)
            x = int(max(0, min(ZW - W, fx * (ZW / W - 1)))); y = int(max(0, min(ZH - H, fy * (ZH / H - 1))))
            zf = f',scale={ZW}:{ZH}:flags=lanczos,crop={W}:{H}:{x}:{y}'
        fc += f"[s{k}]trim=start_frame={round(e0 * fps)}:end_frame={round(e1 * fps)},setpts=PTS-STARTPTS{zf},setsar=1[p{k}];"
    fc += ''.join(f'[p{k}]' for k in range(n)) + f"concat=n={n}:v=1:a=0,format=yuv420p[v]"
    wav = tighten.voice_wav(joined, [(p[0], p[1]) for p in pieces], os.path.join(tmp, 'tight.wav'))
    out = os.path.join(tmp, 'tight.mkv')
    ff('-i', joined, '-i', wav, '-filter_complex', fc, '-map', '[v]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'fast', '-crf', 16,
       '-c:a', 'pcm_s16le', '-ar', 48000, out)
    new, t = [], 0.0
    for e0, e1, tl, j, zz in pieces:
        if 'src' in tl:
            a = tl['src'][0] + e0 - tl['at']
            new.append({'src': [round(a, 3), round(a + e1 - e0, 3)], 'at': round(t, 3), **({'zoom': 1.12} if zz else {})})
        else:
            new.append({**tl, 'at': round(t, 3), 'dur': round(e1 - e0, 3)})
        t += e1 - e0
    rep['zoomed'] = sum(1 for p in pieces if p[4])
    return out, new, rep


def loop_words(words):
    """ids of the words in song lyrics or whisper loops, which captions skip (the audio is untouched): a line (sentence end
    or a 0.45 s pause) whose word trigrams keep coming back, median >= 3 in the cut, is a chorus or a hallucination, not
    speech. The fx session's auto_reality.make_lines test; its annotation edits were captioning lyrics as speech."""
    norm = lambda t: re.sub(r"[^a-z0-9' ]", '', t.lower())
    lines, cur = [], []
    for w in words:
        if cur and (cur[-1]['w'].rstrip().endswith(('.', '?', '!')) or w['s'] - cur[-1]['e'] > 0.45):
            lines.append(cur); cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    toks = [norm(w['w']) for w in words]
    grams = {}
    for i in range(len(toks) - 2):
        g = tuple(toks[i:i + 3]); grams[g] = grams.get(g, 0) + 1
    out = set()
    for ln in lines:
        t = [norm(w['w']) for w in ln]
        gs = [grams.get(tuple(t[i:i + 3]), 0) for i in range(len(t) - 2)]
        if gs and float(np.median(gs)) >= 3:
            out |= {id(w) for w in ln}
    return out


def chunk_talk(words, big=()):
    """Caption chunks for a tightened talk cut (the talkpolish lab's chunker): hard breaks at sentence ends, pauses over
    0.28 s and big-word edges; inside a phrase a small DP picks chunks of 1-3 words, never ending on "the"/"my"/"a"
    when that can be avoided, under 15 characters, not on screen under ~0.3 s, a lone word only when it carries colour."""
    isbig = lambda w: any(w is x for x in big)
    phrases, cur = [], []
    for w in words:
        if cur:
            p = cur[-1]
            if p['w'].rstrip().endswith(('.', '?', '!')) or w['s'] - p['e'] > 0.28 or isbig(w) != isbig(p):
                phrases.append(cur); cur = []
        cur.append(w)
    if cur:
        phrases.append(cur)

    def cost(c, last):
        k = 0.4
        n = sum(len(x['w'].strip()) + 1 for x in c) - 1
        if n > 14:
            k += 2.0 + 0.5 * (n - 14)
        if re.sub(r"[^a-z']", '', c[-1]['w'].lower()) in FUNC and not last and not c[-1]['w'].rstrip().endswith((',', '.', '?', '!')):
            k += 3.0
        k += sum(0.9 for x in c[:-1] if x['w'].rstrip().endswith(','))
        if len(c) == 1 and not c[0].get('col'):
            k += 0.8
        if c[-1]['e'] - c[0]['s'] < 0.32 and not last:
            k += 2.0
        return k
    out = []
    for ph in phrases:
        if isbig(ph[0]):
            out.append(ph); continue
        n = len(ph); best = [0.0] + [1e9] * n; back = [0] * (n + 1)
        for j in range(1, n + 1):
            for i in range(max(0, j - 3), j):
                c = best[i] + cost(ph[i:j], j == n)
                if c < best[j]:
                    best[j], back[j] = c, i
        cuts, j = [], n
        while j > 0:
            cuts.append((back[j], j)); j = back[j]
        out += [ph[i:j] for i, j in reversed(cuts)]
    return out


def big_words(words, beats_e, picks=(), hook='', credit=''):
    """The 1-3 words that go huge (the lab's punch captions: KNIFE / DRENCHED / THAT'S INSANE on the story's beats, never
    more than 3-4). In order: the live editor's picks (judge.py "big"), the punchline (a last line of one or two words, else
    the last word of the last full sentence: "...why is your pillow ORANGE?"), the word said on the editor's beat (the
    loudest second), a shout that is its own sentence ("BRO." "WHAT?"), and a word of the hook. Never a swear or a filler;
    at least 2.5 s apart."""
    norm = lambda t: re.sub(r"[^a-z0-9']", '', t.lower())
    WEAK = {'though', 'actually', 'literally', 'right', 'still', 'even', 'too', 'anyway', 'man', 'dude', 'lol', 'huh', 'alright', 'okay'}
    content = lambda w: norm(w['w']) not in STOP_EMPH | FUNC | WEAK and len(re.sub(r'[^A-Za-z0-9]', '', w['w'])) >= 2 and censor(w['w']) == w['w']
    end = lambda w: w['w'].rstrip().endswith(('.', '?', '!'))
    cand = []
    for p in picks:
        w = next((w for w in words if norm(w['w']) == norm(p) and content(w)), None)
        if w:
            cand.append(w)
    tail, k = [], len(words)
    while k > 0 and not (tail and end(words[k - 1])):
        k -= 1; tail.insert(0, words[k])
    if not (1 <= len(tail) <= 2 and any(content(w) for w in tail) and all(censor(w['w']) == w['w'] for w in tail)
            and sum(len(w['w']) for w in tail) <= 14):
        last = next((words[i] for i in range(k - 1, -1, -1) if end(words[i])), None) if not any(content(w) for w in tail) else None
        tail = [last] if last is not None and content(last) and len(last['w']) <= 12 else []
    cand += tail
    for be in beats_e:
        w = next((w for w in words if w['e'] >= be - 0.15 and content(w)), None)
        if w and w['s'] - be < 0.8:
            cand.append(w)
    for i, w in enumerate(words):
        if (i == 0 or end(words[i - 1])) and end(w) and re.sub(r'[^A-Z]', '', w['w'].upper()) in SHOUT:
            cand.append(w)
    skip = {norm(x)[:5] for x in credit.split()}
    hw = {norm(x)[:5] for x in hook.split() if len(norm(x)) >= 6 and norm(x)[:5] not in skip}
    cand += [w for w in words if norm(w['w'])[:5] in hw and len(norm(w['w'])) >= 6 and content(w)][:1]
    intail = lambda w: any(w is t for t in tail)
    out = []
    for w in cand:
        if len(out) >= 3 and not intail(w):
            continue
        if all(w is not x for x in out) and all(abs(w['s'] - x['s']) >= 2.5 or (intail(w) and intail(x)) for x in out):
            out.append(w)
    return sorted(out, key=lambda w: w['s'])


_CAPFONT = {}


def cap_width(text, size=150):
    """Pixel width of a caption in Anton at this size (fits the chunk to the frame)."""
    from PIL import ImageFont
    if size not in _CAPFONT:
        _CAPFONT[size] = ImageFont.truetype(os.path.join(FONTS, 'Anton-Regular.ttf'), size)
    return _CAPFONT[size].getlength(re.sub(r'\{[^}]*\}', '', text))


def tone(path, dur, sr=48000):
    """A short whoosh for cards (synthesized, no samples)."""
    t = np.arange(int(sr * dur)) / sr
    n = np.random.default_rng(3).standard_normal(len(t))
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    sig = (n * 0.12 + np.sin(2 * np.pi * (180 + 600 * t / dur) * t) * 0.06) * env
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr); wv.writeframes(pcm.tobytes())


def main(spec_path):
    S = json.load(open(spec_path))
    base = os.path.dirname(os.path.abspath(spec_path))
    P = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    audio = P(S['audio'])
    # one VOD file, or per-span section files ({file, offset}) from hls_sections.py
    videos = [{'file': P(v['file']), 'offset': v['offset'], 'span': v['span']} for v in S['videos']] if S.get('videos') else [{'file': P(S['video']), 'offset': 0.0, 'span': [0, 1e9]}]
    pick = lambda a, b: next(v for v in videos if v['span'][0] - 5 <= a and b <= v['span'][1] + 5)
    layout = S.get('layout', 'hooksplit')
    global TOP_H, CENTER_H, CENTER_Y
    # split: the cam panel's height. 608 = a 16:9 cam at full width (trades: more room for the chart); 960 = half and half
    # (someone explaining). center: the footage's box, and the hook sits just above it
    TOP_H = int(S.get('top_h', round(W * 382 / 678 / 2) * 2)) // 2 * 2
    global GAME_H
    GAME_H = int(S.get('game_h', 1010)) // 2 * 2         # gamesplit: the game panel; the cam gets the rest
    if layout == 'gamesplit':
        TOP_H = GAME_H                                   # the hook sits on the seam
    if layout == 'fillersplit':
        TOP_H = int(S.get('top_h', 1100)) // 2 * 2      # the stream panel; the filler loop takes the rest
    # action captions (actioncap.py): stage directions like *GRABS* / *RUNS*, the style of the 94-164M-view Shorts. For a
    # physical moment or one with little talk on a full-frame cut; word-by-word pop words otherwise
    PHYSICAL = re.compile(r'\b(slam|suplex|grab|run|chase|fall|fell|trip|throw|threw|splash|prank|fight|punch|jump|steal|stole|tackle|dance|scare|jumpscare'
                          r'|dies|died|death|killed|kills|shot|explode|blew up|run ends|lava|creeper)', re.I)
    if not S.get('captions') and layout in ('speaker', 'fillersplit', 'gamesplit'):
        T0 = json.load(open(P(S['words'])))
        nw = sum(1 for s_ in T0['segments'] for w in s_['words'] if any(a_ - 0.2 <= w['s'] <= b_ for a_, b_ in S['segments']))
        dur_ = sum(b_ - a_ for a_, b_ in S['segments'])
        if PHYSICAL.search(S.get('hook') or '') or nw / max(1.0, dur_) < 1.3:
            S['captions'] = 'action'
    # the news-headline hook: asked for, or when the editor's hook already carries a CAPS punch word on a full-frame cut
    headline = S.get('hook_style') == 'headline' or (S.get('hook_style') is None and layout in ('speaker', 'letterbox')
                                                     and re.search(r"\b[A-Z]{4,}\b", S.get('hook') or '') is not None)
    CENTER_H = int(S.get('center_h', 980)) // 2 * 2; CENTER_Y = int(S.get('center_y', 500))
    T = json.load(open(P(S['words'])))
    words_all = [w for s in T['segments'] for w in s['words']]
    out_dir = P(S['out_dir']); os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='edit-')
    parts, timeline, t_edit = [], [], 0.0
    cards = {int(k): v for k, v in (S.get('cards') or {}).items()}
    report = {'segments': []}
    def add_card(k):
        """A title/date card (black, centred caps, synthesized whoosh) at this point of the edit; '\\N' breaks lines."""
        nonlocal t_edit
        cd = float(S.get('card_s', 1.0))
        card = os.path.join(tmp, f'card{k}.mp4'); wav = os.path.join(tmp, f'card{k}.wav'); tone(wav, cd)
        ass = os.path.join(tmp, f'card{k}.ass')
        open(ass, 'w').write(ass_header() + f"Dialogue: 0,{ass_time(0)},{ass_time(cd)},Card,,0,0,0,,{{\\an5\\pos({W // 2},{H // 2})\\fad(80,80)}}{cards[k].upper()}\n")
        ff('-f', 'lavfi', '-i', f'color=c=0x0d0d10:s={W}x{H}:d={cd}:r=30', '-i', wav,
           '-vf', f"ass={ass}:fontsdir={FONTS},format=yuv420p", '-c:v', 'libx264', '-crf', 18, '-c:a', 'aac', '-ar', 48000, '-ac', 2, '-shortest', card)
        parts.append(card)
        timeline.append({'card': cards[k], 'at': t_edit, 'dur': cd})
        t_edit += cd

    nseg = len(S['segments'])
    for i, (a, b) in enumerate(S['segments']):
        if i - 1 in cards:                     # cards[k] plays after segment k; cards[-1] opens the edit
            add_card(i - 1)
        V = pick(a, b); video = V['file']; off = V['offset']
        if layout == 'letterbox':
            # two people far apart (a vertical crop would cut one out): the whole frame, bars above and below, the way the
            # 900K-view clip pages post it; the caption sits on the video in TikTok's plain white text
            seg = os.path.join(tmp, f'seg{i}.mp4')
            fc = (f"[0:v]fps=30,scale={W}:-2:flags=lanczos,setsar=1[v0];color=c=black:s={W}x{H}:r=30[bg];"
                  f"[bg][v0]overlay=0:(H-h)/2:shortest=1,setsar=1,format=yuv420p[v]")
            ff('-ss', f'{a - off:.3f}', '-to', f'{b - off:.3f}', '-i', video, '-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', audio,
               '-filter_complex', fc, '-map', '[v]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', 18,
               '-c:a', 'aac', '-ar', 48000, '-ac', 2, '-b:a', '192k', seg)
            parts.append(seg); report['segments'].append({'src': [a, b], 'cam': {'side': 'letterbox'}})
            timeline.append({'src': [a, b], 'at': t_edit}); t_edit += b - a
            continue
        if layout in ('center', 'pip'):
            # no facecam (a trading terminal, a game): the footage zoomed in the middle over a blurred, dimmed copy of the
            # stream; the hook box sits above it, subtitles near its bottom edge.
            # pip: the same, with the streamer's (small) facecam as a rounded inset in the lower left of the footage, the
            # 22.8M-view Jynxzi page's layout: a small cam stays sharp instead of blown up to a full-width panel
            seg = os.path.join(tmp, f'seg{i}.mp4')
            sc = S.get('screen') or {}
            ch = int(sc.get('h', 972)); cw = min(1920, int(round(ch * W / CENTER_H / 2) * 2))
            cx = int(sc.get('x', (1920 - cw) // 2)); cy = int(sc.get('y', (1080 - ch) // 2))
            if layout == 'pip' and 'x' in sc:          # the content crop was sized for a taller panel: keep its centre
                cx = int(sc['x'] + round(sc['h'] * W / 1312) // 2 - cw // 2)
            cx = max(0, min(1920 - cw, cx)); cy = max(0, min(1080 - ch, cy))
            ins = S.get('inset')                         # a fixed piece of the stream kept in view, e.g. a trader's balance/PnL box
            cam_ = S.get('cam') if layout == 'pip' else None
            cc_ = S.get('cam') if S.get('camcard') else None
            if cc_:                                      # the facecam alone (its chat/dashboard beside it isn't content): full
                cw2, ch2 = cc_['w'] // 2 * 2, cc_['h'] // 2 * 2       # width, centred, over a blurred copy of itself
                ih2 = int(round(W * ch2 / cw2 / 2) * 2)
                fc = (f"[0:v]fps=30,scale=1920:1080,setsar=1,split=2[a][b];[a]crop={cw2}:{ch2}:{cc_['x']}:{cc_['y']},scale={W}:{ih2}:flags=lanczos[mid];"
                      f"[b]crop={cw2}:{ch2}:{cc_['x']}:{cc_['y']},scale=-2:{H},crop={W}:{H},gblur=sigma=34,eq=brightness=-0.28[bg];"
                      f"[bg][mid]overlay=0:{CENTER_Y + (CENTER_H - ih2) // 2},setsar=1,format=yuv420p[v]")
            elif cam_:
                iw = int(W * 0.42) // 2 * 2; ih = int(round(iw * cam_['h'] / cam_['w'] / 2) * 2); r = 28
                rnd = (f"if(lt(X,{r})*lt(Y,{r})*gt(hypot({r}-X,{r}-Y),{r}),0,if(gt(X,W-{r})*lt(Y,{r})*gt(hypot(X-(W-{r}),{r}-Y),{r}),0,"
                       f"if(lt(X,{r})*gt(Y,H-{r})*gt(hypot({r}-X,Y-(H-{r})),{r}),0,if(gt(X,W-{r})*gt(Y,H-{r})*gt(hypot(X-(W-{r}),Y-(H-{r})),{r}),0,255))))")
                # the cam's own corner of the stream, where it reaches into the content crop, is blurred out (one face, not two)
                import gamecam
                hx, hy, hw, hh = gamecam.hide_rects([cam_], 1920)[0]
                x0_, y0_ = max(hx, cx), max(hy, cy); x1_, y1_ = min(hx + hw, cx + cw), min(hy + hh, cy + ch)
                hide = ''
                if x1_ - x0_ >= 16 and y1_ - y0_ >= 16:
                    rx, ry, rw, rh = x0_ - cx, y0_ - cy, (x1_ - x0_) // 2 * 2, (y1_ - y0_) // 2 * 2
                    hide = f"split=2[ma][mb];[mb]crop={rw}:{rh}:{rx}:{ry},gblur=sigma=24,eq=brightness=-0.15[mq];[ma][mq]overlay={rx}:{ry},"
                fc = (f"[0:v]fps=30,scale=1920:1080,setsar=1,split=3[a][b][i];[a]crop={cw}:{ch}:{cx}:{cy},{hide}scale={W}:{CENTER_H}:flags=lanczos[mid];"
                      f"[b]scale=-2:{H},crop={W}:{H},gblur=sigma=34,eq=brightness=-0.28[bg];"
                      f"[i]crop={cam_['w']}:{cam_['h']}:{cam_['x']}:{cam_['y']},scale={iw}:{ih}:flags=lanczos,format=yuva420p,"
                      f"geq=lum='p(X,Y)':cb='p(X,Y)':cr='p(X,Y)':a='{rnd}'[ins];"
                      f"[bg][mid]overlay=0:{CENTER_Y}[bm];[bm][ins]overlay=32:{CENTER_Y + CENTER_H - ih - 32},setsar=1,format=yuv420p[v]")
            elif ins:
                iw = int(W * float(S.get('inset_w', 0.46))) // 2 * 2; ih = int(round(iw * ins[3] / ins[2] / 2) * 2)
                fc = (f"[0:v]fps=30,scale=1920:1080,setsar=1,split=3[a][b][i];[a]crop={cw}:{ch}:{cx}:{cy},scale={W}:{CENTER_H}:flags=lanczos[mid];"
                      f"[b]scale=-2:{H},crop={W}:{H},gblur=sigma=34,eq=brightness=-0.28[bg];[i]crop={ins[2]}:{ins[3]}:{ins[0]}:{ins[1]},scale={iw}:{ih}:flags=lanczos[ins];"
                      f"[bg][mid]overlay=0:{CENTER_Y}[bm];[bm][ins]overlay=28:{CENTER_Y + 28},setsar=1,format=yuv420p[v]")
            else:
                fc = (f"[0:v]fps=30,scale=1920:1080,setsar=1,split=2[a][b];[a]crop={cw}:{ch}:{cx}:{cy},scale={W}:{CENTER_H}:flags=lanczos[mid];"
                      f"[b]scale=-2:{H},crop={W}:{H},gblur=sigma=34,eq=brightness=-0.28[bg];[bg][mid]overlay=0:{CENTER_Y},setsar=1,format=yuv420p[v]")
            ff('-ss', f'{a - off:.3f}', '-to', f'{b - off:.3f}', '-i', video, '-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', audio,
               '-filter_complex', fc, '-map', '[v]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', 18,
               '-c:a', 'aac', '-ar', 48000, '-ac', 2, '-b:a', '192k', seg)
            parts.append(seg); report['segments'].append({'src': [a, b], 'cam': {'side': layout}, 'screen': [cx, cy, cw, ch]})
            timeline.append({'src': [a, b], 'at': t_edit}); t_edit += b - a
            continue
        if layout in ('titlebar', 'rank'):
            seg = os.path.join(tmp, f'seg{i}.mp4')
            fc = ("[0:v]fps=30,scale=1280:720,setsar=1,split=3[a][b][c];[a]scale=1080:608:flags=lanczos[mid];"
                  "[b]scale=-2:912,crop=1080:912,gblur=sigma=30,eq=brightness=-0.22[low];"
                  "[c]scale=1080:400,drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill,setsar=1[top];[top][mid][low]vstack=inputs=3,setsar=1,format=yuv420p[v]")
            ff('-ss', f'{a - off:.3f}', '-to', f'{b - off:.3f}', '-i', video, '-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', audio,
               '-filter_complex', fc, '-map', '[v]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', 18,
               '-c:a', 'aac', '-ar', 48000, '-ac', 2, '-b:a', '192k', seg)
            parts.append(seg); report['segments'].append({'src': [a, b], 'cam': {'side': 'titlebar'}})
            timeline.append({'src': [a, b], 'at': t_edit}); t_edit += b - a
            continue
        if layout == 'gamesplit':
            # a game: the game on top at near full height, the cam below (gamecam.py); a duo or squad's cams take turns in
            # the lower panel, whoever is talking or reacting
            import gamecam
            cs = []
            for c in S.get('cams') or []:                       # source pixels -> the 1920x1080 frame the filter works on
                sx, sy = 1920 / c.get('src', [1920, 1080])[0], 1080 / c.get('src', [1920, 1080])[1]
                cs.append({**c, 'x': int(c['x'] * sx) // 2 * 2, 'y': int(c['y'] * sy) // 2 * 2, 'w': int(c['w'] * sx) // 2 * 2,
                           'h': int(c['h'] * sy) // 2 * 2, 'face': [c['face'][0] * sx, c['face'][1] * sy, c['face'][2] * sx, c['face'][3] * sy]})
            ov = S.get('overlays') or []                        # chat, alerts, counters: kept out of the game's window, blurred if they reach in
            gc = S.get('game') or gamecam.game_crop(cs, 1920, 1080, W, GAME_H, ov)
            pk = (S.get('punch') or [None])[0]
            runs = gamecam.speaker_runs(video, a - off, b - off, cs, (pk - off) if pk is not None else None) if len(cs) > 1 else [(a - off, b - off, 0)]
            for k, (ra, rb, ci) in enumerate(runs):
                cc = gamecam.cam_crop(cs[ci], 1920, 1080, W, H - GAME_H)
                seg = os.path.join(tmp, f'seg{i}_{k}.mp4')
                ff('-ss', f'{ra:.3f}', '-to', f'{rb:.3f}', '-i', video, '-ss', f'{ra + off:.3f}', '-to', f'{rb + off:.3f}', '-i', audio,
                   '-filter_complex', gamecam.filter_graph(gc, cc, W, GAME_H, H - GAME_H, gamecam.hide_rects(cs, 1920) + [tuple(r) for r in ov],
                                                         gamecam.cam_lift(video, (ra + rb) / 2, cc), gamecam.cam_lift(video, (ra + rb) / 2, gc)), '-map', '[v]', '-map', '1:a', '-c:v', 'libx264',
                   '-preset', 'medium', '-crf', 18, '-c:a', 'aac', '-ar', 48000, '-ac', 2, '-b:a', '192k', seg)
                parts.append(seg)
                report['segments'].append({'src': [round(ra + off, 2), round(rb + off, 2)], 'cam': {'side': 'game', 'cam': ci, 'card': cc['card']}, 'game': gc})
            timeline.append({'src': [a, b], 'at': t_edit}); t_edit += b - a
            continue
        if layout == 'fillersplit':
            # a sit-down conversation: the speaker camera on top (a squarer window, cut to whoever talks), our own
            # "satisfying" physics loop below for constant motion, the hook on the seam (the 11.2M-view Adin Ross layout;
            # pipeline/studio/filler.py renders the loop, never anyone's gameplay)
            import vcam
            sys.path.insert(0, os.path.join(HERE, 'studio'))
            import filler
            top_h = TOP_H
            seg_t = os.path.join(tmp, f'seg{i}_top.mp4')
            info = vcam.render(FF, video, a - off, b - off, audio, a, seg_t, punches=[t - off for t in S.get('punch', [])], threads=THREADS, out_h=top_h)
            fl = os.path.join(tmp, 'filler.mp4')
            if not os.path.exists(fl):
                tot = sum(y - x for x, y in S['segments']) + 2
                import zlib
                kinds = ['ring', 'trails', 'grow']
                k_ = S.get('filler') or kinds[zlib.crc32(S['id'].encode()) % 3]
                filler.render(fl, k_, tot, W, H - top_h, seed=zlib.crc32(S['id'].encode()) % 97)
            seg = os.path.join(tmp, f'seg{i}.mp4')
            ff('-i', seg_t, '-ss', f'{t_edit:.3f}', '-i', fl, '-filter_complex', '[0:v][1:v]vstack=inputs=2,setsar=1,format=yuv420p[v]',
               '-map', '[v]', '-map', '0:a', '-shortest', '-c:v', 'libx264', '-preset', 'medium', '-crf', 18, '-c:a', 'copy', seg)
            parts.append(seg); report['segments'].append({'src': [round(a, 2), round(b, 2)], 'cam': {'side': 'fillersplit', **{k: info[k] for k in ('cuts', 'seen')}}})
            timeline.append({'src': [a, b], 'at': t_edit}); t_edit += b - a
            continue
        seg_cam = (S.get('seg_cams') or [None] * nseg)[i] if i < len(S.get('seg_cams') or []) else None
        if layout == 'speaker' and not seg_cam:
            # a virtual camera operator: follows the subject smoothly, holds still in a dead zone, cuts on source cuts,
            # eases a 1.15x punch-in on the beat (spec['punch'], source seconds); one pass, no stitched pieces
            import vcam
            seg = os.path.join(tmp, f'seg{i}.mp4')
            import gamecam                               # the editor reads the frame's stream overlays (chat, counters, alerts)
            hide = [] if S.get('no_overlay_check') else gamecam.overlay_rects_span(video, a - off, b - off)
            info = vcam.render(FF, video, a - off, b - off, audio, a, seg, punches=[t - off for t in S.get('punch', [])], threads=THREADS, hide=hide,
                               push=S.get('captions') != 'action')
            parts.append(seg); report['segments'].append({'src': [round(a, 2), round(b, 2)], 'cam': {'side': 'speaker', **info}, 'hidden': hide})
            timeline.append({'src': [a, b], 'at': t_edit}); t_edit += b - a
            continue
        cam_spec = seg_cam or S.get('cam')          # spec cam / seg_cams[i]: a found overlay box (facecam.py)
        runs = [(a, b, {k: cam_spec[k] for k in ('x', 'y', 'w', 'h', 'side')})] if cam_spec else cam_runs(video, a, b)
        if not runs:
            # no facecam corner after all (an IRL frame can look like one for a sample): film it with the speaker camera
            import vcam
            seg = os.path.join(tmp, f'seg{i}.mp4')
            info = vcam.render(FF, video, a - off, b - off, audio, a, seg, punches=[t - off for t in S.get('punch', [])], threads=THREADS)
            parts.append(seg); report['segments'].append({'src': [round(a, 2), round(b, 2)], 'cam': {'side': 'speaker', 'fallback': 'no facecam', **info}})
            timeline.append({'src': [a, b], 'at': t_edit}); t_edit += b - a
            continue
        sc = (seg_cam or {}).get('screen') or S.get('screen') or {}
        for k, (ra, rb, cam) in enumerate(runs):
            # screen crop: away from the cam corner and the sponsor strip, aspect matched to the lower panel
            top_h = TOP_H                           # same panel height for every segment, so cuts don't jump
            low_h = H - top_h
            crop_h = sc.get('h', 840)
            crop_w = round(crop_h * W / low_h / 2) * 2
            # default: hug the side away from the cam, so the cam never shows twice
            crop_x = sc.get('x', 688 if cam['side'] == 'left' else 1920 - 688 - crop_w if cam['side'] == 'right' else 960 - crop_w // 2)
            crop_y = sc.get('y', 90)
            seg = os.path.join(tmp, f'seg{i}_{k}.mp4')
            # the cam box re-cut to the top panel's shape (a taller panel takes a narrower slice, centred), so faces never stretch
            cw_, ch_ = cam['w'], cam['h']
            if cw_ / ch_ > W / top_h:
                cw_ = round(ch_ * W / top_h / 2) * 2
            else:
                ch_ = round(cw_ * top_h / W / 2) * 2
            cx_ = cam['x'] + (cam['w'] - cw_) // 2; cy_ = cam['y'] + (cam['h'] - ch_) // 2
            pl = sc.get('player')
            if pl and cam['side'] != 'full':
                # a video playing inside a page (an X post, a YouTube tab): the player alone, full width, over a blurred copy
                # of itself, not the page around it
                px, py, pw_, ph_ = (int(v) // 2 * 2 for v in pl)
                ph_ = int(ph_ * 0.88) // 2 * 2               # the player's controls (play, timeline) sit along its bottom edge
                fc = (f"[0:v]split=2[c][s];[c]crop={cw_}:{ch_}:{cx_}:{cy_},scale={W}:{top_h}:flags=lanczos,setsar=1[top];"
                      f"[s]crop={pw_}:{ph_}:{px}:{py},split=2[pa][pb];[pa]scale=-2:{low_h},crop={W}:{low_h},gblur=sigma=30,eq=brightness=-0.25[bg];"
                      f"[pb]crop=iw/1.3:ih:(iw-iw/1.3)/2:0,scale={W}:-2:flags=lanczos,setsar=1[pv];[bg][pv]overlay=0:(H-h)/2[low];"
                      f"[top][low]vstack=inputs=2,fps=30,setsar=1,format=yuv420p[v]")
            else:
                fc = (f"[0:v]split=2[c][s];[c]crop={cw_}:{ch_}:{cx_}:{cy_},scale={W}:{top_h}:flags=lanczos,setsar=1[top];"
                      f"[s]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={W}:{low_h}:flags=lanczos{',gblur=sigma=28,eq=brightness=-0.12' if cam['side'] == 'full' else ''}[low];"
                      f"[top][low]vstack=inputs=2,fps=30,setsar=1,format=yuv420p[v]")
            ff('-ss', f'{ra:.3f}', '-to', f'{rb:.3f}', '-i', video, '-ss', f'{ra:.3f}', '-to', f'{rb:.3f}', '-i', audio,
               '-filter_complex', fc, '-map', '[v]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', 18,
               '-c:a', 'aac', '-ar', 48000, '-ac', 2, '-b:a', '192k', seg)
            parts.append(seg)
            report['segments'].append({'src': [round(ra, 2), round(rb, 2)], 'cam': cam, 'screen': [crop_x, crop_y, crop_w, crop_h]})
        timeline.append({'src': [a, b], 'at': t_edit})
        t_edit += b - a
    if nseg - 1 in cards:
        add_card(nseg - 1)
    # concat
    lst = os.path.join(tmp, 'list.txt')
    open(lst, 'w').write(''.join(f"file '{p}'\n" for p in parts))
    joined = os.path.join(tmp, 'joined.mp4')
    ff('-f', 'concat', '-safe', 0, '-i', lst, '-c', 'copy', joined)
    # a talk cut on a full frame gets jump cuts: the pauses go, the voice is one crossfaded track (tighten_cut)
    tight = None
    talk = layout in TIGHT_LAYOUTS or (layout == 'center' and (S.get('camcard') or S.get('irl_fill')))
    if S.get('tighten', talk) and S.get('captions') != 'action':
        nw_ = sum(1 for w in words_all if any(a_ - 0.05 <= w['s'] < b_ for a_, b_ in S['segments']))
        if nw_ / max(1.0, sum(b_ - a_ for a_, b_ in S['segments'])) >= 1.5:
            try:
                tight = tighten_cut(joined, timeline, words_all, S, tmp, zoom=layout != 'letterbox' and not S.get('irl_fill'))   # the IRL fill: never a punch-in
            except Exception as e:  # noqa: BLE001
                print('tighten failed, the cut stays as it was:', e, file=sys.stderr)
    if tight:
        joined, timeline, report['tighten'] = tight
        last = timeline[-1]
        t_edit = last['at'] + (last['dur'] if 'card' in last else last['src'][1] - last['src'][0])
    lift = '' if S.get('no_lift') else exposure_lift(joined)
    # a light unsharp mask on the picture (not the text, drawn after): crops of a 1080p stream are upscaled 1.8-2.5x and read
    # soft on a phone ("muddy, upscaled" was a frequent critic note)
    if not S.get('no_sharpen') and layout in ('speaker', 'fillersplit', 'gamesplit', 'hooksplit', 'pip', 'center'):
        lift = (lift or '') + 'unsharp=5:5:0.55:5:5:0.0,'
    if lift:
        report['exposure_lift'] = lift.rstrip(',')
    total = t_edit
    # captions on the edited timeline: one caption on screen at a time. Word times from the ASR can overlap or run
    # backwards (two people talking over each other), so straighten them first, then end every caption exactly where the
    # next one starts.
    caps = []
    talk_ws = []
    bleeps = []                                          # slurs in the audio get a bleep, the way the captions get a mask
    for tl in timeline:
        if 'src' not in tl:
            continue
        a, b = tl['src']
        # transcript order is reading order. A tightened cut's pieces are short and whisper's word ends run on through the
        # pause that was cut, so there a word belongs to the piece it starts in
        ws = [dict(w) for w in words_all if (a - 0.05 <= w['s'] < b if tight else (w['s'] >= a - 0.05 and w['e'] <= b + 0.2))]
        for w in ws:
            w['s'] = w['s'] - a + tl['at']; w['e'] = min(w['e'], b) - a + tl['at']
        for k in range(1, len(ws)):                      # monotonic, no overlap between words
            ws[k]['s'] = max(ws[k]['s'], ws[k - 1]['s'] + 0.04)
            ws[k - 1]['e'] = min(max(ws[k - 1]['e'], ws[k - 1]['s'] + 0.04), ws[k]['s'])
        bleeps += [(max(0, w['s'] - 0.04), w['e'] + 0.06) for w in ws if SLUR_AUDIO.search(w['w'])]
        # letterbox: the caption alone, no subtitles (the 924K-view letterbox posts carry one caption and nothing under it;
        # subtitles in the black bar read as clutter). Split layouts: huge pop words just above the hook on the seam, the
        # way the biggest split clips do it (an 11.2M-view Adin Ross page, 28M CaseOh; research/references/STYLES.md)
        style = S.get('captions') or ('none' if headline else {'hooksplit': 'pop', 'gamesplit': 'pop', 'fillersplit': 'pop', 'titlebar': 'pop', 'center': 'subs', 'pip': 'pop', 'letterbox': 'none'}.get(layout, 'pop'))
        if tight and style == 'pop':
            style = 'punch'
        if style == 'none':
            pass
        elif style == 'punch':
            talk_ws += ws                                # chunked once over the whole cut below: a phrase runs across jump cuts
        elif style == 'action':
            import actioncap
            V0 = pick(a, b)
            lf = os.path.join(out_dir, S['id'] + '.labels.json')        # the editor's labels, kept: re-renders stay the same
            cache = json.load(open(lf)) if os.path.exists(lf) else {}
            key = f"{a:.2f}-{b:.2f}"
            if key not in cache:
                cache[key] = actioncap.labels(V0['file'], a - V0['offset'], b - V0['offset'],
                                              [{**w, 's': w['s'] - V0['offset']} for w in words_all], S.get('hook') or '')
                json.dump(cache, open(lf, 'w'), indent=1, ensure_ascii=False)
            labs = cache[key]
            report.setdefault('action_labels', []).extend(labs)
            for x in labs:
                col = '&H0AD6FF&' if x['kind'] == 'action' else '&HFFFFFF&'
                caps.append([tl['at'] + x['t0'], tl['at'] + x['t1'], '{\\c' + col + '}' + censor(x['text']), True, 'Pop'])
        elif style == 'subs':
            # subtitles under the hook box, 3-4 words at a time, bold white with a heavy outline (readable over any game
            # or page), the word being said lit yellow
            for ch in chunks(ws, max_words=4, max_gap=0.6):
                for j, w in enumerate(ch):
                    st = w['s']; en = ch[j + 1]['s'] if j + 1 < len(ch) else ch[-1]['e'] + 0.15
                    txt = ' '.join(('{\\c&H0AD6FF&}' + censor(x['w']).strip(',.?!') + '{\\c&HFFFFFF&}') if k == j else censor(x['w']).strip(',.?!')
                                   for k, x in enumerate(ch))
                    caps.append([st, en, txt, j == 0, 'Subs'])
        elif style == 'pop':
            # word pop: one or two huge words at a time, white with a heavy outline; colour only on the words that carry the
            # moment (names, numbers, the hook's words, shouts), yellow then magenta, the way a 2.4M-view Kai Cenat page does it
            emph = emphasis(S.get('hook'), ws); n_e = 0; prev = None
            # no fillers and no stutters on screen ("UH", "I I I"): the pages caption what was meant, not every sound
            FILL = {'uh', 'um', 'uhh', 'umm', 'hmm', 'mm', 'ah', 'er', 'erm'}
            ws_p, last_w = [], None
            lw_ = loop_words(ws)                         # song lyrics / whisper loops: not captioned
            for w in ws:
                lw = re.sub(r"[^a-z']", '', w['w'].lower())
                if not lw or lw in FILL or lw == last_w or id(w) in lw_:
                    continue
                ws_p.append(w); last_w = lw
            # a laugh in a gap between words gets the tag the pages use (*LAUGHING*), only when it's clearly a laugh
            for w0, w1 in (zip(ws_p, ws_p[1:]) if S.get('laugh_tags') else ()):   # off until a real laughter classifier backs it
                if w1['s'] - w0['e'] >= 0.9:
                    g0 = w0['e'] + 0.1 - tl['at'] + a; g1 = w1['s'] - 0.05 - tl['at'] + a
                    if laughing(audio, g0, g1):
                        caps.append([w0['e'] + 0.1, w1['s'] - 0.05, '{\\c&HFFFFFF&}*LAUGHING*', True, 'Pop'])
                        report.setdefault('laughs', []).append([round(w0['e'] + 0.1, 2), round(w1['s'] - 0.05, 2)])
            for k, ch in enumerate(chunks(ws_p, max_words=1 if len(ws_p) < 60 else 2)):
                parts = []
                for x in ch:
                    t = censor(x['w']).upper().strip(',.?!')
                    if emph(x, prev):
                        parts.append('{\\c' + EMPH_COLORS[n_e % 2] + '}' + t + '{\\c&HFFFFFF&}'); n_e += 1
                    else:
                        parts.append(t)
                    prev = x
                caps.append([ch[0]['s'], ch[-1]['e'] + 0.1, '{\\c&HFFFFFF&}' + ' '.join(parts), True, 'Pop'])
        else:
            for ch in chunks(ws):
                for j, w in enumerate(ch):
                    st = w['s']; en = ch[j + 1]['s'] if j + 1 < len(ch) else ch[-1]['e'] + 0.12
                    txt = ' '.join(('{\\c&H0AD6FF&}' + censor(x['w']).upper().strip(',.?!') + '{\\c&HFFFFFF&}') if k == j else censor(x['w']).upper().strip(',.?!') for k, x in enumerate(ch))
                    caps.append([st, en, txt, j == 0, 'Cap'])
    if talk_ws:
        # punch captions (the talkpolish lab's pick of three caption systems): 1-3 words at a time at a readable size, and
        # one huge word on 1-3 beats; colour only on the words that carry the moment; the last line holds to the end
        FILL = {'uh', 'um', 'uhh', 'umm', 'hmm', 'mm', 'ah', 'er', 'erm'}
        ws_p, last_w = [], None
        for w in talk_ws:
            lw = re.sub(r"[^a-z']", '', w['w'].lower())
            if not lw or lw in FILL or lw == last_w:
                continue
            ws_p.append(w); last_w = lw
        lw_ = loop_words(ws_p)
        if lw_:
            report['lyrics_skipped'] = len(lw_)
            ws_p = [w for w in ws_p if id(w) not in lw_]
        emph = emphasis(S.get('hook'), ws_p); prev = None
        for w in ws_p:
            w['col'] = emph(w, prev); prev = w
        beats_e = []
        for p in S.get('punch') or []:
            tl = next((tl for tl in timeline if 'src' in tl and tl['src'][0] <= p <= tl['src'][1]), None) or \
                next((tl for tl in timeline if 'src' in tl and tl['src'][0] > p), None)
            if tl:
                beats_e.append(tl['at'] + max(0.0, p - tl['src'][0]))
        big = big_words(ws_p, beats_e, S.get('big') or [], S.get('hook') or '', S.get('credit') or '')
        report['big_words'] = [w['w'] for w in big]
        chs = chunk_talk(ws_p, big); n_e = 0
        for k, ch in enumerate(chs):
            isbig = any(ch[0] is x for x in big)
            parts = []
            for x in ch:
                t = censor(x['w']).upper().strip(',.?!')
                if isbig:
                    parts.append('{\\c&H0AD6FF&}' + t)
                elif x['col']:
                    parts.append('{\\c' + EMPH_COLORS[n_e % 2] + '}' + t + '{\\c&HFFFFFF&}'); n_e += 1
                else:
                    parts.append(t)
            txt = '{\\c&HFFFFFF&}' + ' '.join(parts)
            fit = 980 / max(1.0, cap_width(txt)) * 100
            caps.append([ch[0]['s'], total if k == len(chs) - 1 else ch[-1]['e'] + 0.1, txt, True, 'Pop', min(190 if isbig else 100, fit), isbig])
    caps.sort(key=lambda c: c[0])
    ev = []
    for k, (st, en, txt, first, sty, *size) in enumerate(caps):
        if k + 1 < len(caps):
            en = min(en, caps[k + 1][0])                 # hand over to the next caption, never overlap it
        if en - st < 0.05:
            continue
        sc_ = 78 if (sty == 'Pop' and layout == 'hooksplit') else 100      # a split's cam panel is short: smaller pop words
        if size:                                         # punch captions: their own size; a big word slams in from 135%
            sc_ = int(size[0])
        up, ms = (1.35, 120) if size and size[1] else (1.12, 80)
        pop = (f'\\fscx{int(sc_ * up)}\\fscy{int(sc_ * up)}\\t(0,{ms},\\fscx{sc_}\\fscy{sc_})' if first else (f'\\fscx{sc_}\\fscy{sc_}' if sc_ != 100 else ''))
        y = {'Subs': S.get('subs_y', (TOP_H + (H - TOP_H) * 0.44) if layout == 'hooksplit' else (GAME_H + (H - GAME_H) * 0.64) if layout == 'gamesplit' else (CENTER_Y + CENTER_H - 110) if layout in ('center', 'pip') else (H // 2 + W * 9 // 32 + 90) if layout == 'letterbox' else 1500), 'Pop': S.get('cap_y', (GAME_H - 235) if layout == 'gamesplit' else (TOP_H - 235) if layout == 'fillersplit' else ((TOP_H + (H - TOP_H) // 2 + int(W * 1.3 * (S['screen']['player'][3] * 0.88) / S['screen']['player'][2]) // 2 + 100)
                                            if (S.get('screen') or {}).get('player') else (TOP_H - 150)) if layout == 'hooksplit' else (CENTER_Y + 540) if layout == 'pip' else 1380 if layout == 'speaker' else 1160 if layout != 'titlebar' else 1340)}.get(sty, S.get('cap_y', 1330))
        ev.append(f"Dialogue: 1,{ass_time(st)},{ass_time(en)},{sty},,0,0,0,,{{\\an5\\pos({W // 2},{y}){pop}}}{txt}\n")
    hook_y = TOP_H if layout not in ('center', 'pip') else max(200, CENTER_Y - 120)
    hook_end = total if S.get('hook_full', True) else min(total, 6)
    if layout == 'speaker':
        hook_y = 300
    # the hook as TikTok draws text (white rounded boxes, TikTok Sans, colour emoji); libass box style is the fallback
    hook_png = None
    overlays = []   # (png, x, y, t0, t1): extra hook boxes, e.g. one caption per beat of a story edit (spec seg_titles)
    if S.get('seg_titles'):
        import hookbox
        srcs = [tl for tl in timeline if 'src' in tl]
        for k, (tl, text) in enumerate(zip(srcs, S['seg_titles'])):
            if not text:
                continue
            png = os.path.join(tmp, f'seg{k}.png')
            hb = hookbox.render(text, png, size=S.get('hook_size', 54))
            overlays.append((png, (W - hb['w']) // 2, max(130, hook_y - hb['h'] // 2), tl['at'], tl['at'] + tl['src'][1] - tl['src'][0]))
    hook_ev_plain = ''
    if layout == 'letterbox' and not overlays:          # the hook as plain text on the lower part of the video, no box
        hook_png = os.path.join(tmp, 'hook.png')
        hw, hh = plain_text_png(S['hook'], hook_png, W)
        hook_xy = ((W - hw) // 2, H // 2 + W * 9 // 64 - hh // 2)
    # hook_on_video=False: the YouTube Shorts copy. The biggest Shorts (85-132M views) carry no on-screen hook; the title
    # does that job (17 of 21 top Shorts in research/references), while most big TikToks burn the hook in
    if S.get('hook_on_video') is False:
        hook_png = None
    elif headline and not overlays:
        hook_png = os.path.join(tmp, 'hook.png')
        nw = len((S['hook'] or '').split())
        hw, hh = headline_png(S['hook'], hook_png, W, size=84 if nw <= 6 else 70 if nw <= 10 else 60)
        hy = (H // 2 + W * 9 // 64 - hh // 2) if layout == 'letterbox' else int(H * 0.69 - hh / 2)    # lower third, over the picture
        hook_xy = ((W - hw) // 2, hy)
        report['hook_style'] = 'headline'
    elif layout not in ('titlebar', 'rank', 'letterbox') and not overlays:
        try:
            import hookbox
            hook_png = os.path.join(tmp, 'hook.png')
            hb = hookbox.render(S['hook'], hook_png, size=S.get('hook_size', 54))
            hook_xy = ((W - hb['w']) // 2, max(130, hook_y - hb['h'] // 2))
            if layout == 'gamesplit':                      # the top of the game (sky, ceiling), below TikTok's own top bar: the
                hook_xy = ((W - hb['w']) // 2, 210)           # hotbar on the seam and the face below both stay clear
        except Exception as e:  # noqa: BLE001
            print('hookbox failed, using the libass hook:', e, file=sys.stderr)
            hook_png = None
    title = S.get('rank_title') or S['hook']
    if layout == 'rank':
        # ranking v2 (research/strategies): one accent word in the title, a "wait for #1" line under it
        accent = next((w for w in ('FUNNIEST', 'WILDEST', 'CRAZIEST', 'BEST', 'WORST', 'MOST', 'BIGGEST', 'FUNNY', 'WILD', 'TOP') if w in title.upper().split()), None)
        if accent:
            title = re.sub(rf'\b{accent}\b', lambda m: f'{{\\c&H0AD6FF&}}{accent}{{\\c&HFFFFFF&}}', title.upper(), count=1)
        title += '\\N{\\fs34\\c&H9CA3AF&}WAIT FOR #1'
    hook_ev = (f"Dialogue: 2,{ass_time(0)},{ass_time(hook_end)},Title,,0,0,0,,{{\\an5\\pos({W // 2},{210 if layout == 'rank' else 200})}}{title}\n" if layout in ('titlebar', 'rank')
               else '' if hook_png or overlays or layout == 'letterbox' or S.get('hook_on_video') is False else f"Dialogue: 2,{ass_time(0)},{ass_time(hook_end)},Hook,,0,0,0,,{{\\an5\\pos({W // 2},{hook_y})}}{S['hook']}\n")
    # no on-screen credit or watermark by default (the user, 2026-10-21: "why do we watermark our videos"; no program asks for
    # one, and platforms rank watermarked reposts lower); the caption's tags still name the streamer. CUTROOM_ONSCREEN_CREDIT=1 brings it back
    onscreen = os.environ.get('CUTROOM_ONSCREEN_CREDIT') == '1'
    credit_ev = f"Dialogue: 2,{ass_time(0)},{ass_time(total)},Credit,,0,0,0,,{S['credit']}\n" if onscreen and S.get('credit') else ''
    if onscreen and S.get('watermark'):
        credit_ev += f"Dialogue: 2,{ass_time(0)},{ass_time(total)},Mark,,0,0,0,,{{\\an2\\pos({W // 2},{S.get('mark_y', 1640)})}}{S['watermark'].upper()}\n"
    if layout == 'rank':
        items = [tl for tl in timeline if 'src' in tl]
        labels = S.get('rank') or []
        n = len(items)
        for k, tl in enumerate(items):
            num = n - k                                   # count down: #5 plays first, #1 last
            label = (labels[k] if k < len(labels) else '').upper()
            y = S.get('rank_y', 1060) + (num - 1) * 74     # #1 on top
            seg_end = tl['at'] + (tl['src'][1] - tl['src'][0])
            nc = RANK_COLORS[(num - 1) % len(RANK_COLORS)]       # each number its own color, the label white once played
            hook_ev += f"Dialogue: 3,{ass_time(0)},{ass_time(tl['at'])},Rank,,0,0,0,,{{\\pos(56,{y})\\c{nc}}}{num}.\n"
            hook_ev += f"Dialogue: 3,{ass_time(tl['at'])},{ass_time(seg_end)},Rank,,0,0,0,,{{\\pos(56,{y})\\fscx106\\fscy106\\t(0,120,\\fscx100\\fscy100)}}{{\\c{nc}}}{num}. {{\\c&H0AD6FF&}}{label}\n"
            hook_ev += f"Dialogue: 3,{ass_time(seg_end)},{ass_time(total)},Rank,,0,0,0,,{{\\pos(56,{y})}}{{\\c{nc}}}{num}. {{\\c&HFFFFFF&}}{label}\n"
    ass = os.path.join(tmp, 'captions.ass')
    open(ass, 'w').write(ass_header() + hook_ev + hook_ev_plain + credit_ev + ''.join(ev))
    out = os.path.join(out_dir, S['id'] + '.mp4')
    if hook_png:
        overlays.insert(0, (hook_png, hook_xy[0], hook_xy[1], 0, hook_end))
    gain = measured_gain(joined)
    if overlays:
        ins, chain, last = [], ([f"[0:v]{lift.rstrip(',')}[lv]"] if lift else []), ('lv' if lift else '0:v')
        for k, (png, x, y, t0, t1) in enumerate(overlays):
            ins += ['-loop', 1, '-i', png]
            chain.append(f"[{last}][{k + 1}:v]overlay={x}:{y}:shortest=1:enable='between(t,{t0:.2f},{t1:.2f})'[o{k}]")
            last = f'o{k}'
        fc = ';'.join(chain) + f";[{last}]ass={ass}:fontsdir={FONTS}[v];{audio_chain(bleeps, total, gain)}"
        ff('-i', joined, *ins, '-filter_complex', fc, '-map', '[v]', '-map', '[a]', '-shortest', '-ar', 48000,
           '-c:v', 'libx264', '-preset', PRESET, '-crf', 19, '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', out)
    else:
        ff('-i', joined, '-filter_complex', f"[0:v]{lift}ass={ass}:fontsdir={FONTS}[v];{audio_chain(bleeps, total, gain)}", '-map', '[v]', '-map', '[a]', '-ar', 48000,
           '-c:v', 'libx264', '-preset', PRESET, '-crf', 19, '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', out)
    # poster + post sheet
    ff('-ss', f"{S.get('poster_t', min(2.0, total / 3)):.2f}", '-i', out, '-frames:v', 1, '-q:v', 3, os.path.join(out_dir, S['id'] + '.jpg'))
    sheet = {k: S[k] for k in ('id', 'program', 'source_url', 'hook', 'post_caption', 'rules_check') if k in S}
    sheet.update({'duration': round(total, 2), 'timeline': timeline, 'render': report, 'file': os.path.basename(out)})
    json.dump(sheet, open(os.path.join(out_dir, S['id'] + '.json'), 'w'), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out} ({total:.1f}s)  layout runs: {[(r['src'], r['cam']['side']) for r in report['segments']]}")


if __name__ == '__main__':
    main(sys.argv[1])
