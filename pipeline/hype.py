#!/usr/bin/env python3
"""Hype edit: the IShowSpeed / deagzzz clip-page style for IRL moments (research/speedstyle/SPEEDSTYLE.md).

The moment stays the base (auto-framed by vcam, punch-ins on the beats). On top, from an editor plan:
  captions      1-4 words a card, wide black caps (TikTok Sans Wide), white with one neon keyword that glows, 115%->100% pop
  *stage*       *BARKS* / *SPEECHLESS* on beats with no words
  context       a label at the top for a whole stretch (*MEANWHILE ...*)
  name tags     NAME (ROLE) with a red pointer, pinned above the person's head (face found in the frame)
  stickers      at most 3 emoji on the payoff, anchored to a head
  color beats   red / green / yellow / magenta / black-and-white flash on the punchline, with a white pre-flash, a
                punch-in and a short shake, about one every 8-10 s
  verdict       a one-line verdict over the last beat, a fade to black, one second of black with the watermark
  sound         synthesized boom / whoosh / pop under the original audio (no samples, no music)
  watermark     @cutroom<streamer> and the streamer credit, bottom-left, every frame

Labels follow the guardrails: only what's said or visible; bystanders, fans and minors get neutral or kind labels.

usage: hype.py --clip portfolio/clips/<id>  [--handle cutroomn3on] [--judge sonnet]   (a clipcut/live clip dir)
       hype.py --src src.mp4 --audio src.m4a --words words.json --span a b --id name --streamer N3on --credit kick.com/n3on
"""
import argparse, json, math, os, re, subprocess, sys, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'live'))
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
FF = os.path.join(HERE, '..', '.venv', 'bin', 'ffmpeg')
FONTS = os.path.join(HERE, 'fonts')
W, H, FPS = 1080, 1920, 30
PAL = {'green': '39FF5A', 'red': 'FF2B2B', 'yellow': 'FFE600', 'cyan': '22E4FF', 'magenta': 'FF3BD4', 'purple': '9D5CFF', 'white': 'FFFFFF'}
TINT = {'red': (1.0, 0.25, 0.25), 'green': (0.35, 1.0, 0.45), 'yellow': (1.0, 0.95, 0.2), 'magenta': (1.0, 0.3, 0.9), 'purple': (0.6, 0.35, 1.0)}

RULES = """You are the editor of a big IRL clip page (the IShowSpeed / deagzzzshorts style: wide glowing caps, one neon keyword per
caption, *stage directions*, name tags over people, a color flash on the punchline, a verdict at the end).
Plan the edit for this moment. You get the transcript (word times in seconds from the clip start) and frames with their times.

Return JSON only:
{"title": "a sentence like Bro / Ain't NO WAY / W ... / ... is Crazy, plus two identical emojis",
 "kind": "funny|wholesome|chaos|aura",
 "keywords": [{"t": sec, "word": "ONE WORD FROM THE TRANSCRIPT", "color": "green|red|yellow|cyan|magenta"}],
 "stage": [{"t": sec, "dur": 0.8-2.5, "text": "*VERB* or *STATE*", "color": "..."}],
 "context": [{"t0": sec, "t1": sec, "text": "*SHORT CONTEXT*"}],
 "tags": [{"t0": sec, "t1": sec, "name": "1-3 WORDS", "sub": "(ROLE) or empty", "who": "how to spot them in the frame: clothes, hair, position", "color": "..."}],
 "stickers": [{"t": sec, "emoji": "one of ❗ 💧 👑 🥶 ☠️ ❤️‍🩹 😭 🔥", "who": "how to spot the person it sits on", "anchor": "above|beside|chest"}],
 "beats": [{"t": sec, "look": "red|green|yellow|magenta|purple|bw"}],
 "verdict": {"text": "2-4 WORDS: a superlative (CRAZIEST ... EVER), a W stamp (W DAD), a callback, or AURA : N", "emoji": "optional one emoji"}}

Rules:
- keywords: about one per 2-3 seconds of speech, the word that carries meaning. red = no/danger/yelling, green = yes/love/win,
  yellow = questions/shock, cyan = names/places, magenta = family/emotion.
- stage: only on beats with no speech where something visible happens (laughs, screams, walks off, speechless). 0-4 of them.
- tags: name people when who they are matters (ROLE: "N3ON'S MOM", NAME (ROLE), or TRAIT + WHO: "PURE AURA"). Only what is said
  or clearly visible; the role line too (no guessed moods). Put t0 where that person is clearly on screen in the frames.
  Fans, bystanders and minors get neutral or kind labels only. 0-2 tags.
- beats: 1 per 8-10 seconds at most, on the punchline or the loudest reaction. stickers: 0-3, only on the payoff.
- Never invent facts. No slurs, no sexual labels, no mocking real non-streamers.

MOMENT:
"""


def ass_c(hexrgb, alpha='00'):
    return f'&H{alpha}{hexrgb[4:6]}{hexrgb[2:4]}{hexrgb[0:2]}&'


def t_ass(t):
    t = max(0.0, t)
    return f'{int(t // 3600)}:{int(t // 60) % 60:02d}:{t % 60:05.2f}'


def ff(*a):
    r = subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', *map(str, a)], capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors='ignore')[-600:])
    return r.stdout


# ---------- sound: synthesized, never sampled ----------
SR = 48000


def _wav(path, sig):
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def sfx(kind, path):
    rng = np.random.default_rng(7)
    if kind == 'boom':        # 70 -> 38 Hz sweep, fast decay, a click on top
        t = np.arange(int(SR * 0.7)) / SR
        f = 38 + 32 * np.exp(-t / 0.12)
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.18) * 0.9
        sig[:int(SR * 0.015)] += rng.standard_normal(int(SR * 0.015)) * 0.4
        sig = np.tanh(sig * 1.6) * 0.5
    elif kind == 'whoosh':    # band-swept noise
        t = np.arange(int(SR * 0.3)) / SR
        n = rng.standard_normal(len(t))
        env = np.sin(np.pi * t / t[-1]) ** 2
        sig = np.convolve(n, np.ones(6) / 6, 'same') * env * 0.25
    else:                     # pop: 40 ms chirp 900 -> 250 Hz
        t = np.arange(int(SR * 0.05)) / SR
        f = 900 - 650 * t / t[-1]
        sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.02) * 0.45
    _wav(path, sig)


# ---------- faces in the rendered frame (for tags and stickers) ----------
def faces_at(video, t):
    import cv2, facecam
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = cap.read(); cap.release()
    if not ok:
        return []
    return [(x, y, w, h) for x, y, w, h in facecam.faces(img, 0.5)]


def who_box(video, t, name, who, judge_mode):
    """The face box of a named person at t: number the faces found in the frame and let the editor say which (or none)."""
    import cv2, facecam, judge as J
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = cap.read(); cap.release()
    if not ok:
        return None
    boxes = [tuple(int(v) for v in b) for b in facecam.faces(img, 0.5)]
    if not boxes:
        return None
    sheet = img.copy()
    for k, (x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(sheet, (x, y), (x + w, y + h), (0, 255, 255), 6)
        cv2.putText(sheet, str(k), (x + 8, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0, 255, 255), 8)
    jpg = cv2.imencode('.jpg', cv2.resize(sheet, (540, 960)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    ans = J._ask(f'Numbered yellow boxes mark the faces in this frame. Which box is {name} ({who})? Reply with only the number, '
                 f'or 0 if that person is not one of the boxes.', [jpg], 40, J.CLAUDE_MODELS.get(judge_mode, 'sonnet'))
    m = re.search(r'\d+', ans or '')
    k = int(m.group(0)) if m else 0
    return boxes[k - 1] if 1 <= k <= len(boxes) else None


def wide_scene(video, a, b):
    """Two or more faces far apart in most sampled frames: a group scene a 9:16 crop can't hold."""
    import cv2, facecam
    cap = cv2.VideoCapture(video); wide = n = 0
    for t in np.linspace(a + 0.5, b - 0.5, 6):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = cap.read()
        if not ok:
            continue
        n += 1
        fs = facecam.faces(img, 960 / img.shape[1])
        if len(fs) >= 2:
            xs = [x + w / 2 for x, y, w, h in fs]
            wide += (max(xs) - min(xs)) > 0.4 * img.shape[1]
    cap.release()
    return n and wide >= n / 3


def pick(boxes, who):
    if not boxes:
        return None
    if who == 'left':
        return min(boxes, key=lambda b: b[0])
    if who == 'right':
        return max(boxes, key=lambda b: b[0] + b[2])
    if who == 'center':
        return min(boxes, key=lambda b: abs(b[0] + b[2] / 2 - W / 2))
    return max(boxes, key=lambda b: b[2] * b[3])


def emoji_png(ch, out, size=170):
    from PIL import Image, ImageDraw, ImageFont
    import hookbox
    f = ImageFont.truetype(hookbox.EMOJI, 109, layout_engine=ImageFont.Layout.RAQM)
    box = f.getbbox(ch)
    im = Image.new('RGBA', (box[2] - box[0] + 8, box[3] - box[1] + 8), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((4 - box[0], 4 - box[1]), ch, font=f, embedded_color=True)
    im = im.crop(im.getbbox() or (0, 0, im.width, im.height))
    im = im.resize((round(im.width * size / im.height), size), Image.LANCZOS)
    im.save(out)
    return im.width, im.height


# ---------- captions ----------
def cards(words, dur):
    """1-4 words a card: break on punctuation, a gap over 0.25 s, or past 22 characters."""
    out, cur = [], []
    for w in words:
        txt = w['w'].strip()
        if not txt:
            continue
        if cur and (len(cur) >= 4 or w['s'] - cur[-1]['e'] > 0.25 or len(' '.join(x['w'] for x in cur) + txt) > 22):
            out.append(cur); cur = []
        cur.append(w)
        if re.search(r'[.!?,]$', txt):
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    res = []
    for k, c in enumerate(out):
        a = max(0.0, c[0]['s'] - 0.05)
        nxt = out[k + 1][0]['s'] - 0.05 if k + 1 < len(out) else min(dur, c[-1]['e'] + 0.6)
        b = min(max(a + 0.35, nxt), a + 1.6, dur)
        res.append((a, b, c))
    return res


def main(a):
    import judge as J, vcam
    import edit as E
    if a.clip:
        d = os.path.abspath(a.clip); cid = os.path.basename(d)
        spec = json.load(open(os.path.join(d, 'spec.json')))
        src, aud = os.path.join(d, spec['video']), os.path.join(d, spec['audio'])
        wpath = os.path.join(d, spec['words'])
        s0, s1 = spec['segments'][0]
        streamer = a.streamer or spec.get('credit', '').split(' · ')[0]
        credit = a.credit or spec.get('source_url', '').replace('https://', '').split('/clips')[0] or spec.get('credit')
    else:
        d = os.path.abspath(a.out or os.path.dirname(a.src)); cid = a.id
        src, aud, wpath = a.src, a.audio, a.words
        s0, s1 = a.span; streamer, credit = a.streamer, a.credit
    if a.start is not None:
        s0 = a.start
    if a.end is not None:
        s1 = a.end
    handle = a.handle or 'cutroom' + re.sub(r'[^a-z0-9]', '', (streamer or '').lower())
    out_id = f'{cid}-hype'
    dur = s1 - s0
    T = json.load(open(wpath))
    words = [{**w, 's': w['s'] - s0, 'e': w['e'] - s0} for seg in T['segments'] for w in seg['words'] if s0 - 0.05 <= w['s'] <= s1 + 0.1]

    # 1. the plan
    pf = os.path.join(d, f'{out_id}.plan.json')
    if a.replan or not os.path.exists(pf):
        frames, ftimes = [], []
        for k in range(8):
            t = dur * (k + 0.5) / 8
            try:
                frames.append(ff('-ss', f'{s0 + t:.2f}', '-i', src, '-frames:v', 1, '-vf', 'scale=360:-2', '-q:v', 6, '-f', 'image2pipe', '-c:v', 'mjpeg', '-'))
                ftimes.append(round(t, 1))
            except RuntimeError:
                pass
        payload = {'streamer': streamer, 'duration': round(dur, 1), 'frame_times': ftimes,
                   'transcript': ' '.join(f"[{w['s']:.1f}] {w['w']}" for w in words)[:3500]}
        import lore
        raw = J._ask(RULES + lore.prompt_block(streamer) + '\n' + json.dumps(payload, ensure_ascii=False), frames, 120, J.CLAUDE_MODELS.get(a.judge, 'sonnet'))
        plan = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
        json.dump(plan, open(pf, 'w'), indent=1, ensure_ascii=False)
    plan = json.load(open(pf))
    clamp = lambda t: min(max(0.0, float(t)), dur)
    beats = sorted((b for b in plan.get('beats', []) if 0 < float(b['t']) < dur - 0.3), key=lambda b: b['t'])
    kept = []
    for b in beats:                                   # at most one every 7 s
        if not kept or b['t'] - kept[-1]['t'] >= 7:
            kept.append(b)
    beats = kept

    # 2. the base: auto-framed with punch-ins on the beats; a wide group scene (faces spread across the frame) keeps the
    #    whole picture in the middle over a blurred fill instead, so the action isn't cropped away
    base = os.path.join(d, f'{out_id}.base.mp4')
    if a.replan or not os.path.exists(base):
        wide = a.layout == 'wide' or (a.layout != 'crop' and wide_scene(src, s0, s1))
        json.dump({'wide': bool(wide)}, open(base + '.json', 'w'))
        if wide:
            print('  wide group scene: whole frame over a blurred fill', flush=True)
            fc = ("[0:v]fps=30,split=2[a][b];[b]scale=-2:1920,crop=1080:1920,gblur=sigma=34,eq=brightness=-0.18[bg];"
                  "[a]scale=1080:-2,setsar=1[fg];[bg][fg]overlay=0:(H-h)/2-80,setsar=1,format=yuv420p[v]")
            ff('-ss', f'{s0:.3f}', '-to', f'{s1:.3f}', '-i', src, '-ss', f'{s0:.3f}', '-to', f'{s1:.3f}', '-i', aud, '-filter_complex', fc,
               '-map', '[v]', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', 18, '-c:a', 'aac', '-ar', 48000, '-ac', 2, base)
        else:
            vcam.render(FF, src, s0, s1, aud, s0, base, punches=[s0 + b['t'] for b in beats], threads=4)

    wide = json.load(open(base + '.json')).get('wide') if os.path.exists(base + '.json') else False
    CAP_Y = 1330 if wide else 1000          # the wide layout's picture spans ~576-1184: captions go under it
    # 3. the overlay script
    lines = ['[Script Info]', 'ScriptType: v4.00+', f'PlayResX: {W}', f'PlayResY: {H}', 'WrapStyle: 2', 'ScaledBorderAndShadow: yes', '',
             '[V4+ Styles]',
             'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
             'Style: Glow,TikTok Sans Wide,68,&HFF000000,&H00FFFFFF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,2,0,1,9,0,5,60,60,0,1',
             'Style: Cap,TikTok Sans Wide,68,&H00FFFFFF,&H00FFFFFF,&H00141414,&H80000000,0,0,0,0,100,100,2,0,1,2.5,3,5,60,60,0,1',
             'Style: Mark,TikTok Sans Wide,28,&H30FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,1,0,1,1.5,2,1,40,40,0,1',
             'Style: Ink,TikTok Sans Wide,10,&H002B2BFF,&H00FFFFFF,&H002B2BFF,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1',
             '', '[Events]', 'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text']
    ev = lines.append

    def glow_text(t0, t1, x, y, text_glow, text_cap, fs=68, pop=True):
        p = '\\fscx115\\fscy115\\t(0,90,\\fscx100\\fscy100)' if pop else ''
        ev(f'Dialogue: 0,{t_ass(t0)},{t_ass(t1)},Glow,,0,0,0,,{{\\pos({x},{y})\\fs{fs}\\blur14\\3a&HA0&{p}}}{text_glow}')
        ev(f'Dialogue: 1,{t_ass(t0)},{t_ass(t1)},Cap,,0,0,0,,{{\\pos({x},{y})\\fs{fs}{p}}}{text_cap}')

    kws = [(float(k['t']), re.sub(r'[^A-Z0-9]', '', str(k['word']).upper()), PAL.get(k.get('color'), PAL['yellow'])) for k in plan.get('keywords', [])]
    stage = [(clamp(s['t']), clamp(float(s['t']) + float(s.get('dur', 1.2))), J.mask(str(s['text']).upper()), PAL.get(s.get('color'), PAL['red'])) for s in plan.get('stage', [])]
    verdict_t = max(0.0, dur - 1.4)
    for c0, c1, ws in cards(words, dur):
        if any(st0 - 0.1 <= c0 <= st1 for st0, st1, _, _ in stage) or c0 >= verdict_t:
            continue
        g, c, used = [], [], False
        for w in ws:
            tok = E.censor(w['w']).upper()
            key = re.sub(r'[^A-Z0-9]', '', tok)
            col = next((cc for t, kw, cc in kws if kw == key and abs(t - w['s']) < 1.0), None) if not used else None
            if col:
                used = True
                g.append(f'{{\\3c{ass_c(col)}\\3a&H40&}}{tok}{{\\3c&H00FFFFFF&\\3a&HA0&}}'); c.append(f'{{\\c{ass_c(col)}}}{tok}{{\\c&H00FFFFFF&}}')
            else:
                g.append(tok); c.append(tok)
        n = len(ws); split = n // 2 if n >= 3 else n          # two lines for 3-4 words, the punch on line 2
        jg = ' '.join(g[:split]) + ('\\N' + ' '.join(g[split:]) if split < n else '')
        jc = ' '.join(c[:split]) + ('\\N' + ' '.join(c[split:]) if split < n else '')
        glow_text(c0, c1, 540, CAP_Y, jg, jc)
    for st0, st1, text, col in stage:
        glow_text(st0, st1, 540, CAP_Y, f'{{\\3c{ass_c(col)}\\3a&H40&}}{text}', f'{{\\c{ass_c(col)}}}{text}', fs=64)
    for cx in plan.get('context', [])[:2]:
        t0, t1 = clamp(cx['t0']), clamp(cx['t1'])
        txt = J.mask(str(cx['text']).upper())
        glow_text(t0, min(t1, t0 + 10), 540, 400, f'{{\\3c{ass_c(PAL["yellow"])}\\3a&H40&}}{txt}', f'{{\\c{ass_c(PAL["yellow"])}}}{txt}', fs=54, pop=False)

    overlays, tmpd = [], os.path.join(d, f'{out_id}.tmp'); os.makedirs(tmpd, exist_ok=True)
    pops = []
    for tg in plan.get('tags', [])[:2]:
        t0, t1 = clamp(tg['t0']), clamp(tg['t1'])
        t1 = min(max(t1, t0 + 1.5), t0 + 3.0, dur)
        box = who_box(base, t0 + 0.2, tg['name'], tg.get('who') or '', a.judge)
        if not box:
            print(f"  tag {tg['name']}: not in frame at {t0:.1f} s, skipped", flush=True); continue
        x, y = box[0] + box[2] // 2, box[1]
        x = min(max(x, 160), 920)
        ny = max(300, y - 150)
        col = PAL.get(tg.get('color'), PAL['cyan'])
        name = J.mask(str(tg['name']).upper()); sub = J.mask(str(tg.get('sub') or '').upper())
        pop = '\\fscx0\\fscy0\\t(0,100,\\fscx110\\fscy110)\\t(100,150,\\fscx100\\fscy100)'
        ev(f'Dialogue: 0,{t_ass(t0)},{t_ass(t1)},Glow,,0,0,0,,{{\\pos({x},{ny})\\fs60\\blur14\\3c{ass_c(col)}\\3a&H40&{pop}}}{name}')
        ev(f'Dialogue: 1,{t_ass(t0)},{t_ass(t1)},Cap,,0,0,0,,{{\\pos({x},{ny})\\fs60\\c{ass_c(col)}{pop}}}{name}')
        py = ny + 42
        if sub:
            ev(f'Dialogue: 1,{t_ass(t0)},{t_ass(t1)},Cap,,0,0,0,,{{\\pos({x},{ny + 52})\\fs32{pop}}}{sub}'); py = ny + 80
        ev(f'Dialogue: 1,{t_ass(t0)},{t_ass(t1)},Ink,,0,0,0,,{{\\an7\\pos({x - 18},{py})\\p1}}m 0 0 l 36 0 l 18 26{{\\p0}}')
        pops.append(t0)
    for k, sk in enumerate(plan.get('stickers', [])[:3]):
        t0 = clamp(sk['t']); t1 = min(dur, t0 + 1.6)
        png = os.path.join(tmpd, f'st{k}.png')
        try:
            ew, eh = emoji_png(sk['emoji'], png, 170)
        except Exception:  # noqa: BLE001
            continue
        box = who_box(base, t0 + 0.1, 'the person', sk.get('who') or 'the main person', a.judge)
        if box:
            bx, by, bw, bh = box
            x, y = {'beside': (bx + bw + 10, by), 'chest': (bx + bw // 2 - ew // 2, by + bh + 60)}.get(sk.get('anchor'), (bx + bw // 2 - ew // 2, by - eh - 20))
        else:
            x, y = W // 2 - ew // 2, 560
        overlays.append((png, int(min(max(x, 20), W - ew - 20)), int(min(max(y, 260), 1500)), t0, t1))
        pops.append(t0)
    vd = plan.get('verdict') or {}
    if vd.get('text'):
        vt = J.mask(str(vd['text']).upper())
        glow_text(verdict_t, dur + 0.5, 540, CAP_Y, f'{{\\3c{ass_c(PAL["yellow"])}\\3a&H40&}}{vt}', f'{{\\c{ass_c(PAL["yellow"])}}}{vt}', fs=72)
        if vd.get('emoji'):
            png = os.path.join(tmpd, 'verdict.png')
            try:
                ew, eh = emoji_png(vd['emoji'], png, 150)
                overlays.append((png, W // 2 - ew // 2, 1100, verdict_t, dur + 0.5))
            except Exception:  # noqa: BLE001
                pass
    total = dur + 1.5
    ev(f'Dialogue: 2,{t_ass(0)},{t_ass(total)},Mark,,0,0,0,,{{\\an1\\pos(48,1560)}}@{handle.upper()}')
    ev(f'Dialogue: 2,{t_ass(0)},{t_ass(total)},Mark,,0,0,0,,{{\\an1\\pos(48,1598)\\fs22}}{(credit or "").upper()}')
    ass = os.path.join(tmpd, 'hype.ass')
    open(ass, 'w').write('\n'.join(lines) + '\n')

    # 4. color beats: white pre-flash, the tint (or black and white), a short shake
    vf = ['eq=contrast=1.12:brightness=0.03:saturation=1.12', 'vignette=PI/6']
    for b in beats:
        t = float(b['t']); look = b.get('look', 'yellow')
        vf.append(f"eq=brightness=0.45:enable='between(t,{t - 0.05:.2f},{t:.2f})'")
        if look == 'bw':
            vf.append(f"hue=s=0:enable='between(t,{t:.2f},{t + 0.4:.2f})'")
            vf.append(f"eq=contrast=1.5:enable='between(t,{t:.2f},{t + 0.4:.2f})'")
        else:
            r, g, bl = TINT.get(look, TINT['yellow'])
            vf.append(f"colorchannelmixer=rr={r}:gg={g}:bb={bl}:enable='between(t,{t:.2f},{t + 0.4:.2f})'")
    shake = '+'.join(f"14*sin((t-{b['t']:.2f})*70)*exp(-(t-{b['t']:.2f})*12)*between(t,{b['t']:.2f},{float(b['t']) + 0.3:.2f})" for b in beats) or '0'
    vf += [f"crop={W - 28}:{H - 28}:x='14+({shake})':y='14+({shake})*0.6'", f'scale={W}:{H}',
           f'tpad=stop_mode=clone:stop_duration=0.5,tpad=stop_mode=add:stop_duration=1.0:color=black',
           f'fade=t=out:st={dur:.2f}:d=0.5']

    # 5. sound effects
    sounds = []
    for name in ('boom', 'whoosh', 'pop'):
        sfx(name, os.path.join(tmpd, f'{name}.wav'))
    for b in beats:
        sounds.append(('boom', float(b['t'])))
        sounds.append(('whoosh', max(0.0, float(b['t']) - 0.12)))
    for t in pops:
        sounds.append(('pop', t))

    # 6. render
    out = os.path.join(d, f'{out_id}.mp4')
    ins = ['-i', base]
    for png, *_ in overlays:
        ins += ['-loop', 1, '-i', png]
    for name, _ in sounds:
        ins += ['-i', os.path.join(tmpd, f'{name}.wav')]
    chain = [f"[0:v]{','.join(vf)}[g]"]
    last = 'g'
    for k, (png, x, y, t0, t1) in enumerate(overlays):
        chain.append(f"[{k + 1}:v]format=rgba,fade=t=in:st={t0:.2f}:d=0.12:alpha=1[s{k}];[{last}][s{k}]overlay={x}:{y}:shortest=1:enable='between(t,{t0:.2f},{t1:.2f})'[o{k}]")
        last = f'o{k}'
    chain.append(f"[{last}]ass={ass}:fontsdir={FONTS}[v]")
    base_a = f"[0:a]apad=pad_dur=1.5,afade=t=out:st={dur:.2f}:d=0.6[a0]"
    chain.append(base_a)
    mix = ['[a0]']
    n0 = 1 + len(overlays)
    for k, (name, t) in enumerate(sounds):
        vol = {'boom': 0.55, 'whoosh': 0.3, 'pop': 0.35}[name]
        chain.append(f"[{n0 + k}:a]volume={vol},adelay={int(t * 1000)}|{int(t * 1000)},aresample={SR}[x{k}]")
        mix.append(f'[x{k}]')
    chain.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first,loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    ff(*ins, '-filter_complex', ';'.join(chain), '-map', '[v]', '-map', '[a]', '-t', f'{total:.2f}', '-r', FPS,
       '-c:v', 'libx264', '-preset', 'slow', '-crf', 19, '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-ar', SR, '-movflags', '+faststart', out)
    ff('-ss', f'{dur * 0.33:.2f}', '-i', out, '-frames:v', 1, '-q:v', 3, os.path.join(d, f'{out_id}.jpg'))
    json.dump({'id': out_id, 'title': plan.get('title'), 'kind': plan.get('kind'), 'duration': round(total, 1), 'handle': handle, 'credit': credit,
               'beats': len(beats), 'tags': [t['name'] for t in plan.get('tags', [])[:2]], 'stage': [s[2] for s in stage],
               'verdict': vd.get('text'), 'post_caption': f"{plan.get('title', '')} #{re.sub(r'[^a-z0-9]', '', (streamer or '').lower())} #edit"},
              open(os.path.join(d, f'{out_id}.json'), 'w'), indent=1, ensure_ascii=False)
    print(f'{out}: "{plan.get("title")}" · {len(beats)} beats · tags {[t["name"] for t in plan.get("tags", [])[:2]]} · verdict {vd.get("text")}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip')
    ap.add_argument('--src'); ap.add_argument('--audio'); ap.add_argument('--words'); ap.add_argument('--span', nargs=2, type=float)
    ap.add_argument('--id'); ap.add_argument('--out'); ap.add_argument('--streamer'); ap.add_argument('--credit')
    ap.add_argument('--handle')
    ap.add_argument('--start', type=float); ap.add_argument('--end', type=float)
    ap.add_argument('--judge', default='sonnet')
    ap.add_argument('--replan', action='store_true')
    ap.add_argument('--layout', choices=['auto', 'wide', 'crop'], default='auto', help='wide: whole frame over a blurred fill; crop: follow the subject')
    main(ap.parse_args())
