#!/usr/bin/env python3
"""hypecut: one streamer moment -> a clip-page hype edit, planned by the editor (Claude) and rendered by studio.py.

The editor reads the moment (word-timed transcript + frame sheets with face counts) and writes a plan: the tease (the
funniest visual instant, shown first in slow motion, then a rewind), the story as beats (talk beats at normal speed with
word-pop captions, split screen when two people trade lines; action beats that go into slow motion with pitched-down
sound), punch words, a verdict, the punchline word where the music drops, and the reactions for the montage after the
drop. This file turns that plan into a studio EDL: music muffled under the talking and full from its drop, cuts on the
beat after it, transitions every cut in the montage, SFX on hits, a credit mark the whole time.

usage: hypecut.py SRC.mp4 OUT_DIR [--a 10 --b 50] [--streamer N3on] [--credit kick.com/n3on] [--handle cutroomn3on]
                  [--song montagem-tenta] [--plan plan.json (skip the editor)] [--workers 6]
Writes OUT_DIR/<id>.mp4 (with music, for TikTok), <id>-site.mp4 (no music), <id>.json, plan.json, edit.json.
"""
import argparse, io, json, math, os, re, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
ROOT = os.path.dirname(PIPE)
sys.path.insert(0, HERE); sys.path.insert(0, PIPE); sys.path.insert(0, os.path.join(PIPE, 'live'))
import studio  # noqa: E402

FF = studio.FF
MUSIC = os.path.join(ROOT, 'portfolio', 'music')
COLORS = {'red': [255, 128, 104], 'yellow': [255, 230, 60], 'green': [110, 255, 120], 'cyan': [80, 220, 255], 'pink': [255, 140, 235], 'white': [255, 255, 255]}   # bright enough to read on a dark frame

RULES = """You edit a TikTok clip page that turns streamer moments into over-the-top hype edits (the IShowSpeed / N3on
clip-page style: word-pop captions, stage directions, slow-motion punch-ins, a phonk drop on the punchline, a fast montage
after it). Plan ONE edit of this moment. The images are frame sheets: each frame is labelled with its source time and how
many faces are visible.

The edit, in order:
1. TEASE: the single funniest visual instant (the edit ends on it; with a rewind opener it also opens it).
2. BEATS: the story, 3-8 beats in source order, cutting dead air, repeats and filler. Keep the setup short. Before the drop
   the beats should add up to 12-22 s of edit time (an action beat's slow part plays at 0.45x, so it takes 2.2x longer).
   The first beat's first frame is the thumbnail and the first impression: start it where the people are clearly in
   frame (not a back, a blur, a dark or empty frame), even if that skips a word of preamble.
   - "talk": dialogue at normal speed, captioned. Start and end on word boundaries (use the word times).
     layout: "split" when two people trade lines and both matter (top/bottom split screen), "left"/"right" for the
     person talking (their side of the frame), "both" when two people are close together, "react" when the streamer
     talks in the foreground while the thing happens behind or beside them (the action on top, their face below),
     "center" otherwise.
   - "action": something physical happens (a sniff, a fall, a slap of the table, a dance). Normal speed until slow_from,
     then slow motion until b; b - slow_from is at most 1.6 s. focus: left/right/both/center on faces; "action" = the
     crop follows the movement itself (use it when the thing happening is away from the biggest face, e.g. people in the
     background); "react" = the action on top and the streamer's reacting face below. LOOK at the frames: the edit must
     SHOW the thing that happens, not only the person watching it. stage: a stage direction in
     asterisks for it, e.g. *SNIFFING*, 1-3 words.
3. PUNCH: up to 3 words to hit with a zoom punch and a boom (the funniest words), by their start time.
4. VERDICT (optional): a short line someone says that sums it up, by start time, shown big, e.g. SMELLS LIKE A ROSE.
5. DROP: the punchline word where the music drops, by its start time. It must be inside the last beat, and that beat ends
   there. Everything after it is the montage.
6. MONTAGE: 6-8 source times of the best reactions and actions (faces, motion; never the same instant twice), each with a
   focus (left/right/both/center, or action/react as above), for a beat-cut montage after the drop.

Vibe (picks the sound): "hype" = chaos, a big win, a crowd going off; "aura" = someone cold and unbothered, a clean skill
shot, a composed flex; "funny" = a goofy bit, an awkward moment, a fail played for laughs.

Text: title = the hook, max 8 words, what happens, ending in two identical emojis. Name the streamer only when the frames
make clear which person is them (they are usually the one the camera follows); a name someone shouts ("Druski, help!") is
the person being called, not the one it happens to; otherwise say He / She / Bro (e.g. "Ain't NO WAY He
Smelled Her Armpit 😭😭"). keywords = up to 8 words that appear in the transcript, each with a colour (red, yellow, green,
cyan, pink). drop_text = 1-3 words for the drop (usually the punchline word) plus an optional emoji. Every word you put on
screen must be said in the transcript or visible in the frames; describe, don't invent. No slurs, no sexual framing.

Answer with JSON only:
{"title": "...", "tease": {"t": sec, "focus": "left|right|both|center|action|react"},
 "beats": [{"a": sec, "b": sec, "type": "talk", "layout": "split|left|right|both|react|center"},
           {"a": sec, "b": sec, "type": "action", "slow_from": sec, "focus": "both|action|react|...", "stage": "*SNIFFING*"}],
 "punch": [sec], "verdict": {"t": sec, "text": "SMELLS LIKE A ROSE", "emoji": "🌹"} or null,
 "drop": {"t": sec, "text": "SECRET", "emoji": "🤫"},
 "montage": [{"t": sec, "focus": "left|right|both|center|action|react"}],
 "keywords": {"WORD": "red"}, "caption": "the post caption with 2-4 hashtags", "vibe": "hype|aura|funny",
 "fix": {"Heard": "Name"} (a name the transcript misheard, e.g. Juski for Druski, only when the context makes you sure; {} if none)}"""


def video_end(src, b):
    """Where the picture really ends (stream captures often carry audio past the last video frame)."""
    t = b
    while t > 1:
        n = len(subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-ss', f'{max(0, t - 0.6):.2f}', '-i', src, '-t', '0.5', '-f', 'rawvideo',
                                '-pix_fmt', 'gray', '-s', '32x18', '-'], capture_output=True).stdout)
        if n:
            return round(t, 2)
        t -= 0.5
    return b


def hear(src, a, b, cache):
    if os.path.exists(cache):
        return json.load(open(cache))
    import rehear
    W, t = [], a
    while t < b - 0.3:
        e = min(b, t + 30)
        W += rehear.hear_safe(src, t, e) or []
        t = e
    json.dump(W, open(cache, 'w'))
    return W


def sheets(src, a, b, step=1.0):
    """Frame sheets (6 frames each, labelled with source time and face count) as JPEG bytes, plus the face counts."""
    import cv2, numpy as np
    from PIL import Image, ImageDraw
    cap = cv2.VideoCapture(src)
    det = None
    frames = []
    t = a
    while t <= b:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            sm = cv2.resize(fr, (640, int(h * 640 / w)))
            if det is None:
                det = cv2.FaceDetectorYN.create(studio.YUNET, '', (sm.shape[1], sm.shape[0]), 0.6, 0.3, 50)
            _, f = det.detect(sm)
            frames.append((round(t, 1), 0 if f is None else len(f), cv2.cvtColor(cv2.resize(fr, (480, 270)), cv2.COLOR_BGR2RGB)))
        t += step
    out = []
    for k in range(0, len(frames), 6):
        sh = Image.new('RGB', (480 * 3, 270 * 2))
        for j, (tt, nf, im) in enumerate(frames[k:k + 6]):
            p = Image.fromarray(im); d = ImageDraw.Draw(p)
            d.rectangle((0, 0, 200, 26), fill=(0, 0, 0)); d.text((6, 5), f't={tt:.1f}s  faces {nf}', fill=(255, 230, 60))
            sh.paste(p, ((j % 3) * 480, (j // 3) * 270))
        bio = io.BytesIO(); sh.save(bio, 'JPEG', quality=80); out.append(bio.getvalue())
    return out, {f[0]: f[1] for f in frames}


def transcript_lines(W):
    """Words with start times, a line per pause, for the prompt."""
    lines, cur = [], []
    for w in W:
        if cur and (w['s'] - cur[-1]['s'] > 0.9 or len(cur) >= 12):
            lines.append(cur); cur = []
        cur.append(w)
    if cur:
        lines.append(cur)
    return '\n'.join(' '.join(f"{w['w']}@{w['s']:.2f}" for w in ln) for ln in lines)


def ask_plan(src, a, b, W, streamer, title_hint=''):
    import judge as J
    imgs, faces = sheets(src, a, b)
    prompt = (RULES + f"\n\nSTREAMER: {streamer}\nSOURCE SPAN: {a:.1f}-{b:.1f} s" + (f"\nWHAT CHAT CALLED IT: {title_hint}" if title_hint else '')
              + "\nTRANSCRIPT (word@start seconds):\n" + transcript_lines(W))
    try:
        import lore
        prompt += lore.prompt_block(streamer)
    except Exception:  # noqa: BLE001
        pass
    last = None
    for attempt in range(2):
        raw = J._ask(prompt + (f"\n\nYOUR LAST ANSWER HAD A PROBLEM: {last}. Answer again with fixed JSON." if last else ''), imgs[:8], timeout=150)
        m = re.search(r'\{.*\}', raw, re.S)
        try:
            plan = json.loads(m.group(0)) if m else None
            problem = check(plan, a, b, W) if plan else 'no JSON'
        except (ValueError, KeyError, TypeError) as e:
            plan, problem = None, f'invalid: {e}'
        if plan and not problem:
            return plan
        last = problem
    raise SystemExit(f'editor plan unusable: {last}')


def check(p, a, b, W):
    bs = p.get('beats') or []
    if not 2 <= len(bs) <= 9:
        return 'need 3-8 beats'
    for x in bs:
        if not (a - 1 <= x['a'] < x['b'] <= b + 1):
            return f"beat {x} is outside the source span"
        if x['type'] == 'action' and not (x['a'] <= x.get('slow_from', x['a']) < x['b']):
            return f"action beat {x} needs a <= slow_from < b"
    if any(y['a'] < x['b'] - 0.05 for x, y in zip(bs, bs[1:])):
        return 'beats overlap or are out of order'
    d = p.get('drop') or {}
    if not (bs[-1]['a'] <= d.get('t', -1) <= bs[-1]['b'] + 0.3) or bs[-1]['type'] != 'talk':
        return 'the drop word must be inside the last beat, which must be a talk beat'
    if len(p.get('montage') or []) < 4:
        return 'need 6-8 montage moments'
    return ''


# ---------------------------------------------------------------- plan -> EDL
FOCUS = {'left': 'auto:left', 'right': 'auto:right', 'both': 'auto:both', 'center': 'auto', 'action': 'motion', 'react': 'react'}


def pick_song(mood, key=''):
    """The bed for a mood: the popular TikTok sounds first (the user's call, 2026-10-09: no royalty-free music), the same
    one for the same clip (key), the others taking turns across clips."""
    import zlib
    lib = json.load(open(os.path.join(MUSIC, 'library.json')))
    fit = [x for x in lib if mood in [m.strip() for m in x.get('mood', '').split(',')]]
    pool = [x for x in fit if x.get('kind') == 'tiktok-popular'] or fit or [x for x in lib if 'hype' in x.get('mood', '')]
    return pool[zlib.crc32(key.encode()) % len(pool)]['name']


def best_drop(S, build):
    """The song's drop for a build of `build` seconds: the payoff lands on it, and the stretch of song under the build (the
    muffled bed) has no long hole. A song's quiet break right before its first drop left a 3.5 s near-silence under Kai's
    slow-motion lift (BLUE WATER CANDY at 0:18-0:22); a short silence just before the drop is fine (the classic pre-drop
    gap). Among clean ones, the earliest. Falls back to the first drop at or after the build."""
    raw = subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-i', S['path'], '-ac', '1', '-ar', '4000', '-f', 's16le', '-'], capture_output=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    win = 1000                                          # 0.25 s
    db = 20 * np.log10(np.sqrt((x[: len(x) // win * win].reshape(-1, win) ** 2).mean(1)) + 1e-6)
    ref = np.percentile(db, 75)
    best = None
    for d in S['drops']:
        if d < build:
            continue
        seg = db[int((d - build) * 4): int(max(d - build, d - 0.8) * 4)]
        hole = float(np.mean(seg < ref - 14)) if len(seg) else 1.0
        if best is None or hole < best[0] - 0.05:
            best = (hole, d)
    return best[1] if best else S['drops'][-1]


def song(name):
    lib = json.load(open(os.path.join(MUSIC, 'library.json')))
    s = next(x for x in lib if x['name'] == name)
    return {**s, 'path': os.path.join(MUSIC, s['file']), 'beats': json.load(open(os.path.join(MUSIC, s['beats'])))}


def overlay_layout(src, a, b, game=False):
    """A facecam over gameplay (or a browser): the cam box and the content crop, normalised, else None. game=True: the
    gaming layout (pipeline/gamecam.py): the game on top at near full height, the streamer's cam below."""
    if game:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        import gamecam
        cs = gamecam.cams(src, a, b)
        if cs:
            Wd, Hd = cs[0]['src']; cam_h = 910
            g = gamecam.game_crop(cs, Wd, Hd, 1080, 1920 - cam_h)
            cc = gamecam.cam_crop(cs[0], Wd, Hd, 1080, cam_h)
            return {'cam': [cc['x'] / Wd, cc['y'] / Hd, cc['w'] / Wd, cc['h'] / Hd], 'screen': [g['x'] / Wd, g['y'] / Hd, g['w'] / Wd, g['h'] / Hd],
                    'order': 'game_top', 'cam_frac': cam_h / 1920}
    import facecam
    try:
        cam = facecam.detect(src, a, b)
    except Exception:  # noqa: BLE001
        return None
    if not cam:
        return None
    Wd, Hd = cam['src']
    sc = cam['screen']; sw = round(sc['h'] * 1080 / 1312)
    return {'cam': [cam['x'] / Wd, cam['y'] / Hd, cam['w'] / Wd, cam['h'] / Hd], 'screen': [sc['x'] / Wd, sc['y'] / Hd, sw / Wd, sc['h'] / Hd]}


def build(p, src, W, credit, handle, song_name='montagem-tenta', layout=None):
    shots, voice, texts, trans, hits, sfx = [], [], [], [], [], []
    import cv2
    _cap = cv2.VideoCapture(src)

    def content_dark(t_src):
        """Is the overlay layout's content area (the game or page) blank at this source time (screen share off)?"""
        if not layout:
            return False
        _cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t_src) * 1000); ok, fr = _cap.read()
        if not ok:
            return False
        h, w = fr.shape[:2]; x, y, cw, ch = layout['screen']
        reg = fr[int(y * h): int((y + ch) * h), int(x * w): int((x + cw) * w)]
        return reg.size > 0 and float(reg.mean()) < 14 and float(reg.std()) < 12
    key = {k.upper().strip('.,!?'): COLORS.get(v, COLORS['yellow']) for k, v in (p.get('keywords') or {}).items()}
    t = 0.0
    ZIN = lambda x, y: [[0, x, 0, 0, 0], [1, y, 0, 0, 0]]
    tmap = []                                   # (src a, src b, edit at, speed) for 1x spans, to place punches/verdict/drop

    def caption(a, b, at, y):
        ws = [w for w in W if a - 0.05 <= w['s'] < b - 0.1]
        groups, cur = [], []
        for w in ws:
            if cur and w['s'] - cur[-1]['s'] > 0.55:
                groups.append(cur); cur = []
            cur.append(w)
            if len(cur) == 2 or w['w'][-1:] in '.,?!' or w['w'].strip('.,!?').upper() in key:
                groups.append(cur); cur = []
        if cur:
            groups.append(cur)
        FILLER = {'MAN', 'GOD', 'DUDE', 'BRO', 'YO', 'BRUH', 'OH', 'AH', 'UH', 'UM', 'LIKE', 'YEAH', 'OKAY', 'OK', 'WHAT', 'HUH', 'NAH', 'LOOK'}
        groups = [g for g in groups if not (len(g) == 1 and g[0]['w'].strip('.,!?').upper() in FILLER and g[0]['w'].strip('.,!?').upper() not in key)]
        for k, g in enumerate(groups):
            t0 = at + g[0]['s'] - a
            t1 = at + (groups[k + 1][0]['s'] - a if k + 1 < len(groups) else min(b - a, g[-1]['e'] - a + 0.25))
            col = next((key[w['w'].strip('.,!?').upper()] for w in g if w['w'].strip('.,!?').upper() in key), COLORS['white'])
            from storyedit import mask
            fx = {k.lower(): v for k, v in (p.get('fix') or {}).items() if isinstance(v, str) and len(k) >= 3}   # misheard names
            said = ' '.join(fx.get(w['w'].strip('.,!?').lower(), w['w'].strip('.,!')) for w in g)
            texts.append({'t0': round(t0, 3), 't1': round(max(t1, t0 + 0.25), 3), 'text': mask(said), 'said': True,
                          'style': 'hype', 'anim': 'pop', 'y': y, 'color': col})

    def add(i, dur, focus='auto', speed=1.0, cam=None, look=None, split=False, close=False):
        """close=True: a reaction (with an overlay layout, the facecam box alone); otherwise, with an overlay layout,
        the cam on top and the game or page below."""
        nonlocal t
        if layout and not close and content_dark(i + max(0.0, dur) / 2):
            close = True                                # nothing on the shared screen right now: the cam alone
        sh = {'src': src, 'in': round(i, 3), 'at': round(t, 3), 'dur': round(dur, 3), 'speed': speed, 'focus': focus,
              'cam': cam or ZIN(1.0, 1.06), 'look': look or [], **({'split': True, 'split_zoom': 1.8} if split and not layout else {})}
        if sh['focus'] == 'react':                      # the action on top, the reactor's face below (studio motion_track)
            sh['focus'] = 'auto'
            if not layout:
                sh.update(react=True, cam=ZIN(1.0, 1.08))
        if layout and close:
            sh['region'] = layout['cam']; sh['focus'] = 'auto'
        elif layout:
            sh['camsplit'] = layout; sh['cam'] = ZIN(1.0, 1.0 if layout.get('order') == 'game_top' else 1.08)   # a game isn't zoomed further
        shots.append(sh)
        t += dur
        return shots[-1]
    SEAM = (round(1920 * (1 - layout['cam_frac'])) if layout.get('order') == 'game_top' else round(1920 * 0.317)) if layout else None

    def stage(t0, t1, txt, col=COLORS['cyan'], y=420, style='stage'):
        texts.append({'t0': round(t0, 3), 't1': round(t1, 3), 'text': txt, 'style': style, 'anim': 'pop', 'y': y, 'color': col})

    # the opener, varied per clip so a page doesn't open every post the same way:
    #   rewind: the funniest instant in slow motion with the title, then a rewind into the setup
    #   freeze: the instant at speed, a freeze frame with WAIT FOR IT, a hard cut into the setup
    #   cold:   straight into the setup, the title popping over its first seconds
    import zlib
    # cold open on the action by default (what nearly every top reference does; the tease openers that start on the payoff and
    # jump back confused strangers in our critic's reads). A plan can still ask for 'rewind' or 'freeze'.
    opener = p.get('opener') or os.environ.get('CUTROOM_OPENER') or 'cold'
    te = p['tease']['t']; tf = FOCUS.get(p['tease'].get('focus'), 'auto:both')
    if opener == 'rewind':
        add(te, 1.3, focus=tf, speed=0.55, cam=ZIN(1.15, 1.45), close=True)
        voice.append({'src': src, 'in': te, 'out': te + 0.72, 'at': 0.0, 'rate': 0.55, 'duck': 0.5})
        stage(0.0, 1.7, p['title'], COLORS['white'], 300, 'hook')
        hits += [{'at': 0.0, 'type': 'flash', 'amt': 0.8}, {'at': 0.4, 'type': 'shake', 'amt': 1.2}]
        sfx.append({'at': 0.02, 'type': 'boom', 'gain': 0.6})
        add(te + 0.72, 0.4, focus=tf, speed=-3.5, cam=ZIN(1.45, 1.1), look=['rgbsplit'], close=True)
        trans.append({'at': 1.3, 'type': 'glitch', 'dur': 0.3}); sfx.append({'at': 1.5, 'type': 'whoosh', 'gain': 0.45, 'dur': 0.4})
        stage(1.32, 1.7, '*REWIND*', COLORS['white'], 960)
    elif opener == 'freeze':
        add(te - 0.5, 0.7, focus=tf, speed=1.0, cam=ZIN(1.1, 1.3), close=True)
        voice.append({'src': src, 'in': te - 0.5, 'out': te + 0.2, 'at': 0.0, 'duck': 0.5})
        add(te + 0.2, 1.0, focus=tf, speed=0.0, cam=ZIN(1.3, 1.42), look=['bw'], close=True)       # the freeze frame
        stage(0.0, 1.7, p['title'], COLORS['white'], 300, 'hook')
        stage(0.72, 1.68, 'WAIT FOR IT 😭', COLORS['yellow'], 1240, 'hype')
        hits += [{'at': 0.7, 'type': 'flash', 'amt': 1.0}, {'at': 0.7, 'type': 'punch', 'amt': 1.5}]
        sfx += [{'at': 0.7, 'type': 'boom', 'gain': 0.55}]
        trans.append({'at': 1.7, 'type': 'flash', 'dur': 0.2})
    else:
        stage(0.0, 2.2, p['title'], COLORS['white'], 300, 'hook')
        sfx.append({'at': 0.02, 'type': 'whoosh', 'gain': 0.4, 'dur': 0.4})
        hits.append({'at': 0.0, 'type': 'punch', 'amt': 1.2})

    beats = p['beats']
    drop_src = p['drop']['t']
    cyc = ['flash', 'zoom', 'whip']
    DROP = None
    for k, bt in enumerate(beats):
        last = k == len(beats) - 1
        a, b = bt['a'], bt['b']
        if last:
            b = drop_src
        start = t
        if bt['type'] == 'talk':
            lay = bt.get('layout', 'center')
            split = lay == 'split'
            s = add(a, b - a, focus=FOCUS.get(lay, 'auto'), cam=None if split else ZIN(1.18, 1.28), split=split)
            voice.append({'src': src, 'in': a, 'out': b + (0.35 if last else 0), 'at': s['at'], 'duck': 0.22})
            caption(a, b + (0.4 if last else 0), s['at'], SEAM or (960 if split else 1190 if s.get('react') else 1240))
            tmap.append((a, b, s['at']))
        else:
            sf = bt.get('slow_from', a); sf = min(max(a, sf), b - 0.2)
            b = min(b, sf + 1.6)
            d1 = sf - a; d2 = (b - sf) / 0.45
            sp = [[0, 1.0], [max(0.01, d1 / (d1 + d2)), 1.0], [min(0.99, (d1 + 0.25) / (d1 + d2)), 0.45], [1, 0.45]] if d1 > 0.05 else 0.45
            s = add(a, d1 + d2, focus=FOCUS.get(bt.get('focus'), 'auto:both'), speed=sp, cam=ZIN(1.1, 1.55))
            if d1 > 0.05:
                voice.append({'src': src, 'in': a, 'out': sf, 'at': s['at'], 'duck': 0.22}); tmap.append((a, sf, s['at']))
            voice.append({'src': src, 'in': sf, 'out': b, 'at': round(s['at'] + d1, 3), 'rate': 0.45, 'duck': 0.5})
            if bt.get('stage'):
                stage(s['at'] + d1, s['at'] + d1 + d2 - 0.05, bt['stage'])
            hits.append({'at': s['at'] + d1 + 0.3, 'type': 'shake', 'amt': 1.0}); sfx.append({'at': s['at'] + d1, 'type': 'whoosh', 'gain': 0.35})
        if k == 0:
            trans.append({'at': start, 'type': 'flash', 'dur': 0.2})
        elif not (bt['type'] == 'talk' and beats[k - 1]['type'] == 'talk' and bt.get('layout') == beats[k - 1].get('layout')):
            trans.append({'at': start, 'type': cyc[k % 3], 'dur': 0.3, **({'dir': 'left'} if cyc[k % 3] == 'whip' else {})})
        if last:
            DROP = round(t, 3)

    def at_edit(ts):
        for a, b, at in tmap:
            if a - 0.05 <= ts <= b + 0.05:
                return at + ts - a
        return None

    for pt in (p.get('punch') or [])[:3]:
        e = at_edit(pt)
        if e is not None and e < DROP - 0.2:
            hits += [{'at': e, 'type': 'punch', 'amt': 2.2}, {'at': e, 'type': 'shake', 'amt': 0.8}]
            sfx.append({'at': e, 'type': 'boom', 'gain': 0.5})
    vd = p.get('verdict') or None
    if vd and at_edit(vd['t']) is not None:
        e = at_edit(vd['t'])
        stage(e - 0.2, min(DROP, e + 1.6), f"{vd['text']} {vd.get('emoji') or ''}".strip(), COLORS['green'], 500, 'hype')
        for x in texts:                                 # the verdict IS the line: no pop captions of it underneath (said twice)
            if x.get('said') and x['t0'] < min(DROP, e + 1.6) and x['t1'] > e - 0.2:
                x['t1'] = x['t0']
        hits.append({'at': e - 0.2, 'type': 'pulse', 'amt': 1.4})

    # music: the drop that fits (the song has to have started by edit time 0)
    S = song(song_name)
    drop_song = best_drop(S, DROP)
    mstart = round(drop_song - DROP, 3)
    mat = 0.0
    if mstart < 0:
        mat, mstart = -mstart, 0.0
    beats_e = [round(x - mstart + mat, 3) for x in S['beats'] if x >= drop_song - 0.02][:20]

    t = DROP
    VEL = [[0, 1.8], [0.35, 0.4], [0.75, 0.4], [1, 1.9]]
    mont = (p.get('montage') or [])[:8]
    if p.get('punch'):                                  # the key hit again, slowed, under INSTANT REPLAY (the 323M-view edit's move)
        mont = [{'t': p['punch'][0] - 0.9, 'replay': True, 'focus': (p.get('tease') or {}).get('focus')}] + mont[:7]
    # end on the funniest instant (the plan's tease), not on whatever reaction came last in source order: the critic kept
    # reading our endings as "a static face, no payoff"
    te_ = (p.get('tease') or {}).get('t')
    if te_ is not None and len(mont) >= 3:
        mont = [m for m in mont if m.get('replay') or abs(m['t'] - te_) > 1.0][:7] + [{'t': te_, 'focus': p['tease'].get('focus') or 'both', 'last': True}]
    looks = [['cold'], [], ['warm'], []]
    for k, m in enumerate(mont):
        a, b = beats_e[2 * k], beats_e[2 * k + 2]
        if m.get('replay'):
            # crisp, no ghost trails (the critic read them as a degraded source)
            add(max(0, m['t']), b - a, focus=FOCUS.get(m.get('focus'), 'auto:both'), speed=0.5, cam=ZIN(1.1, 1.3), close=True)
            if not p['drop'].get('text'):                   # the drop text already sits over this slot: one label at a time
                stage(a + 0.05, b - 0.05, '*INSTANT REPLAY*', COLORS['white'], 300, 'stage')
            continue
        if m.get('last'):                               # the ending: into the funniest instant at speed, then slow motion ON it
            add(max(0, m['t'] - 0.6), b - a, focus=FOCUS.get(m.get('focus'), 'auto:both'), speed=[[0, 1.0], [0.35, 1.0], [0.5, 0.45], [1, 0.45]],
                cam=ZIN(1.1, 1.35), close=bool(layout))  # (it started on the instant and sped past it: the dunk ended on an empty rim)
            trans.append({'at': a, 'type': 'flash', 'dur': 0.2})
            continue
        add(max(0, m['t'] - 0.4), b - a, focus=FOCUS.get(m.get('focus'), 'auto'), speed=VEL, cam=ZIN(1.15, 1.35) if k % 2 == 0 else ZIN(1.35, 1.15),
            look=looks[k % 4], close=bool(layout) and k % 2 == 0)
        if k % 2 == 0:                                  # a transition on every other cut, hard cuts between: 12 smears in 20 s
            ty_ = ['flash', 'whip', 'spin', 'auto'][(k // 2) % 4]          # one spin per edit, the rest flash or motion-picked
            trans.append({'at': a, 'type': ty_, 'dur': 0.3 if ty_ == 'spin' else 0.2, **({'dir': 'left'} if ty_ == 'whip' else {})})
        else:                                           # left ~18% of the runtime blurred (the critic kept sampling "junk" frames)
            hits.append({'at': a, 'type': 'punch', 'amt': 1.0})
        hits.append({'at': beats_e[2 * k + 1] - 1 / 30, 'type': 'pulse', 'amt': 1.0})
    endb = 2 * len(mont)
    hits.append({'at': DROP, 'type': 'shake', 'amt': 1.5}); sfx.append({'at': DROP, 'type': 'boom', 'gain': 0.6})
    dt = f"{p['drop'].get('text', '')} {p['drop'].get('emoji') or ''}".strip()
    if dt:
        stage(DROP, beats_e[4], dt, COLORS['pink'], 960, 'hype')
        for x in texts:                                 # spoken pop captions stop where the drop text starts (they overlapped it)
            if x.get('said') and x['t0'] < DROP < x['t1']:
                x['t1'] = round(DROP - 0.03, 3)
            elif x.get('said') and DROP - 0.05 <= x['t0'] < beats_e[4]:     # the drop word itself: the drop text says it
                x['t1'] = x['t0']
    if opener == 'cold':                                # and wait for the title in a cold open: one text at a time
        for x in texts:
            if x.get('said') and x['t0'] < 2.2:
                x['t0'] = 2.2
    texts[:] = [x for x in texts if x['t1'] - x['t0'] >= 0.15]
    if vd:
        stage(beats_e[min(10, endb - 4)], beats_e[endb] - 0.05, f"{vd['text']} {vd.get('emoji') or ''}".strip(), COLORS['green'],
              SEAM - 330 if layout and layout.get('order') == 'game_top' else 1240, 'hype')     # a game layout: over the game, off his face
    END = beats_e[endb] + 0.6                           # end on the last shot, held, not a black card (the top edits end on the payoff)
    shots[-1]['dur'] = round(END - shots[-1]['at'], 3)
    if os.environ.get('CUTROOM_ONSCREEN_CREDIT') == '1':   # no watermark by default (the user, 2026-10-21: not a program rule)
        texts.append({'t0': 0.0, 't1': round(END, 3), 'text': f'{credit} · @{handle}', 'style': 'mark', 'anim': 'none', 'y': 1855, 'opacity': 0.75})
    if os.environ.get('CUTROOM_RETRO_FX') != '1':
        # no pitched-down slow-motion voices and no synthesized booms/whooshes: they read as cheap ("the audio is weird");
        # the top pages' edits carry slow motion on the music alone and use no meme sound effects (STYLE-GUIDE.md)
        voice = [v for v in voice if not v.get('rate')]
        sfx = []
    E = {'size': [1080, 1920], 'fps': 30, 'duration': round(END, 3), 'mutes': {src: studio.slur_mutes(W)},
         'music': {'src': S['path'], 'start': mstart, 'at': mat, 'gain': 0.95, 'drop_gain': 0.35, 'fade_in': 0.2, 'fade_out': 0.4, 'muffle_until': DROP,
                   'levels': {'pre': -19, 'post': -11}},   # measured: the muffled bed under the talking, the drop 4-5 dB over it
         'voice': voice, 'shots': shots, 'transitions': trans, 'hits': hits, 'sfx': sfx, 'texts': texts,
         'finish': {'grain': 0.02, 'vignette': 0.22}}
    for x, y in zip(shots, shots[1:]):
        assert abs(x['at'] + x['dur'] - y['at']) < 0.02, (x['at'], x['dur'], y['at'])
    return E, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--a', type=float, default=0); ap.add_argument('--b', type=float, default=None)
    ap.add_argument('--streamer', default=''); ap.add_argument('--credit', default=''); ap.add_argument('--handle', default='cutroom')
    ap.add_argument('--song', default='', help='a name from portfolio/music/library.json; default: a hype sound picked per clip')
    ap.add_argument('--plan'); ap.add_argument('--id')
    ap.add_argument('--hint', default='', help="what viewers titled the clip, if it came from a viewer clip")
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--game', action='store_true', help='a gaming stream: the game on top at near full height, the cam below')
    a = ap.parse_args()
    a.src = os.path.abspath(a.src)                      # the EDL must work from any directory
    os.makedirs(a.out, exist_ok=True)
    cid = a.id or os.path.basename(os.path.normpath(a.out))
    if a.b is None:
        import cv2
        c = cv2.VideoCapture(a.src); a.b = c.get(7) / (c.get(5) or 30)
    a.b = min(a.b, video_end(a.src, a.b))
    W = [w for w in hear(a.src, a.a, a.b, os.path.join(a.out, 'words.json')) if w['s'] < a.b - 0.1]
    plan = json.load(open(a.plan)) if a.plan else ask_plan(a.src, a.a, a.b, W, a.streamer, a.hint)
    json.dump(plan, open(os.path.join(a.out, 'plan.json'), 'w'), indent=1, ensure_ascii=False)
    layout = overlay_layout(a.src, a.a, a.b, a.game)
    if layout:
        print('overlay layout (facecam over content): cam on top, content below', layout)
    if not a.song:                                      # the editor's vibe picks the sound (popular TikTok sounds first)
        a.song = pick_song(plan.get('vibe') if plan.get('vibe') in ('hype', 'aura', 'funny') else 'hype', cid)
        print('song:', a.song, '| vibe:', plan.get('vibe'))
    E, S = build(plan, a.src, W, a.credit, a.handle, a.song, layout)
    if not layout:                                      # stream alerts, chat, another page's captions: cropped out per shot
        import castcam
        os.makedirs(os.path.join(a.out, 'log'), exist_ok=True)
        print('overlays: %d shots cropped clear, %d blurred (a crop would cut a face)' % castcam.regions_or_hide(a.out, [s for s in E['shots'] if s.get('src')]))
    json.dump(E, open(os.path.join(a.out, 'edit.json'), 'w'), indent=1, ensure_ascii=False)
    mp4 = os.path.join(a.out, f'{cid}.mp4')
    studio.render(E, mp4, a.workers)
    site = os.path.join(a.out, f'{cid}-site.mp4')
    import tempfile, shutil
    td = tempfile.mkdtemp(prefix='hypecut-'); tmp = os.path.join(td, 'site.wav')
    studio.mix_audio({**E, 'music': None}, tmp, td)
    subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-i', mp4, '-i', tmp, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
                    '-b:a', '160k', '-shortest', '-movflags', '+faststart', site], check=True)
    shutil.rmtree(td, ignore_errors=True)
    meta = {'id': cid, 'title': plan['title'], 'kind': 'hype', 'duration': E['duration'], 'handle': a.handle, 'credit': a.credit,
            'music': f"{S['credit']}: baked into the TikTok copy; the site copy has none", 'post_caption': plan.get('caption') or plan['title'],
            # the sound to attach in the TikTok app (a post on the official sound joins its trend): where our cut starts it
            'sound': {'name': S['credit'], 'tiktok': S.get('tiktok'), 'start': round(E['music'].get('start', 0), 1), 'at': round(E['music'].get('at', 0), 1)},
            'made_by': 'hypecut (editor plan) + studio'}
    json.dump(meta, open(os.path.join(a.out, f'{cid}.json'), 'w'), indent=1, ensure_ascii=False)
    print(f"{mp4}: {E['duration']:.1f} s · {plan['title']}")


if __name__ == '__main__':
    main()
