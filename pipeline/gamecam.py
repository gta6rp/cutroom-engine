#!/usr/bin/env python3
"""Gaming streams: the game on top, the cam below, the hook on the seam (the layout top gaming clip pages use, e.g. a
165K-view LosPollosTV x Jynxzi clip: research/references/framing). Two things the general facecam layout got wrong here:

  - it zoomed the game into a narrow slice beside the cam; a game needs most of the screen (the crosshair, what's coming)
  - a duo or squad in voice chat shows each friend's cam on the streamer's layout, and the moment is often the friend's
    line or reaction; one fixed cam misses it. The lower panel cuts to whoever is talking or reacting.

cams(video, a, b)                -> every cam overlay [{x, y, w, h, face, main}] in source pixels, the main (largest) first
speaker_runs(video, a, b, cams)  -> [(t0, t1, k)]: which cam the lower panel shows, with a minimum shot length
game_crop(cams, W, H, pw, ph)    -> {x, y, w, h}: the game panel's window, the middle of the screen at near full height
cam_crop(cam, W, H, pw, ph)      -> {x, y, w, h, card}: the cam re-cut to the panel; `card` when it's too small to fill it

usage: gamecam.py VIDEO A B [PEAK]   prints the cams and the speaker runs, writes gamecam-debug.jpg next to the video
"""
import json, os, sys
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facecam  # noqa: E402

GAME_WORDS = (r'minecraft|fortnite|valorant|warzone|call of duty|\bcod\b|\bgta\b|grand theft|rust\b|apex|\bcs2\b|counter.?strike|'
              r'league of legends|\blol\b|elden ring|roblox|rainbow six|\br6\b|siege|overwatch|marvel rivals|tarkov|dayz|hardcore|'
              r'speedrun|\bnba ?2k|\bfifa\b|\bfc ?2[0-9]|madden|chess|valheim|lethal company|r\.?e\.?p\.?o|schedule i|peak\b|'
              r'horror|resident evil|dead by daylight|\bdbd\b|pubg|fall guys|among us|deadlock|dota|helldivers|battlefield|'
              r'smash|tekken|street fighter|mortal kombat|pokemon|zelda|mario|gaming|\bplays?\b|playthrough|walkthrough')
NOT_GAME = {'just chatting', 'irl', 'talk shows & podcasts', 'podcasts', 'pools, hot tubs & bikinis', 'pools, hot tubs & beaches',
            'music', 'art', 'asmr', 'sports', 'slots & casino', 'slots', 'casino', 'crypto', 'trading', 'special events',
            'travel & outdoors', 'food & drink', 'science & technology', 'fitness & health', 'makers & crafting'}


def is_game(category=None, title=''):
    """A gaming stream: Kick's category when we have it (anything that isn't a talk/IRL/music category), else the title."""
    import re
    if category:
        return category.strip().lower() not in NOT_GAME
    return bool(re.search(GAME_WORDS, title or '', re.I))


def _clusters(imgs, W, H):
    """Faces that stay put across the samples: cams, not people moving in the game."""
    dets = [(i, f) for i, img in enumerate(imgs) for f in facecam.faces(img, 960 / W) if f[2] >= 0.022 * W]
    out, used = [], set()
    for k, (_, (x, y, w, h)) in sorted(enumerate(dets), key=lambda kv: -kv[1][1][2]):
        if k in used:
            continue
        cx, cy = x + w / 2, y + h / 2
        near = [j for j, (_, f) in enumerate(dets) if j not in used and abs(f[0] + f[2] / 2 - cx) < max(0.04 * W, 0.9 * w)
                and abs(f[1] + f[3] / 2 - cy) < max(0.05 * H, 0.9 * h)]
        used |= set(near)
        if len({dets[j][0] for j in near}) >= 0.45 * len(imgs):
            out.append([dets[j][1] for j in near])
    return out


def cams(video, a, b, n=12):
    imgs = facecam.frames(video, a, b, n)
    if len(imgs) < 4:
        return []
    H, W = imgs[0].shape[:2]
    ex, ey = facecam.static_edges(imgs)
    found = []
    for c in _clusters(imgs, W, H):
        fx, fy, fw, fh = (float(np.median([f[k] for f in c])) for k in range(4))
        if np.std([f[0] + f[2] / 2 for f in c]) > 0.035 * W or fw > 0.16 * W:
            continue                                   # a wandering or huge face: a camera filming the room, not a cam overlay
        bx = facecam.box_for(c, ex, ey, W, H)
        if bx and bx.get('outer'):
            x0, y0, x1, y1 = bx['outer']
        else:                                          # a small cam in a stack: its own borders where they show, else the face's neighbourhood
            rows = slice(int(max(0, fy - fh)), int(min(H, fy + 2 * fh)))
            cols = slice(int(max(0, fx - fw)), int(min(W, fx + 2 * fw)))
            colp = np.percentile(ex[rows], 30, axis=0); rowp = np.percentile(ey[:, cols], 30, axis=1)
            l_ = facecam.border(colp, fx - fw * 2.5, fx - fw * 0.1, 'left'); r_ = facecam.border(colp, fx + fw * 1.1, fx + fw * 3.5, 'right')
            t_ = facecam.border(rowp, fy - fh * 1.6, fy - fh * 0.05, 'left'); b_ = facecam.border(rowp, fy + fh * 1.15, fy + fh * 2.4, 'right')
            x0 = l_ if l_ is not None else fx - 1.15 * fw
            x1 = r_ if r_ is not None else fx + 2.15 * fw
            y0 = t_ if t_ is not None else fy - 0.75 * fh
            y1 = b_ if b_ is not None else fy + 1.9 * fh
        # a webcam is head and shoulders: a box far bigger than that around the face ran into a dark, edgeless game
        x0, x1 = max(x0, fx - 2.4 * fw), min(x1, fx + fw + 2.4 * fw)
        y0, y1 = max(y0, fy - 1.3 * fh), min(y1, fy + fh + 2.3 * fh)
        found.append({'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0, 'face': [fx, fy, fw, fh]})
    if not found:
        return []
    # stacked cams: a box must not run into the next cam's face
    for c in found:
        for o in found:
            if o is c:
                continue
            ofx, ofy, ofw, ofh = o['face']; fx, fy, fw, fh = c['face']
            if abs(ofx - fx) < max(fw, ofw) and ofy > fy and c['y'] + c['h'] > ofy - 0.2 * ofh:       # one below this one
                c['h'] = max(fh * 1.6, (ofy - 0.25 * ofh) - c['y'])
            if abs(ofy - fy) < max(fh, ofh) and ofx > fx and c['x'] + c['w'] > ofx - 0.3 * ofw:       # one beside it
                c['w'] = max(fw * 1.8, (ofx - 0.35 * ofw) - c['x'])
    for c in found:
        x0 = max(0.0, c['x']); y0 = max(0.0, c['y']); x1 = min(W, c['x'] + c['w']); y1 = min(H, c['y'] + c['h'])
        c.update(x=int(x0), y=int(y0), w=int(x1 - x0) // 2 * 2, h=int(y1 - y0) // 2 * 2, face=[round(v) for v in c['face']], src=[W, H])
    for c in found:                                    # a small webcam is head and shoulders: never down into the game under it
        fx, fy, fw, fh = c['face']
        if c["w"] < 0.2 * W and c["y"] + c["h"] > fy + 1.35 * fh:
            c["h"] = int(fy + 1.35 * fh - c["y"]) // 2 * 2
    big = max(c['face'][2] for c in found)
    found = [c for c in found if c['face'][2] >= 0.35 * big]          # a face on a poster behind someone isn't a cam
    # the streamer's own cam is the biggest box, not the biggest face (a friend close to a small webcam fills it)
    found.sort(key=lambda c: -(c['w'] * c['h']))
    for k, c in enumerate(found):
        c['main'] = k == 0
    return found[:4]


def _envelope(video, a, b, fps):
    """The stream audio's loudness, fps values a second (who is talking moves their mouth with it)."""
    import subprocess
    ff = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.venv', 'bin', 'ffmpeg')
    r = subprocess.run([ff, '-nostdin', '-loglevel', 'error', '-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', video, '-vn', '-ac', '1', '-ar', '8000',
                        '-f', 's16le', '-'], capture_output=True)
    x = np.frombuffer(r.stdout, np.int16).astype(np.float32)
    n = int(8000 / fps)
    if len(x) < n * 4:
        return None
    return np.sqrt(np.array([np.mean(x[k:k + n] ** 2) for k in range(0, len(x) - n + 1, n)]) + 1)


def speaker_runs(video, a, b, cs, peak=None, fps=8, min_shot=1.4):
    """Which cam the lower panel shows. The streamer's own cam (the main one: it's his stream, his mic) is the anchor. A
    friend's cam takes over when they are talking (their mouth moves with the stream audio, clearly more than the
    streamer's) or, after the payoff, when they react (a burst of movement: laughing, a hand over the face), for a few
    seconds at a time, never cut faster than min_shot. Each face's movement is read against its own quiet baseline."""
    if len(cs) < 2:
        return [(a, b, 0)]
    cap = cv2.VideoCapture(video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000)
    step = max(1, int(round(src_fps / fps))); fps = src_fps / step
    prev = [None] * len(cs); mo = [[] for _ in cs]; bo = [[] for _ in cs]
    i = 0
    while True:
        if not cap.grab() or a + i / src_fps > b:
            break
        if i % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            for k, c in enumerate(cs):
                fx, fy, fw, fh = c['face']
                m = g[int(fy + 0.5 * fh): int(fy + 1.15 * fh), int(max(0, fx - 0.05 * fw)): int(fx + 1.05 * fw)]
                f = g[int(max(0, fy - 0.1 * fh)): int(fy + 1.2 * fh), int(max(0, fx - 0.2 * fw)): int(fx + 1.2 * fw)]
                cur = (cv2.resize(m, (32, 20)), cv2.resize(f, (32, 32))) if m.size and f.size else None
                if prev[k] is None or cur is None:
                    mo[k].append(0.0); bo[k].append(0.0)
                else:
                    mo[k].append(float(np.abs(cur[0] - prev[k][0]).mean())); bo[k].append(float(np.abs(cur[1] - prev[k][1]).mean()))
                prev[k] = cur
        i += 1
    cap.release()
    M, B = np.array(mo), np.array(bo)
    n = M.shape[1]
    if n < 8:
        return [(a, b, 0)]
    M = M / (np.percentile(M[:, 1:], 30, axis=1, keepdims=True) + 0.5)
    B = B / (np.percentile(B[:, 1:], 30, axis=1, keepdims=True) + 0.5)
    env = _envelope(video, a, b, fps)
    win = int(round(1.5 * fps))                                       # 1.5 s windows, every half second
    hop = max(1, int(round(0.5 * fps)))
    talk = np.zeros((len(cs), 0)).tolist(); burst = [[] for _ in cs]; ts = []
    for s0 in range(0, max(1, n - win + 1), hop):
        ts.append(a + s0 / fps)
        for k in range(len(cs)):
            mk = M[k, s0:s0 + win]
            c_ = 0.0
            if env is not None and len(env) >= s0 + win:
                e = np.diff(env[s0:s0 + win]); m_ = np.diff(mk)
                if e.std() > 0 and m_.std() > 0:
                    c_ = float(np.corrcoef(e, m_)[0, 1])
            talk[k].append(max(0.0, c_) * float(mk.mean()))
            burst[k].append(float(B[k, s0:s0 + win].mean()))
    T, Bu = np.array(talk), np.array(burst)
    cur, since, runs = 0, a, []
    for j, t in enumerate(ts):
        want = 0
        fr = [k for k in range(1, len(cs))]
        best = max(fr, key=lambda k: T[k, j])
        if T[best, j] > 0.9 and T[best, j] > 1.6 * T[0, j]:
            want = best                                               # a friend is talking
        elif peak is not None and peak - 0.5 <= t <= peak + 4:
            rb = max(fr, key=lambda k: Bu[k, j])
            if Bu[rb, j] > 2.6 and Bu[rb, j] > 1.5 * Bu[0, j]:
                want = rb                                             # a friend's reaction to the payoff
        if cur != 0 and t - since > 4.0 and want == cur:
            want = 0                                                  # back to the streamer after a few seconds
        if want != cur and t - since >= min_shot:
            runs.append((since, t, cur)); cur, since = want, t
    runs.append((since, b, cur))
    # a friend who left their seat leaves an empty chair moving in the dark: a friend's shot needs their face in the cam
    cap = cv2.VideoCapture(video)
    for j, (x, y, k) in enumerate(runs):
        if k == 0:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, (x + y) / 2 * 1000); ok, img = cap.read()
        c = cs[k]
        crop = img[c['y']: c['y'] + c['h'], c['x']: c['x'] + c['w']] if ok else None
        if crop is None or not crop.size or not facecam.faces(crop, min(1.0, 480 / max(1, crop.shape[1])) * 2):
            runs[j] = (x, y, 0)
    cap.release()
    merged = []
    for r in runs:                                                    # neighbours on the same cam become one shot
        if merged and merged[-1][2] == r[2]:
            merged[-1] = (merged[-1][0], r[1], r[2])
        else:
            merged.append(r)
    runs = merged
    if len(runs) > 1 and runs[-1][1] - runs[-1][0] < 1.0:
        p = runs.pop(); runs[-1] = (runs[-1][0], p[1], runs[-1][2])
    return [(round(x, 2), round(y, 2), k) for x, y, k in runs if y - x > 0.05]


def chat_columns(video, a, b, cs):
    """Source x ranges where a scrolling chat sits (facecam.overlays), to keep out of the game's window."""
    imgs = facecam.frames(video, a, b, 10)
    if len(imgs) < 4:
        return []
    H, W = imgs[0].shape[:2]
    small = [cv2.cvtColor(cv2.resize(im, (W // 4, H // 4)), cv2.COLOR_BGR2GRAY).astype(np.float32) for im in imgs]
    main = cs[0] if cs else {'x': -10 ** 6, 'w': 0}
    m = facecam.overlays(imgs, small, main, W)
    out, x0 = [], None
    for x in range(W + 1):
        on = x < W and m[x] > 0
        if on and x0 is None:
            x0 = x
        elif not on and x0 is not None:
            if x - x0 >= 48:
                out.append([int(x0 * 1920 / W), int(x * 1920 / W)])
            x0 = None
    return out


def game_crop(cs, W, H, pw, ph, avoid=()):
    """The middle of the screen at near full height (the top strip, where viewer counts and timers sit, cut off), at the
    panel's shape, slid sideways (at most 12% of its width) to keep cams out of it."""
    ch = int(round(0.92 * H / 2) * 2)
    cw = min(W, int(round(ch * pw / ph / 2) * 2))
    y = H - ch
    mid = (W - cw) // 2
    best, bx = None, mid
    for x in range(int(mid - 0.16 * cw), int(mid + 0.16 * cw) + 1, 8):
        x = max(0, min(W - cw, x))
        over = sum(max(0, min(x + cw, c['x'] + c['w']) - max(x, c['x'])) * max(0, min(y + ch, c['y'] + c['h']) - max(y, c['y'])) for c in cs)
        chat = sum(max(0, min(x + cw, r[0] + r[2]) - max(x, r[0])) * max(0, min(y + ch, r[1] + r[3]) - max(y, r[1])) for r in avoid)   # chat counts like a cam
        cost = over + chat + 40 * abs(x - mid)
        if best is None or cost < best:
            best, bx = cost, x
    return {'x': int(bx), 'y': int(y), 'w': cw, 'h': ch}


def cam_crop(c, W, H, pw, ph, max_up=2.9, card_up=3.6):
    """The cam re-cut to the panel's shape: its full height when it's wider than the panel (heads never cut), centred on
    the face. A cam that would have to be blown up more than max_up times is shown as a card instead (sharp, centred,
    over a blurred fill of itself)."""
    asp = pw / ph
    x0, y0, w, h = c['x'], c['y'], c['w'], c['h']
    fx, fy, fw, fh = c['face']
    if w / h > asp:
        ch = h; cw = int(round(ch * asp / 2) * 2)
    else:
        cw = w; ch = int(round(cw / asp / 2) * 2)
    cx = int(min(max(x0, fx + fw / 2 - cw / 2), x0 + w - cw))
    cy = int(min(max(y0, fy + fh / 2 - 0.42 * ch), y0 + h - ch))
    up = ph / ch
    return {'x': max(0, cx), 'y': max(0, cy), 'w': cw // 2 * 2, 'h': ch // 2 * 2, 'card': up > max_up, 'up': round(up, 2),
            'card_h': int(min(ph * 0.86, ch * card_up)) // 2 * 2}


def hide_rects(cs, W):
    """Where each cam sits, with a small margin (the boxes come from the cam's borders or the editor's read, so they're
    close): blurred where they reach into the game's window."""
    out = []
    for c in cs:
        mx, my = int(0.06 * c['w']), int(0.06 * c['h'])
        out.append((max(0, c['x'] - mx), max(0, c['y'] - my), c['w'] + 2 * mx, c['h'] + 2 * my))
    return out


def cam_lift(video, t, c):
    """A dark panel (a gamer's cam in a dim room, a black cave) lifted on its own: '' or an eq filter."""
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000); ok, img = cap.read()
    if not ok:
        return ''
    H, W = img.shape[:2]; sx, sy = W / 1920, H / 1080
    crop = img[int(c['y'] * sy): int((c['y'] + c['h']) * sy), int(c['x'] * sx): int((c['x'] + c['w']) * sx)]
    if not crop.size:
        return ''
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    m, sd = float(g.mean()), float(g.std())
    lift = ''
    if m < 80:
        lift = f",eq=gamma={min(1.7, 1 + (80 - m) / 60):.2f}:saturation=1.05"
        if m < 40:                                      # a near-black cave: open the shadows too (gamma alone tops out)
            lift += ",curves=all='0/0.03 0.12/0.24 0.4/0.55 1/1'"
    if sd < 42 and m >= 60:                             # a grey haze (GTA's warehouses read "murky, low-contrast" on a phone)
        lift += f",eq=contrast={min(1.25, 1 + (42 - sd) / 80):.2f}:saturation=1.1"
    return lift


def filter_graph(game, cam, W_out, game_h, cam_h, hide=None, lift='', game_lift=''):
    """ffmpeg filter for one run: [0:v] (1920x1080) -> [v] (W_out x game_h+cam_h)."""
    g, c = game, cam
    low = (f"[c]crop={c['w']}:{c['h']}:{c['x']}:{c['y']}{lift},scale={W_out}:{cam_h}:flags=lanczos,setsar=1[low]" if not c['card'] else
           f"[c]crop={c['w']}:{c['h']}:{c['x']}:{c['y']}{lift},split=2[c1][c2];"
           f"[c1]scale={W_out}:{cam_h},gblur=sigma=26,eq=brightness=-0.3:saturation=0.8,setsar=1[cbg];"
           f"[c2]scale=-2:{c.get('card_h') or int(min(cam_h * 0.86, c['h'] * 2.9)) // 2 * 2}:flags=lanczos,unsharp=5:5:0.6,setsar=1[cfg];"
           f"[cbg][cfg]overlay=(W-w)/2:(H-h)/2[low]")
    # cams that reach into the game's window (a corner of the streamer's own cam) are blurred out of it: a second copy of
    # a face, or half a face, pulls the eye off the game
    blur, k = '', 0                                    # k: the blurs chained so far (rects outside the window are skipped)
    for (x, y, w, h) in (hide or []):
        x0, y0 = max(x, g['x']), max(y, g['y']); x1, y1 = min(x + w, g['x'] + g['w']), min(y + h, g['y'] + g['h'])
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            rx, ry, rw, rh = x0 - g['x'], y0 - g['y'], (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
            blur += (f"[gg{k}]split=2[gb{k}][gp{k}];[gp{k}]crop={rw}:{rh}:{rx}:{ry},gblur=sigma=26[gq{k}];"
                     f"[gb{k}][gq{k}]overlay={rx}:{ry}[gg{k + 1}];")
            k += 1
    head = f"[g]crop={g['w']}:{g['h']}:{g['x']}:{g['y']}{game_lift}[gg0];"
    tail = f"[gg{k}]scale={W_out}:{game_h}:flags=lanczos,setsar=1[top];"
    return (f"[0:v]fps=30,scale=1920:1080,setsar=1,split=2[g][c];" + head + blur + tail + f"{low};"
            f"[top][low]vstack=inputs=2,setsar=1,format=yuv420p[v]")


if __name__ == '__main__':
    v = sys.argv[1]; a, b = float(sys.argv[2]), float(sys.argv[3])
    cs = cams(v, a, b)
    print(json.dumps(cs))
    runs = speaker_runs(v, a, b, cs, float(sys.argv[4]) if len(sys.argv) > 4 else None)
    print(runs)
    cap = cv2.VideoCapture(v); cap.set(cv2.CAP_PROP_POS_MSEC, (a + b) / 2 * 1000); ok, img = cap.read()
    if ok:
        H, W = img.shape[:2]
        gc = game_crop(cs, W, H, 1080, 1010)
        cv2.rectangle(img, (gc['x'], gc['y']), (gc['x'] + gc['w'], gc['y'] + gc['h']), (255, 120, 0), 4)
        for k, c in enumerate(cs):
            cv2.rectangle(img, (c['x'], c['y']), (c['x'] + c['w'], c['y'] + c['h']), (0, 214, 255) if c['main'] else (0, 255, 120), 4)
            cc = cam_crop(c, W, H, 1080, 910)
            cv2.rectangle(img, (cc['x'], cc['y']), (cc['x'] + cc['w'], cc['y'] + cc['h']), (255, 0, 255), 2)
            cv2.putText(img, f"{k} up{cc['up']}{' card' if cc['card'] else ''}", (c['x'] + 4, c['y'] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(v)), 'gamecam-debug.jpg'), cv2.resize(img, (960, int(960 * H / W))))


def refine_cams(cs, video, t):
    """Tighten each cam to the webcam box the editor reads on the frame (castcam.overlay_boxes, cached per frame), when
    one holds that cam's face: border detection misses the edges of small stacked cams."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'studio'))
    try:
        import castcam
        bs = [b['box'] for b in castcam.overlay_boxes(video, t, _OV) if b.get('kind') == 'webcam']
    except Exception:  # noqa: BLE001
        return cs
    for c in cs:
        W, H = c.get('src', [1920, 1080]); fx, fy, fw, fh = c['face']
        cx, cy = (fx + fw / 2) / W, (fy + fh / 2) / H
        for b in bs:
            bw_, bh_ = (b[2] - b[0]) * W, (b[3] - b[1]) * H
            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and bw_ >= 1.6 * fw and bw_ * bh_ <= 14 * fw * fh and bw_ * bh_ < 0.3 * W * H:
                c.update(x=int(b[0] * W) // 2 * 2, y=int(b[1] * H) // 2 * 2, w=int((b[2] - b[0]) * W) // 2 * 2, h=int((b[3] - b[1]) * H) // 2 * 2)
                break
    for c in cs:                                      # small webcams stay head and shoulders (the editor can merge a cam with a panel under it)
        fx, fy, fw, fh = c['face']
        if c['w'] < 0.2 * c.get('src', [1920])[0] and c['y'] + c['h'] > fy + 1.35 * fh:
            c['h'] = int(fy + 1.35 * fh - c['y']) // 2 * 2
    cs.sort(key=lambda c: -(c['w'] * c['h']))
    for k, c in enumerate(cs):
        c['main'] = k == 0
    return cs


_OV = {}                                                # one editor read per frame, shared by refine_cams and overlay_rects


def overlay_rects(video, t):
    """Stream overlays on the frame at t that aren't cams (chat, alerts, counters, timers), read by the editor
    (studio/castcam.overlay_boxes), as [x, y, w, h] in 1920x1080 pixels: the game's window keeps clear of them."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'studio'))
    try:
        import castcam
        bs = castcam.overlay_boxes(video, t, _OV)
    except Exception:  # noqa: BLE001
        return []
    return [[int(b['box'][0] * 1920), int(b['box'][1] * 1080), int((b['box'][2] - b['box'][0]) * 1920), int((b['box'][3] - b['box'][1]) * 1080)]
            for b in bs if b.get('kind') in ('chat', 'alert', 'counter', 'caption', 'other') and (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1]) < 0.3]


def overlay_rects_span(video, a, b):
    """overlay_rects over a cut's stretch: three frames (a quarter, half and three quarters in), the boxes of all three.
    A clip can switch sources (a stream view with chat, then a room cam), so one frame isn't enough."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as ex:
        rs = list(ex.map(lambda t: overlay_rects(video, t), [a + (b - a) * f for f in (0.25, 0.5, 0.75)]))
    out = []
    for r in (x for rr in rs for x in rr):
        if not any(abs(r[0] - o[0]) < 40 and abs(r[1] - o[1]) < 40 and abs(r[2] - o[2]) < 60 and abs(r[3] - o[3]) < 60 for o in out):
            out.append(r)
    return out
