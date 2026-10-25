#!/usr/bin/env python3
"""cutroom studio: a timeline compositor for real edits (story edits, velocity/hype edits), cut to music.

An edit is a JSON plan (EDL). Pass 1 renders every shot as a 1080x1920 plate: the source frame range read at the shot's
speed curve (slow-mo is frame-blended), framed on a focus point, with its own camera move (zoom/pan/rotate, eased) and
look (black-and-white, glow, sad/warm/cold grade, RGB split). Pass 2 lays the plates on the timeline and joins them with
transitions (zoom-blur punch, whip pan, white flash, dip to black, glitch, dissolve), adds hits on beats (flash, shake,
zoom punch), animated text (dates, lyrics, captions, titles; TikTok Sans), grain and vignette, and mixes the audio: the
music from its offset with fades, voice lines from the streams on top with the music ducked under them.

EDL (all times in seconds on the edit's timeline unless noted):
{
 "size": [1080, 1920], "fps": 30, "duration": 32.0,
 "music": {"src": "song.wav", "start": 8.0, "gain": 1.0, "fade_in": 0.3, "fade_out": 1.5, "muffle_until": 8.2},
 "voice": [{"src": "a.mp4", "in": 12.3, "out": 15.1, "at": 0.6, "gain": 1.4, "duck": 0.3}],
 "shots": [{"src": "a.mp4", "in": 3.2, "at": 0.0, "dur": 1.3,
            "speed": 1.0 | [[0, 1.0], [0.6, 0.25], [1, 1.0]],      # speed over the shot's normalised time
            "focus": [0.5, 0.45] | "auto" | "auto:both" | "auto:left" | "auto:right", "cover": true,   # point to keep centred, or follow faces
            "face_fill": 0.2, "headroom": 0.35,                     # with auto focus: zoom so the face is this share of the height
            "box": [0, 420, 1080, 1350],                            # render into this part of the frame, black around it
            "cam": [[0, 1.0, 0, 0, 0], [1, 1.12, 0, -0.02, 0]],      # [norm t, zoom, dx, dy (frac of frame), rot deg], eased
            "look": ["bw", "glow", "sad", "rgbsplit"]}],
 "transitions": [{"at": 1.3, "type": "zoom" | "whip" | "flash" | "black" | "glitch" | "dissolve", "dur": 0.3}],
 "hits": [{"at": 4.6, "type": "flash" | "shake" | "punch" | "pulse", "amt": 1.0}],
 "layers": [{"src": "b.mp4", "in": 2.0, "at": 40.0, "dur": 2.5, "opacity": 0.5, "fade": 0.4, ...shot keys}],   # double exposure
 "sfx": [{"at": 8.2, "type": "riser" | "boom" | "whoosh", "dur": 2.0, "gain": 0.5}],     # a riser ends at "at"
 "texts": [{"t0": 0.2, "t1": 2.0, "text": "JUNE 2025", "color": [255, 182, 213], "style": "date" | "lyric" | "caption" | "title", "anim": "pop" | "type" | "fade" | "rise", "cps": 22, "y": 1500}],
 "finish": {"grain": 0.035, "vignette": 0.35, "fade_out": 1.2}
}
usage: studio.py edit.json out.mp4 [--preview] [--workers 4]
       studio.py edit.json site.mp4 --remix-without-music out.mp4      (same picture, voice lines and effects only)
"""
import argparse, json, math, os, re, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
FF = os.path.join(PIPE, '..', '.venv', 'bin', 'ffmpeg')
FONTS = os.path.join(PIPE, 'fonts')
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def keyed(keys, t):
    """Interpolate [t, v...] keys (norm time) with smoothstep easing."""
    if not keys:
        return None
    if t <= keys[0][0]:
        return keys[0][1:]
    for a, b in zip(keys, keys[1:]):
        if a[0] <= t <= b[0]:
            u = ease((t - a[0]) / max(1e-6, b[0] - a[0]))
            return [x + (y - x) * u for x, y in zip(a[1:], b[1:])]
    return keys[-1][1:]


def probe(src):
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def source_times(shot, fps):
    """Source time for every output frame of the shot, from its speed curve."""
    n = max(1, round(shot['dur'] * fps))
    sp = shot.get('speed', 1.0)
    out, t = [], shot['in']
    for i in range(n):
        out.append(t)
        u = i / max(1, n - 1)
        s = sp if isinstance(sp, (int, float)) else keyed(sp, u)[0]
        t += s / fps
    return out


def atmos(img, kinds, t, seed=0):
    """Atmosphere over a frame at second t, so a still or a static shot reads as filmed: 'dust' (motes drifting in the
    light), 'snow', 'rain', 'embers' (rising sparks), 'flicker' (candle light breathing). Particles are a pure function
    of (seed, t): continuous across frames, the same on a re-render."""
    import cv2
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    if 'flicker' in kinds:
        f = 1 + 0.035 * np.sin(t * 7.3 + seed) + 0.025 * np.sin(t * 13.1 + 2 * seed) + 0.015 * np.sin(t * 23.7)
        out *= f
    lay = np.zeros((H, W, 3), np.float32)
    def field(n, k):
        r = np.random.default_rng(seed * 101 + k)
        return r.random(n), r.random(n), r.random(n), r.random(n)
    if 'dust' in kinds:
        x0, y0, sp, ph = field(70, 1)
        x = (x0 * W + 18 * np.sin(t * 0.4 + ph * 6.28) + t * 6 * (sp - 0.5)) % W
        y = (y0 * H - t * (8 + 14 * sp)) % H
        a = 0.35 + 0.35 * np.sin(t * (1 + 2 * sp) + ph * 6.28)
        for xi, yi, ai, si in zip(x, y, a, sp):
            cv2.circle(lay, (int(xi), int(yi)), 1 + int(3 * si), (200 * ai, 225 * ai, 245 * ai), -1, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 1.6)
    if 'snow' in kinds:
        x0, y0, sp, ph = field(140, 2)
        x = (x0 * W + 25 * np.sin(t * 0.9 + ph * 6.28)) % W
        y = (y0 * H + t * (60 + 140 * sp)) % H
        for xi, yi, si in zip(x, y, sp):
            cv2.circle(lay, (int(xi), int(yi)), 2 + int(5 * si), (235, 235, 235), -1, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 1.2) * 0.8
    if 'rain' in kinds:
        x0, y0, sp, ph = field(220, 3)
        y = (y0 * H + t * (1400 + 900 * sp)) % H
        x = (x0 * W + (y - y0 * H) * 0.12) % W
        for xi, yi, si in zip(x, y, sp):
            L = 30 + 50 * si
            cv2.line(lay, (int(xi), int(yi)), (int(xi + L * 0.12), int(yi + L)), (190, 200, 210), 1 + int(si > 0.7), cv2.LINE_AA)
        lay *= 0.45
    if 'embers' in kinds:
        x0, y0, sp, ph = field(60, 4)
        x = (x0 * W + 30 * np.sin(t * 1.3 + ph * 6.28)) % W
        y = (y0 * H - t * (90 + 160 * sp)) % H
        fl = 0.55 + 0.45 * np.sin(t * (6 + 9 * sp) + ph * 6.28)
        for xi, yi, fi, si in zip(x, y, fl, sp):
            cv2.circle(lay, (int(xi), int(yi)), 2 + int(3 * si), (40 * fi, 140 * fi, 255 * fi), -1, cv2.LINE_AA)
        lay = cv2.GaussianBlur(lay, (0, 0), 2.0) * 1.4
    out = out + lay                                     # light adds (motes, sparks and drops catch the light)
    return np.clip(out, 0, 255).astype(np.uint8)


def look(img, fx, t=0.0):
    import cv2
    if not fx:
        return img
    f = img.astype(np.float32)
    if 'film' in fx:                                  # the reference edits' look: gentle S-curve, warm highlights, teal shadows
        x = f / 255.0
        x = x + 0.12 * (x - 0.5) * (1 - np.abs(2 * x - 1))         # contrast, soft at the ends
        lum = x.mean(axis=2, keepdims=True)
        x = x + (lum - 0.5) * np.array([-0.035, 0.0, 0.04]) * 2     # BGR: shadows to teal, highlights to warm
        x = x * 0.97 + 0.012                                          # a touch of lift in the blacks
        f = np.clip(x, 0, 1) * 255
    if 'sad' in fx:                                   # cool, lifted blacks, low saturation
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        f = f * 0.55 + g * 0.45
        f = f * np.array([1.08, 0.98, 0.9]) * 0.92 + 14
    if 'warm' in fx:
        f = f * np.array([0.9, 1.0, 1.1]) + np.array([0, 4, 10])
    if 'cold' in fx:
        f = f * np.array([1.12, 1.0, 0.88])
    if 'bw' in fx:
        g = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = np.clip((g - 128) * 1.35 + 128, 0, 255)
        f = np.repeat(g[..., None], 3, axis=2)
    if 'glow' in fx:
        b = cv2.GaussianBlur(np.clip(f, 0, 255).astype(np.uint8), (0, 0), 18).astype(np.float32)
        f = 255 - (255 - f) * (255 - b * 0.55) / 255          # screen blend
    if 'rgbsplit' in fx:
        k = 6
        f[..., 2] = np.roll(f[..., 2], k, axis=1); f[..., 0] = np.roll(f[..., 0], -k, axis=1)
    return np.clip(f, 0, 255).astype(np.uint8)


def frame_cover(img, W, H, focus, zoom, dx, dy, rot):
    """Scale to cover WxH (times zoom), keep the focus point centred (clamped to the image), shift and rotate."""
    import cv2
    h, w = img.shape[:2]
    s = max(W / w, H / h) * zoom
    fx, fy = focus
    cx = min(max(fx * w * s, W / 2), w * s - W / 2) if w * s >= W else w * s / 2
    cy = min(max(fy * h * s, H / 2), h * s - H / 2) if h * s >= H else h * s / 2
    A = np.array([[s, 0, W / 2 - cx + dx * W], [0, s, H / 2 - cy + dy * H]], np.float32)
    if rot:
        R = cv2.getRotationMatrix2D((W / 2, H / 2), rot, 1.0)
        A = (np.vstack([R, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2].astype(np.float32)
    return cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def frame_contain(img, W, H, zoom, dx, dy):
    """The whole source over a blurred, darkened copy of itself (for wide shots)."""
    import cv2
    bg = cv2.GaussianBlur(frame_cover(img, W // 4, H // 4, (0.5, 0.5), 1.0, 0, 0, 0), (0, 0), 6)
    bg = (cv2.resize(bg, (W, H)) * 0.55).astype(np.uint8)
    h, w = img.shape[:2]
    s = min(W / w, H / h) * zoom
    A = np.array([[s, 0, (W - w * s) / 2 + dx * W], [0, s, (H - h * s) / 2 + dy * H]], np.float32)
    fg = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), A, (W, H)) > 0
    bg[mask] = fg[mask]
    return bg


YUNET = os.path.join(PIPE, 'models', 'yunet.onnx')


def face_track(frames, base, src_fps, mode, default, maxspan=1.0, headroom=0.35):
    """Focus over time from faces (OpenCV YuNet), smoothed. mode: auto (biggest face), auto:both (the pair), auto:left,
    auto:right. Returns t -> (fx, fy) in normalised source coordinates; the default where no face is found."""
    import cv2
    n, h, w = frames.shape[:3]
    sc = 640 / w
    det = cv2.FaceDetectorYN.create(YUNET, '', (640, int(h * sc)), 0.6, 0.3, 50)
    step = max(1, int(src_fps / 6))
    pts, sizes = [], []
    for k in range(0, n, step):
        _, f = det.detect(cv2.resize(frames[k], (640, int(h * sc))))
        if f is None or not len(f):
            continue
        fs = sorted(([x / sc, y / sc, fw / sc, fh / sc, c] for x, y, fw, fh, *_, c in f), key=lambda r: -r[2] * r[3])
        big = [r for r in fs if r[2] >= 0.45 * fs[0][2]]
        if mode == 'auto:left':
            pick = [min(big, key=lambda r: r[0])]
        elif mode == 'auto:right':
            pick = [max(big, key=lambda r: r[0])]
        elif mode == 'auto:both' and len(big) > 1 and max(big[0][0] + big[0][2], big[1][0] + big[1][2]) - min(big[0][0], big[1][0]) < maxspan * w:
            pick = big[:2]                              # the pair, when both fit in the crop
        else:
            pick = big[:1]
        x0 = min(r[0] for r in pick); x1 = max(r[0] + r[2] for r in pick)
        y0 = min(r[1] for r in pick); y1 = max(r[1] + r[3] for r in pick)
        fh_ = max(r[3] for r in pick)
        pts.append((base + k / src_fps, (x0 + x1) / 2 / w, min(0.85, ((y0 + y1) / 2 + headroom * fh_) / h)))
        sizes.append(fh_)
    if not pts:
        f0 = lambda t: default
        f0.fh = None
        return f0
    T = np.array([p[0] for p in pts]); X = np.array([p[1] for p in pts]); Y = np.array([p[2] for p in pts])
    if len(pts) >= 3:                                   # drop one-sample jumps, then smooth
        med = np.array([np.median(X[max(0, i - 2): i + 3]) for i in range(len(X))])
        keep = np.abs(X - med) < 0.12
        T, X, Y = T[keep], X[keep], Y[keep]
        k = min(len(X), 5)
        X = np.convolve(np.pad(X, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')[:len(T)]
        Y = np.convolve(np.pad(Y, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')[:len(T)]
    f = lambda t: (float(np.interp(t, T, X)), float(np.interp(t, T, Y)))
    f.fh = float(np.median(sizes))                      # face height in source pixels (for zoom-to-face)
    return f


def motion_track(frames, base, src_fps, default=(0.5, 0.5)):
    """Focus over time from movement, not faces: the biggest moving blob (someone lifted and spun, a fall, a dance), with a
    foreground reactor's body masked out (the biggest face when it's large: the streamer watching it happen). Chat and
    alert boxes change in small scattered patches, so the largest blob wins over them. Smoothed. t -> (fx, fy)."""
    import cv2
    n, h, w = frames.shape[:3]
    sw_, sh_ = 320, int(h * 320 / w)
    det = cv2.FaceDetectorYN.create(YUNET, '', (640, int(h * 640 / w)), 0.6, 0.3, 50)
    step = max(1, int(src_fps / 10))
    prev, pts, mask_box = None, [], None
    for j, k in enumerate(range(0, n, step)):
        if j % 5 == 0:                                  # the foreground reactor, every ~0.5 s
            _, f = det.detect(cv2.resize(frames[k], (640, int(h * 640 / w))))
            mask_box = None
            if f is not None and len(f):
                x, y, fw, fh = max(f, key=lambda r: r[2] * r[3])[:4] / 640 * sw_
                if fh > 0.12 * sh_:
                    mask_box = (int(x - 1.3 * fw), int(y - 0.6 * fh), int(x + 2.3 * fw), sh_)
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frames[k], (sw_, sh_)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if prev is not None:
            d = cv2.absdiff(g, prev)
            _, m = cv2.threshold(d, 14, 255, cv2.THRESH_BINARY)
            if mask_box:
                x0, y0, x1, y1 = mask_box; m[max(0, y0):y1, max(0, x0):max(0, x1)] = 0
            m = cv2.dilate(m, np.ones((7, 7), np.uint8))
            nl, lab, st, cen = cv2.connectedComponentsWithStats(m)
            if nl > 1:
                b = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                if st[b, cv2.CC_STAT_AREA] > 0.004 * sw_ * sh_:   # the blob's box centre: someone held up high stays in (a
                    bx_, by_, bw_, bh_ = st[b, :4]                   # pixel centroid sank to the legs and cut the lifted body)
                    pts.append((base + k / src_fps, (bx_ + bw_ / 2) / sw_, (by_ + bh_ / 2) / sh_))
        prev = g
    if len(pts) < 2:
        return lambda t: default
    T = np.array([p[0] for p in pts]); X = np.array([p[1] for p in pts]); Y = np.array([p[2] for p in pts])
    med = np.array([np.median(X[max(0, i - 3): i + 4]) for i in range(len(X))])
    keep = np.abs(X - med) < 0.15                      # a one-off flicker elsewhere doesn't yank the frame
    T, X, Y = T[keep], X[keep], Y[keep]
    k = min(len(X), 7)
    X = np.convolve(np.pad(X, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')[:len(T)]
    Y = np.convolve(np.pad(Y, (k // 2, k // 2), mode='edge'), np.ones(k) / k, 'valid')[:len(T)]
    return lambda t: (float(np.interp(t, T, X)), float(np.interp(t, T, Y)))


# the most a shot's source pixels get enlarged (crop region x camera zoom x face zoom compound: Kai's montage reaction hit
# ~3.4x and read as "blurry, grainy"); past this the camera move gives way
MAX_UP = 2.6


def card(img, W, H, zoom=1.0, maxup=None):
    """A small source (a tiny facecam) as a card: sharp at no more than maxup times its size, centred over a blurred,
    darkened copy of itself that fills the frame, the way clip pages pop out a small cam, instead of a 4-9x blow-up
    (Kai's Minecraft cam is 250x210 px: full width it was mush)."""
    import cv2
    maxup = maxup or MAX_UP
    h, w = img.shape[:2]
    bg = frame_cover(cv2.resize(img, (max(2, w // 8), max(2, h // 8)), interpolation=cv2.INTER_AREA), W, H, (0.5, 0.5), 1.0, 0, 0, 0)
    bg = (cv2.GaussianBlur(bg, (0, 0), 14) * 0.5).astype(np.uint8)
    sc = min(maxup * zoom, 0.94 * W / w, 0.94 * H / h)
    fw, fh = max(2, int(w * sc) // 2 * 2), max(2, int(h * sc) // 2 * 2)
    fg = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_CUBIC)
    x0, y0 = (W - fw) // 2, (H - fh) // 2
    bg[max(0, y0 - 4):y0 + fh + 4, max(0, x0 - 4):x0 + fw + 4] = 245
    bg[y0:y0 + fh, x0:x0 + fw] = fg
    return bg


def cover_or_card(img, W, H, focus, zoom=1.0):
    h, w = img.shape[:2]
    return card(img, W, H) if max(W / w, H / h) > MAX_UP * 1.15 else frame_cover(img, W, H, focus, zoom, 0, 0, 0)


def render_plate(job):
    """One shot -> raw BGR frames file (plate) at the output size."""
    import cv2
    shot, W, H, fps, out = job
    if shot.get('box') and shot.get('src'):             # the shot inside a box [x, y, w, h] of the frame, black around it
        bx, by, bw, bh = shot['box']
        inner = {k: v for k, v in shot.items() if k != 'box'}
        p, n = render_plate((inner, bw, bh, fps, out + '.box'))
        fr = np.memmap(p, np.uint8, 'r', shape=(n, bh, bw, 3))
        bg = np.full((H, W, 3), shot.get('box_bg', 0), np.uint8)
        with open(out, 'wb') as f:
            for k in range(n):
                img = bg.copy(); img[by:by + bh, bx:bx + bw] = fr[k]
                f.write(img.tobytes())
        del fr; os.unlink(p)
        return out, n
    if not shot.get('src'):                             # a black shot (date cards over black)
        n = max(1, round(shot['dur'] * fps))
        with open(out, 'wb') as f:
            f.write(np.zeros((n, H, W, 3), np.uint8).tobytes())
        return out, n
    src_fps, sw, sh = probe(shot['src'])
    times = source_times(shot, fps)
    lo, hi = min(times), max(times) + 2 / src_fps
    # decode just this range at the source rate
    raw = subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-ss', f'{max(0, lo - 0.05):.3f}', '-i', shot['src'], '-t', f'{hi - lo + 0.2:.3f}',
                          '-vf', f'scale={sw}:{sh}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout
    n_src = len(raw) // (sw * sh * 3)
    if not n_src:                                       # past the last video frame (audio runs on): hold the last picture
        raw = subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-sseof', '-1.5', '-i', shot['src'], '-vf', f'scale={sw}:{sh}',
                              '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True).stdout[-(sw * sh * 3):]
        n_src = len(raw) // (sw * sh * 3)
    frames = np.frombuffer(raw, np.uint8)[: n_src * sw * sh * 3].reshape(n_src, sh, sw, 3) if n_src else None
    if frames is not None and shot.get('balance'):
        # one grade across scenes from different streams: half-strength gray-world white balance and exposure toward a common
        # level, measured on a few of the shot's frames (the reference edits keep a consistent look across sources)
        smp = frames[:: max(1, n_src // 6)].astype(np.float32)
        m = smp.reshape(-1, 3).mean(axis=0) + 1e-3
        wb = np.clip((m.mean() / m) ** 0.5, 0.86, 1.16)
        luma = float((smp[..., 0] * 0.114 + smp[..., 1] * 0.587 + smp[..., 2] * 0.299).mean())
        ex = float(np.clip((112.0 / max(1.0, luma)) ** 0.6, 0.82, 1.3))
        frames = np.clip(frames.astype(np.float32) * (wb * ex), 0, 255).astype(np.uint8)
    if frames is not None and shot.get('hide'):        # overlay boxes [x0, y0, x1, y1] (fractions) painted out: stream alerts or a
        frames = frames.copy()                          # corner webcam that a crop can't avoid without cutting someone's head.
        # Filled from the surroundings (inpaint at quarter size, soft edge): a blurred box read as "leftover censoring"
        m = np.zeros((sh, sw), np.uint8)
        for x0, y0, x1, y1 in shot['hide']:
            m[max(0, int(y0 * sh) - 4):int(y1 * sh) + 4, max(0, int(x0 * sw) - 4):int(x1 * sw) + 4] = 255
        q = 4; mq = cv2.resize(m, (sw // q, sh // q), interpolation=cv2.INTER_NEAREST)
        soft = (cv2.GaussianBlur(m, (0, 0), 6).astype(np.float32) / 255.0)[..., None]
        ys, xs = np.where(m > 0); y0_, y1_, x0_, x1_ = max(0, ys.min() - 24), min(sh, ys.max() + 24), max(0, xs.min() - 24), min(sw, xs.max() + 24)
        for k in range(n_src):
            fill = cv2.inpaint(cv2.resize(frames[k], (sw // q, sh // q), interpolation=cv2.INTER_AREA), mq, 9, cv2.INPAINT_TELEA)
            fill = cv2.GaussianBlur(cv2.resize(fill, (sw, sh), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
            sl = (slice(y0_, y1_), slice(x0_, x1_))
            frames[k][sl] = (frames[k][sl] * (1 - soft[sl]) + fill[sl] * soft[sl]).astype(np.uint8)
    base = max(0, lo - 0.05)
    if shot.get('camsplit') and frames is not None:   # a streamer's facecam over gameplay: the cam on top, the game below
        cs = shot['camsplit']; top_h = int(H * cs.get('cam_frac', 0.317)) // 2 * 2
        def sub(img, r):
            x, y, w, h = r
            return img[int(y * sh): int((y + h) * sh), int(x * sw): int((x + w) * sw)]
        cam_k = shot.get('cam') or [[0, 1.0, 0, 0, 0], [1, 1.0, 0, 0, 0]]
        fx_ = shot.get('look') or []
        with open(out, 'wb') as f:
            for i, t in enumerate(times):
                x = (t - base) * src_fps
                img = frames[int(min(max(0, math.floor(x)), n_src - 1))]
                u = i / max(1, len(times) - 1)
                z, dx, dy, rot = keyed(cam_k, u)
                if cs.get('order') == 'game_top':      # a game: the game on top at near full height, the cam below
                    gh = H - top_h
                    top = frame_cover(sub(img, cs['screen']), W, gh, (0.5, 0.5), z, dx, dy, 0)
                    bot = cover_or_card(sub(img, cs['cam']), W, top_h, (0.5, 0.42))
                    fr = np.vstack([top, bot]); fr[gh - 3: gh + 3] = 0
                else:
                    top = cover_or_card(sub(img, cs['cam']), W, top_h, (0.5, 0.45))
                    bot = frame_cover(sub(img, cs['screen']), W, H - top_h, (0.5, 0.5), z, dx, dy, 0)
                    fr = np.vstack([top, bot]); fr[top_h - 3: top_h + 3] = 0
                f.write(look(fr, fx_, u).tobytes())
        return out, len(times)
    if shot.get('region') and frames is not None:     # only part of the frame (e.g. the facecam box), as if it were the source
        x, y, w, h = shot['region']
        frames = np.ascontiguousarray(frames[:, int(y * sh): int((y + h) * sh), int(x * sw): int((x + w) * sw)])
        sh, sw = frames.shape[1], frames.shape[2]
    focus = shot.get('focus', [0.5, 0.45])
    cam = shot.get('cam') or [[0, 1.0, 0, 0, 0], [1, 1.0, 0, 0, 0]]
    fx = shot.get('look') or []
    track = None
    if shot.get('split') and frames is not None:      # two people: left face on top, right face below
        tl = face_track(frames, base, src_fps, 'auto:left', (0.3, 0.45))
        tr = face_track(frames, base, src_fps, 'auto:right', (0.7, 0.45))
        zs = shot.get('split_zoom', 1.9)
        with open(out, 'wb') as f:
            for i, t in enumerate(times):
                x = (t - base) * src_fps
                k0 = int(min(max(0, math.floor(x)), n_src - 1))
                img = frames[k0]
                u = i / max(1, len(times) - 1)
                z, dx, dy, rot = keyed(cam, u)
                top = frame_cover(img, W, H // 2, tl(t), zs * z, 0, 0, 0)
                bot = frame_cover(img, W, H - H // 2, tr(t), zs * z, 0, 0, 0)
                fr = np.vstack([top, bot]); fr[H // 2 - 3: H // 2 + 3] = 0
                f.write(look(fr, fx, u).tobytes())
        return out, len(times)
    if shot.get('react') and frames is not None:      # the action on top, the reactor's face below (a streamer reacting
        mt = motion_track(frames, base, src_fps)        # to something behind them: the clip pages' reaction split)
        rf = face_track(frames, base, src_fps, 'auto', (0.25, 0.5))
        top_h = int(H * 0.62)                          # the action gets the bigger share (critic: 56% made it small)
        za = shot.get('action_zoom', 1.25)
        zr = min(3.0, max(1.2, 0.36 * (H - top_h) / (rf.fh * max(W / sw, (H - top_h) / sh)))) if rf.fh else 1.6
        with open(out, 'wb') as f:
            for i, t in enumerate(times):
                x = (t - base) * src_fps
                img = frames[int(min(max(0, round(x)), n_src - 1))]
                u = i / max(1, len(times) - 1)
                z, dx, dy, rot = keyed(cam, u)
                top = frame_cover(img, W, top_h, mt(t), za * z, 0, 0, 0)
                bot = frame_cover(img, W, H - top_h, rf(t), zr, 0, 0, 0)
                fr = np.vstack([top, bot]); fr[top_h - 3: top_h + 3] = 0
                f.write(look(fr, fx, u).tobytes())
        return out, len(times)
    if focus == 'motion' and frames is not None:       # the crop follows the movement instead of the faces
        track = motion_track(frames, base, src_fps, tuple(shot.get('focus_default', [0.5, 0.5])))
        focus = None
    if isinstance(focus, str) and frames is not None:
        zmax = max(k[1] for k in (shot.get('cam') or [[0, 1.0]]))
        span = W / (sw * max(W / sw, H / sh) * zmax)    # crop width as a fraction of the source width
        track = face_track(frames, base, src_fps, focus, tuple(shot.get('focus_default', [0.5, 0.45])), 0.92 * span, shot.get('headroom', 0.35))
    elif isinstance(focus, str):
        focus = shot.get('focus_default', [0.5, 0.45])
    zf = 1.0
    if track is not None and shot.get('face_fill') and getattr(track, 'fh', None):
        # zoom so the face is face_fill of the frame height: close on one person wherever they stand, never past 2.2x
        zf = min(2.4, max(1.0, shot['face_fill'] * H / (track.fh * max(W / sw, H / sh))))
        track = face_track(frames, base, src_fps, focus, tuple(shot.get('focus_default', [0.5, 0.45])), 0.92 * span / zf, shot.get('headroom', 0.35))
    trail = None                                         # 'trails' look: a fading echo of the last frames (velocity edits' ghosting)
    with open(out, 'wb') as f:
        for i, t in enumerate(times):
            if frames is None:
                img = np.zeros((H, W, 3), np.uint8)
            else:
                x = (t - base) * src_fps
                k0 = int(min(max(0, math.floor(x)), n_src - 1)); k1 = min(k0 + 1, n_src - 1); a = x - math.floor(x)
                img = frames[k0] if a < 0.5 or k1 == k0 else frames[k1]   # nearest frame: slow motion stays sharp (blending smeared it)
            u = i / max(1, len(times) - 1)
            z, dx, dy, rot = keyed(cam, u)
            z *= zf
            z = max(1.0, min(z, MAX_UP / max(W / sw, H / sh)))   # no blurry 3x blow-ups (never below cover: < 1 mirrored the edges)
            fc = track(t) if track else focus
            if shot.get('region') and max(W / sw, H / sh) > MAX_UP * 1.15:      # a tiny cam region alone: a card, not mush
                img = card(img, W, H, zoom=min(1.25, z))
            else:
                img = frame_cover(img, W, H, fc, z, dx, dy, rot) if shot.get('cover', True) else frame_contain(img, W, H, z, dx, dy)
            if 'trails' in (fx or []):
                trail = img.astype(np.float32) if trail is None else trail * 0.5 + img.astype(np.float32) * 0.5
                img = np.maximum(img, (trail * 0.78).astype(np.uint8))      # a light echo: the hit itself stays readable
            if shot.get('atmos'):
                img = atmos(img, shot['atmos'], i / fps, shot.get('atmos_seed', 0))
            f.write(look(img, fx, u).tobytes())
    return out, len(times)


# ---------------- pass 2 ----------------
def zoom_blur(img, amt):
    """Radial blur toward the centre: average a few scaled copies."""
    import cv2
    H, W = img.shape[:2]
    acc = img.astype(np.float32)
    n = 6
    for k in range(1, n):
        s = 1 + amt * k / n
        M = np.array([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]], np.float32)
        acc += cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT101).astype(np.float32)
    return (acc / n).astype(np.uint8)


def motion_blur_x(img, px):
    import cv2
    if px < 2:
        return img
    k = np.zeros((1, int(px)), np.float32); k[0, :] = 1 / int(px)
    return cv2.filter2D(img, -1, k)


def shift(img, dx, dy):
    import cv2
    H, W = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT101)


def scale_center(img, s):
    import cv2
    H, W = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]]), (W, H), borderMode=cv2.BORDER_REFLECT101)


EMOJI_FONT = '/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf'


def is_emoji(ch):
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or o in (0x200D, 0xFE0F)


def emoji_img(seq, size):
    """A colour emoji (Noto, drawn at its native 109 px) scaled to the text size."""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(EMOJI_FONT, 109)
    im = Image.new('RGBA', (150 * max(1, len(seq)), 160), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((10, 10), seq, font=f, embedded_color=True)
    bb = im.getbbox() or (0, 0, 1, 1)
    im = im.crop(bb)
    k = size * 1.05 / max(1, im.height)
    return im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))))


def runs(line):
    """Split a line into text and emoji runs."""
    out = []
    for ch in line:
        e = is_emoji(ch)
        if out and out[-1][0] == e:
            out[-1][1] += ch
        else:
            out.append([e, ch])
    return out


STYLES = {  # size, font, stroke, uppercase
    'date': (86, 'TikTokSansWide-Black.ttf', 4, False), 'title': (92, 'TikTokSansWide-Black.ttf', 4, False),
    'lyric': (64, 'TikTokSans-Bold.ttf', 3, False), 'caption': (54, 'TikTokSans-Bold.ttf', 4, False),
    'hype': (100, 'TikTokSansWide-Black.ttf', 8, True), 'hook': (70, 'TikTokSansWide-Black.ttf', 6, False),
    'stage': (76, 'TikTokSansWide-Black.ttf', 7, True), 'mark': (34, 'TikTokSans-Bold.ttf', 2, False),
    'ranktitle': (54, 'TikTokSansWide-Black.ttf', 5, True), 'ranktitle46': (46, 'TikTokSansWide-Black.ttf', 4, True),
    'ranktitle40': (40, 'TikTokSansWide-Black.ttf', 4, True), 'ranktitle34': (34, 'TikTokSansWide-Black.ttf', 3, True), 'rankrow': (42, 'TikTokSans-Bold.ttf', 3, True),
    'num': (260, 'TikTokSansWide-Black.ttf', 14, False), 'cinecap': (40, 'TikTokSans-Bold.ttf', 3, True),
    'cinetitle': (96, 'TikTokSansWide-Black.ttf', 0, True), 'numsm': (170, 'TikTokSansWide-Black.ttf', 12, False),
}


def text_png(text, style, W, color=(255, 255, 255)):
    """A text layer as RGBA (PIL: TikTok Sans with outline and glow, colour emoji from Noto), wrapped to the frame."""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    size, fname, stroke, upper = STYLES.get(style, (60, 'TikTokSans-Bold.ttf', 4, False))
    font = ImageFont.truetype(os.path.join(FONTS, fname), size)
    if upper:
        text = text.upper()
    d0 = ImageDraw.Draw(Image.new('RGBA', (1, 1)))

    def width(line):                                    # '§word' = the word in the accent colour (the marker isn't drawn)
        return sum(d0.textlength(t.replace('§', ''), font=font) if not e else emoji_img(t, size).width + 6 for e, t in runs(line)) + 2 * stroke

    lines = []                                          # wrap to 90% of the frame
    for para in text.split('\n'):
        cur = ''
        for wd in para.split(' '):
            trial = (cur + ' ' + wd).strip()
            if cur and width(trial) > W * 0.9:
                lines.append(cur); cur = wd
            else:
                cur = trial
        lines.append(cur)
    asc, desc = font.getmetrics()
    lh = asc + desc
    tw = int(max(width(l) for l in lines)) + 40; th = lh * len(lines) + int(size * 0.25) * (len(lines) - 1) + 40
    im = Image.new('RGBA', (tw, th), (0, 0, 0, 0)); dr = ImageDraw.Draw(im)
    y = 20
    for l in lines:
        x = (tw - width(l)) / 2 + stroke
        for e, t in runs(l):
            if e:
                em = emoji_img(t, size)
                im.alpha_composite(em, (int(x + 3), int(y + (lh - em.height) / 2)))
                x += em.width + 6
            else:
                for piece in re.split(r'(\s+)', t):
                    acc = piece.startswith('§')
                    piece = piece.replace('§', '')
                    if piece:
                        dr.text((x, y), piece, font=font, fill=(*((255, 214, 10) if acc else color), 255), stroke_width=stroke, stroke_fill=(0, 0, 0, 235))
                        x += d0.textlength(piece, font=font)
        y += lh + int(size * 0.25)
    glow = im.filter(ImageFilter.GaussianBlur(10))
    out = Image.alpha_composite(Image.new('RGBA', im.size, (0, 0, 0, 0)), glow)
    out = Image.alpha_composite(out, im)
    return np.array(out)


def paste_rgba(dst, rgba, cx, cy, scale=1.0, alpha=1.0):
    import cv2
    if scale != 1.0:
        rgba = cv2.resize(rgba, (max(1, int(rgba.shape[1] * scale)), max(1, int(rgba.shape[0] * scale))))
    h, w = rgba.shape[:2]; H, W = dst.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return dst
    src = rgba[ya - y0: yb - y0, xa - x0: xb - x0]
    a = (src[..., 3:4].astype(np.float32) / 255) * alpha
    rgb = src[..., [2, 1, 0]].astype(np.float32)
    dst[ya:yb, xa:xb] = (dst[ya:yb, xa:xb] * (1 - a) + rgb * a).astype(np.uint8)
    return dst


def compose_chunk(job):
    """Frames [f0, f1) of the edit -> raw BGR file."""
    import cv2
    E, plates, f0, f1, out = job
    W, H = E['size']; fps = E['fps']
    shots = E['shots']
    mm = {i: np.memmap(p, np.uint8, 'r', shape=(n, H, W, 3)) for i, (p, n) in plates.items()}
    texts = {k: text_png(t['text'], t.get('style', 'caption'), W, tuple(t.get('color', (255, 255, 255)))) for k, t in enumerate(E.get('texts', []))}
    typed = {}

    def typed_png(k, n):                                # typewriter: the first n characters, laid out like the full line
        if (k, n) not in typed:
            full = E['texts'][k]['text']
            typed[(k, n)] = text_png(full[:n] + ''.join(' ' if c != '\n' else c for c in full[n:]), E['texts'][k].get('style', 'caption'), W,
                                     tuple(E['texts'][k].get('color', (255, 255, 255))))
        return typed[(k, n)]
    rng = np.random.default_rng(f0)
    fin = E.get('finish', {})
    vign = None
    if fin.get('vignette'):
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        vign = (1 - fin['vignette'] * np.clip(r - 0.35, 0, 1) ** 1.6)[..., None].astype(np.float32)

    layers = E.get('layers', [])
    lm = {j: np.memmap(p, np.uint8, 'r', shape=(n, H, W, 3)) for j, (p, n) in E.get('_layer_plates', {}).items()}

    def shot_frame(i, t):
        s = shots[i]; n = plates[i][1]
        k = int(round((t - s['at']) * fps))
        return mm[i][min(max(0, k), n - 1)]

    with open(out, 'wb') as f:
        for fr in range(f0, f1):
            t = fr / fps
            cur = next((i for i, s in enumerate(shots) if s['at'] <= t < s['at'] + s['dur']), None)
            if cur is None:
                img = np.zeros((H, W, 3), np.uint8)
            else:
                img = shot_frame(cur, t).copy()
            for j, ly in enumerate(layers):                 # double exposure: screen-blend a layer on top, faded in and out
                if j in lm and ly['at'] <= t < ly['at'] + ly['dur']:
                    k = min(int(round((t - ly['at']) * fps)), lm[j].shape[0] - 1)
                    age, left = t - ly['at'], ly['at'] + ly['dur'] - t
                    op = ly.get('opacity', 0.5) * min(1, age / ly.get('fade', 0.4), left / ly.get('fade', 0.4))
                    top = lm[j][k].astype(np.float32)
                    scr = 255 - (255 - img.astype(np.float32)) * (255 - top) / 255
                    img = (img * (1 - op) + scr * op).astype(np.uint8)
            # transitions: centred on the cut time
            for tr in E.get('transitions', []):
                d = tr.get('dur', 0.3); a = tr['at']
                if not (a - d / 2 <= t < a + d / 2):
                    continue
                u = (t - (a - d / 2)) / d                       # 0..1 across the transition
                peak = 1 - abs(u - 0.5) * 2                     # 0 -> 1 at the cut -> 0
                ty = tr['type']
                if ty == 'zoom':
                    img = scale_center(img, 1 + 0.35 * peak if u < 0.5 else 1 + 0.25 * peak)
                    img = zoom_blur(img, 0.22 * peak)
                elif ty in ('whip', 'smear'):
                    prv = next((i for i, s_ in enumerate(shots) if abs(s_['at'] + s_['dur'] - a) < 0.02), None)
                    nxt = next((i for i, s_ in enumerate(shots) if abs(s_['at'] - a) < 0.02), None)
                    if prv is None or nxt is None:
                        img = motion_blur_x(img, 4 + 90 * peak)
                    elif ty == 'whip':
                        # one fast pan across the cut: the outgoing shot slides off, the incoming slides in beside it, both
                        # in the same motion, smeared by the speed (never an edge stretched to fill the gap)
                        A_ = shot_frame(prv, min(t, a - 1 / fps)); B_ = shot_frame(nxt, max(t, a))
                        e = 0.5 - 0.5 * math.cos(math.pi * u)            # ease in and out
                        o = int(round(e * W))
                        strip = np.concatenate([A_, B_], axis=1) if tr.get('dir', 'left') == 'left' else np.concatenate([B_, A_], axis=1)
                        x0 = o if tr.get('dir', 'left') == 'left' else W - o
                        img = motion_blur_x(np.ascontiguousarray(strip[:, x0:x0 + W]), 4 + 140 * peak)
                    else:
                        # the smear: both shots blurred by the speed of the cut and crossfaded through it (the 100M+ edits' move)
                        A_ = shot_frame(prv, min(t, a - 1 / fps)); B_ = shot_frame(nxt, max(t, a))
                        mix = cv2.addWeighted(A_, 1 - u, B_, u, 0)
                        img = zoom_blur(motion_blur_x(mix, 4 + 70 * peak), 0.12 * peak)
                elif ty == 'spin':
                    # the CapCut-era spin: one full turn across the cut (the outgoing shot turns 0-180 deg while pushing in,
                    # the incoming finishes 180-360 and settles), blurred along the turn by its speed
                    prv = next((i for i, s_ in enumerate(shots) if abs(s_['at'] + s_['dur'] - a) < 0.02), None)
                    nxt = next((i for i, s_ in enumerate(shots) if abs(s_['at'] - a) < 0.02), None)
                    if prv is None or nxt is None:
                        img = zoom_blur(img, 0.2 * peak)
                    else:
                        e = 0.5 - 0.5 * math.cos(math.pi * u)
                        src_ = shot_frame(prv, min(t, a - 1 / fps)) if u < 0.5 else shot_frame(nxt, max(t, a))
                        ang = (1 if tr.get('dir', 'cw') == 'cw' else -1) * e * 360
                        acc = np.zeros(src_.shape, np.float32)
                        for k_ in range(9):                    # 9 samples along the turn: 5 left visible stepped copies
                            Rm = cv2.getRotationMatrix2D((W / 2, H / 2), ang + 34 * peak * (k_ - 4) / 4, 1 + 0.6 * peak)
                            acc += cv2.warpAffine(src_, Rm, (W, H), borderMode=cv2.BORDER_REFLECT101)
                        img = (acc / 9).astype(np.uint8)
                elif ty == 'flash':
                    img = cv2.addWeighted(img, 1 - 0.95 * peak, np.full_like(img, 255), 0.95 * peak, 0)
                elif ty == 'black':
                    img = (img * (1 - peak)).astype(np.uint8)
                elif ty == 'glitch':
                    if peak > 0.2:
                        f2 = img.copy(); k = int(18 * peak)
                        f2[..., 2] = np.roll(img[..., 2], k, 1); f2[..., 0] = np.roll(img[..., 0], -k, 1)
                        for _ in range(4):
                            y0 = int(rng.integers(0, H - 80)); hh = int(rng.integers(20, 80))
                            f2[y0:y0 + hh] = np.roll(f2[y0:y0 + hh], int(rng.integers(-60, 60)), 1)
                        img = f2
                elif ty == 'dissolve':
                    prv = next((i for i, s in enumerate(shots) if abs(s['at'] + s['dur'] - a) < 0.02), None)
                    nxt = next((i for i, s in enumerate(shots) if abs(s['at'] - a) < 0.02), None)
                    if prv is not None and nxt is not None:
                        img = cv2.addWeighted(shot_frame(prv, min(t, a - 1 / fps)), 1 - u, shot_frame(nxt, max(t, a)), u, 0)
            for h in E.get('hits', []):
                a = h['at']; amt = h.get('amt', 1.0)
                if not (a <= t < a + 0.35):
                    continue
                v = 1 - (t - a) / 0.35
                if h['type'] == 'flash':
                    img = cv2.addWeighted(img, 1 - 0.6 * v * amt, np.full_like(img, 255), 0.6 * v * amt, 0)
                elif h['type'] == 'shake':
                    img = shift(img, 18 * amt * v * math.sin(t * 95), 12 * amt * v * math.cos(t * 80))
                elif h['type'] == 'punch':
                    img = scale_center(img, 1 + 0.08 * amt * v)
                elif h['type'] == 'pulse':                     # beat flash: brightness up on the beat, decaying
                    img = np.clip(img.astype(np.float32) * (1 + 0.16 * amt * v ** 2), 0, 255).astype(np.uint8)
            # text
            for k, tx in enumerate(E.get('texts', [])):
                if not (tx['t0'] <= t < tx['t1']):
                    continue
                age = t - tx['t0']; left = tx['t1'] - t
                anim = tx.get('anim', 'pop')
                sc, al, dy = 1.0, 1.0, 0
                if anim == 'pop':                              # 60% -> 115% -> 100% over 0.2 s
                    sc = 0.6 + 0.55 * ease(age / 0.1) if age < 0.1 else (1.15 - 0.15 * ease((age - 0.1) / 0.1) if age < 0.2 else 1.0)
                elif anim == 'fade':
                    al = min(1, age / 0.25, left / 0.25)
                elif anim == 'rise':
                    al = min(1, age / 0.2, left / 0.2); dy = 30 * (1 - ease(age / 0.3))
                elif anim == 'type':
                    al = min(1, left / 0.2)
                layer = texts[k]
                if anim == 'type':
                    n = int(age * tx.get('cps', 22)) + 1
                    if n < len(tx['text']):
                        layer = typed_png(k, n)
                al = min(al, left / 0.12) if anim != 'fade' else al
                paste_rgba(img, layer, W / 2, tx.get('y', H * 0.75) + dy, sc, max(0, al) * tx.get('opacity', 1.0))
            fo = fin.get('fade_out', 0)
            if fo and t > E['duration'] - fo:
                img = (img * max(0.0, (E['duration'] - t) / fo)).astype(np.uint8)
            if vign is not None:
                img = (img * vign).astype(np.uint8)
            if fin.get('grain'):
                g = rng.normal(0, 255 * fin['grain'], (H // 2, W // 2)).astype(np.float32)
                g = cv2.resize(g, (W, H))[..., None]
                img = np.clip(img.astype(np.float32) + g, 0, 255).astype(np.uint8)
            f.write(img.tobytes())
    return out


def speech_gain(v, target=-12.5):
    """Gain that brings a voice line to the target RMS (dBFS), measured on its loud half (pauses don't count)."""
    pcm = subprocess.run([FF, '-nostdin', '-loglevel', 'error', '-ss', f"{v['in']:.3f}", '-t', f"{v['out'] - v['in']:.3f}", '-i', v['src'],
                          '-f', 's16le', '-ac', '1', '-ar', '16000', '-'], capture_output=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
    if len(x) < 1600:
        return 1.0
    fr = x[: len(x) // 800 * 800].reshape(-1, 800)
    rms = np.sqrt((fr ** 2).mean(1))
    loud = np.sort(rms)[len(rms) // 2:]
    db = 20 * np.log10(np.sqrt((loud ** 2).mean()) + 1e-9)
    return float(min(40.0, max(0.3, 10 ** ((target - db) / 20))))   # quiet Twitch clips can sit 30 dB under Kick audio


def synth(kind, dur, sr=48000, seed=0):
    """Edit sound effects from noise and sines: riser (builds into a drop), boom (the drop), whoosh (whip pans)."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr); t = np.arange(n) / sr
    if kind == 'riser':
        x = rng.normal(0, 1, n)
        out = np.zeros(n); lp = 0.0
        a = np.linspace(0.02, 0.45, n)                  # one-pole low-pass that opens up: dull hiss -> bright
        for i in range(n):
            lp += a[i] * (x[i] - lp); out[i] = lp
        out = out - np.convolve(out, np.ones(64) / 64, 'same')
        out *= np.linspace(0, 1, n) ** 2.2 * 2.2
        out += 0.25 * np.sin(2 * np.pi * np.cumsum(np.linspace(180, 900, n)) / sr) * np.linspace(0, 1, n) ** 3
    elif kind == 'boom':
        out = np.sin(2 * np.pi * np.cumsum(np.linspace(70, 38, n)) / sr) * np.exp(-t * 5.0) * 1.1
        out += rng.normal(0, 1, n) * np.exp(-t * 40) * 0.35
    else:                                               # whoosh
        x = rng.normal(0, 1, n); out = np.zeros(n); lp = 0.0
        a = 0.05 + 0.4 * np.sin(np.pi * t / dur)
        for i in range(n):
            lp += a[i] * (x[i] - lp); out[i] = lp
        out *= np.sin(np.pi * t / dur) ** 2 * 1.6
    out = np.clip(out, -1, 1)
    return (np.repeat(out[:, None], 2, 1) * 32767 * 0.5).astype(np.int16)


# slurs never ship: in a studio edit the word is muted (the bed carries it; captions are masked by the builders). The same
# words edit.py bleeps in plain clips. A moment with a slur said in passing is cut with the word muted, not thrown away
# (the catalog had Kai's 1.1M-view floor scramble at 1/10 for one word).
SLUR_AUDIO = re.compile(r'n+i+g+g|\bf+a+g+(?:s|got|gots|gy)?\b|r+e+t+a+r+d', re.I)


def slur_mutes(words):
    """[[s, e]] source seconds of every slur in a transcript ([{'w', 's', 'e'}]), padded a little either side."""
    return [[round(max(0.0, w['s'] - 0.05), 3), round(w.get('e', w['s'] + 0.4) + 0.07, 3)] for w in words if SLUR_AUDIO.search(str(w.get('w', '')))]


def lufs(src, ss, t, af=''):
    """Integrated loudness (LUFS) of a stretch of a file, after an optional filter; None for silence or nothing."""
    if t <= 0.4:
        return None
    r = subprocess.run([FF, '-nostdin', '-hide_banner', '-ss', f'{max(0, ss):.3f}', '-t', f'{t:.3f}', '-i', src, '-af', (af + ',' if af else '') + 'ebur128',
                        '-f', 'null', '-'], capture_output=True, text=True)
    v = re.findall(r'I:\s+(-?[\d.]+) LUFS', r.stderr)
    return float(v[-1]) if v and float(v[-1]) > -70 else None


# the muffled bed under the build: low-passed, then levelled over ~3.5 s (a song's quiet bar left slow-motion stretches
# with no voice in a 14 dB hole: the deodorant edit, 14-16.5 s)
MUFFLE = 'lowpass=f=650,lowpass=f=650,dynaudnorm=f=400:g=9:p=0.7:m=6'


def music_levels(m, dur):
    """music.levels {'pre': LUFS, 'post': LUFS}: the bed set by measurement instead of fixed multipliers. Songs differ by
    10 dB and a phonk bed keeps its bass when muffled, so a fixed 1.5x muffle sat louder than the drop (the car-crash edit:
    -9 LUFS under the build, -18 after the drop, backwards). Returns (gain before the drop, gain from the drop)."""
    lv = m['levels']; at = m.get('at', 0); st = m.get('start', 0); mu = m.get('muffle_until')
    db = lambda x: 10 ** (x / 20)
    split = (mu - at) if mu else 0.0
    i_pre = lufs(m['src'], st, split, MUFFLE) if mu else None
    i_post = lufs(m['src'], st + split, max(0.5, dur - (mu or at)))
    g_pre = db(lv.get('pre', -22) - i_pre) if i_pre is not None else 0.3
    g_post = db(lv.get('post', -12) - i_post) if i_post is not None else 0.8
    return min(g_pre, 6.0), min(g_post, 4.0)


def mix_audio(E, out_wav, tmp):
    """Music from its offset, voice lines on top, music ducked under each voice line."""
    dur = E['duration']; ins, parts, k = [], [], 0
    m = E.get('music')
    if m:
        ins += ['-ss', f"{m.get('start', 0):.3f}", '-t', f"{dur + 0.5 - m.get('at', 0):.3f}", '-i', m['src']]
        g = m.get('gain', 1.0)
        g_lo = 1.5                                      # the muffled bed's level (relative), the full song's below
        if m.get('levels'):                             # measured levels: the gains below replace gain / drop_gain
            g_lo, g = music_levels(m, dur)
            m = {**m, 'drop_gain': None}
            if not m.get('muffle_until'):
                g_lo = g
        if m.get('swell_at') is not None:              # the bed rises into the payoff (+40% over a second), the way story edits land
            g = f"{g}*(1+0.4*clip((t-{m['swell_at'] + m.get('at', 0):.3f})/1.0,0,1))"
        if m.get('drop_gain') is not None and m.get('muffle_until'):   # the music after the drop, relative to before it
            g = f"{g}*if(gte(t,{m['muffle_until'] + m.get('at', 0):.3f}),{m['drop_gain']},1)"
        # ducking happens below, smoothly, from the voices themselves (sidechain): no hard on/off step around every line
        duck = ''
        mu = m.get('muffle_until')
        open_ = f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo"
        if mu:                                          # low-passed until the drop, then the full song (a 0.12 s crossfade)
            hi = f"{1.0 if not m.get('levels') else 1.0}"
            lo = f"{g_lo / g:.4f}" if m.get('levels') else f"{g_lo}"      # relative to the main gain g applied after
            open_ = (f"{open_},asplit[mlo][mhi];[mlo]{MUFFLE if m.get('levels') else 'lowpass=f=650,lowpass=f=650'},volume='{lo}*clip(({mu - m.get('at', 0):.3f}-t)/0.12,0,1)':eval=frame[mlo2];"
                     f"[mhi]volume='{hi}*clip((t-{mu - m.get('at', 0):.3f}+0.12)/0.12,0,1)':eval=frame[mhi2];[mlo2][mhi2]amix=inputs=2:normalize=0")
        mat = m.get('at', 0)                          # music can come in later in the edit
        dl = f",adelay={int(mat * 1000)}|{int(mat * 1000)}" if mat else ''
        parts.append(f"{open_},afade=t=in:d={m.get('fade_in', 0.3)}{dl},volume='{g}{duck}':eval=frame,"
                     f"afade=t=out:st={max(0, dur - m.get('fade_out', 1.5)):.2f}:d={m.get('fade_out', 1.5)}[m]")
        k += 1
    labels = []
    vlabels = []
    for j, v in enumerate(E.get('voice', [])):
        # a breath of the line's own room before and after it (pre 0.04 s, tail 0.14 s, faded), so lines don't start and
        # stop on a hard edge of silence
        pre, tail = (0.04, 0.14) if not v.get('rate') else (0.0, 0.0)
        # the breath is shortened where the source or the edit has no room for it (in < pre, at < pre), and the delay is
        # taken from the seek actually used: each clamped on its own, a line played up to 40 ms early or late (measured
        # -41 ms by the fx session's openers lab, 2026-10-11)
        lead = min(pre, v['in'], v['at'])
        ss = v['in'] - lead
        ins += ['-ss', f"{ss:.3f}", '-t', f"{v['out'] - ss + tail:.3f}", '-i', v['src']]
        ms = int(round((v['at'] - lead) * 1000))
        g = v.get('gain', 1.0) * (speech_gain(v) if v.get('norm', True) else 1.3)
        rate = v.get('rate')                            # slowed voice (slow-motion sound, pitched down)
        rs = f",asetrate={int(48000 * rate)},aresample=48000" if rate else ''
        vlen = (v['out'] - ss + tail) / (rate or 1)
        base = ss                                       # muted slurs, in this input's own time (it starts at base)
        mutes = [(a_ - base, b_ - base) for a_, b_ in (E.get('mutes') or {}).get(v['src'], []) if b_ > base and a_ < v['out'] + tail]
        mf = (",volume=0:enable='" + '+'.join(f'between(t,{max(0, a_):.3f},{b_:.3f})' for a_, b_ in mutes) + "'") if mutes else ''
        parts.append(f"[{k}:a]aresample=48000{mf}{rs},aformat=channel_layouts=stereo,highpass=f=90,volume={g:.3f},"
                     f"acompressor=threshold=0.12:ratio=3:attack=5:release=120:makeup=1.4,afade=t=in:d=0.07,afade=t=out:st={max(0, vlen - 0.2):.2f}:d=0.2,adelay={ms}|{ms}[v{j}]")
        vlabels.append(f'[v{j}]'); k += 1
    import wave
    for j, fx in enumerate(E.get('sfx', [])):
        kind = fx['type']; d = fx.get('dur', {'riser': 2.0, 'boom': 1.6, 'whoosh': 0.45}.get(kind, 0.5))
        pth = os.path.join(tmp, f'sfx{j}.wav')
        with wave.open(pth, 'wb') as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000); w.writeframes(synth(kind, d, seed=j).tobytes())
        start = fx['at'] - d if kind == 'riser' else fx['at'] - (d / 2 if kind == 'whoosh' else 0)   # a riser ends on its time
        ms = max(0, int(start * 1000))
        ins += ['-i', pth]
        parts.append(f"[{k}:a]volume={fx.get('gain', 0.5)},adelay={ms}|{ms}[s{j}]")
        k += 1
    # the voices as one bus; the music ducks under it smoothly (sidechain: 40 ms down, 400 ms back up), so between two
    # lines the bed comes back up gently instead of a hole and a jump
    if vlabels:
        # the key is padded with silence to the full length: sidechaincompress stops at its shorter input, which cut every
        # edit off at its last voice line
        # (no music: no split; a split with one side into anullsink hung ffmpeg next to SFX inputs)
        parts.append(f"{''.join(vlabels)}amix=inputs={len(vlabels)}:normalize=0:duration=longest,apad=whole_dur={dur:.3f}" + (",asplit=2[vbus][vkey]" if m else "[vbus]"))
        if m:
            # about 9 dB under a line (ratio 2; ratio 6 took ~18 dB and the bed pumped up in every pause between words)
            parts.append(f"[m][vkey]sidechaincompress=threshold=0.025:ratio=2:attack=30:release=600:knee=6:makeup=1[md]")
            labels += ['[md]', '[vbus]']
        else:
            labels += ['[vbus]']
    elif m:
        labels.append('[m]')
    labels = labels + [l for l in [f'[s{j}]' for j in range(len(E.get('sfx', [])))]]
    if not labels:
        subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', f'anullsrc=r=48000:cl=stereo', '-t', f'{dur}', out_wav], check=True)
        return
    # one static gain to the target and a peak limiter, not single-pass loudnorm: that one rides the gain through the edit
    # (it breathed between lines and flattened the drop back down to the talking's level)
    pre = os.path.join(tmp, 'premix.wav')
    fc = ';'.join(parts) + f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,atrim=0:{dur:.3f}[a]"
    subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', *ins, '-filter_complex', fc, '-map', '[a]', '-ar', '48000', '-c:a', 'pcm_f32le', pre], check=True)
    i = lufs(pre, 0, dur + 1)
    gain = max(-20.0, min(30.0, E.get('loudness', -13.0) - i)) if i is not None else 0.0
    subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-i', pre, '-af', f'volume={gain:.2f}dB,alimiter=limit=0.84:attack=3:release=60:level=0',
                    '-ar', '48000', '-c:a', 'pcm_s16le', out_wav], check=True)
    os.unlink(pre)


def motivate_transitions(E):
    import cv2
    """Pick each beat transition from the motion at the cut, the way an editor does: sideways motion carries a whip in
    its own direction, a push/pull a zoom, anything else a smear. Flashes on the drop, dissolves and dips to black stay.
    The glitch (cheap-looking) is always replaced. Measured with optical flow on two frames either side of the cut."""
    shots = E.get('shots') or []
    def flow(src, t):
        if not src:
            return 0.0, 0.0
        c = cv2.VideoCapture(src); fr = []
        for tt in (t, t + 0.1):
            c.set(cv2.CAP_PROP_POS_MSEC, max(0.0, tt) * 1000); ok, im = c.read()
            if not ok:
                return 0.0, 0.0
            fr.append(cv2.cvtColor(cv2.resize(im, (320, 180)), cv2.COLOR_BGR2GRAY))
        f = cv2.calcOpticalFlowFarneback(fr[0], fr[1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx = float(np.median(f[..., 0]))
        yy, xx = np.mgrid[0:180, 0:320]
        div = float(np.median((f[..., 0] * (xx - 160) + f[..., 1] * (yy - 90)) / (np.hypot(xx - 160, yy - 90) + 1)))
        return dx, div
    for tr in E.get('transitions') or []:
        if tr.get('type') not in ('whip', 'zoom', 'glitch', 'auto') or tr.get('keep'):
            continue
        a = tr['at']
        prv = next((s_ for s_ in shots if abs(s_['at'] + s_['dur'] - a) < 0.02), None)
        nxt = next((s_ for s_ in shots if abs(s_['at'] - a) < 0.02), None)
        if not prv or not nxt:
            continue
        sp = prv.get('speed', 1.0) if isinstance(prv.get('speed', 1.0), (int, float)) else 1.0
        d1, v1 = flow(prv.get('src'), prv.get('in', 0) + prv['dur'] * max(0.1, sp) - 0.2)
        d2, v2 = flow(nxt.get('src'), nxt.get('in', 0) + 0.05)
        dx, dv = d1 + d2, v1 + v2
        if abs(dx) > 2.5:
            tr.update(type='whip', dir='left' if dx < 0 else 'right', dur=max(tr.get('dur', 0.22), 0.22))
        elif abs(dv) > 0.6:
            tr.update(type='zoom', dur=max(tr.get('dur', 0.22), 0.22))
        else:
            tr.update(type='smear', dur=max(tr.get('dur', 0.22), 0.24))
        tr['why'] = f'flow dx {dx:.1f} div {dv:.2f}'
    return E


def render(E, out, workers=4, preview=False):
    if E.get('motivate', True):
        E = motivate_transitions(E)
    W, H = E['size']; fps = E['fps']
    if preview:
        E = {**E, 'size': [W // 2, H // 2]}; W, H = W // 2, H // 2
    tmp = tempfile.mkdtemp(prefix='studio-')
    jobs = [(s, W, H, fps, os.path.join(tmp, f'plate{i}.raw')) for i, s in enumerate(E['shots'])]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(render_plate, jobs))
    plates = {i: r for i, r in enumerate(res)}
    if E.get('layers'):
        ljobs = [(s, W, H, fps, os.path.join(tmp, f'layer{j}.raw')) for j, s in enumerate(E['layers'])]
        with ProcessPoolExecutor(workers) as ex:
            E = {**E, '_layer_plates': {j: r for j, r in enumerate(ex.map(render_plate, ljobs))}}
    n = int(round(E['duration'] * fps))
    step = math.ceil(n / workers)
    chunks = [(E, plates, a, min(n, a + step), os.path.join(tmp, f'chunk{a:06d}.raw')) for a in range(0, n, step)]
    with ProcessPoolExecutor(workers) as ex:
        outs = list(ex.map(compose_chunk, chunks))
    wav = os.path.join(tmp, 'mix.wav')
    mix_audio(E, wav, tmp)
    enc = subprocess.Popen([FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-i', wav,
                            '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-maxrate', '14M', '-bufsize', '28M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                            '-shortest', '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for o in outs:                                      # stream the chunks straight into the encoder
        with open(o, 'rb') as g:
            while True:
                b = g.read(1 << 24)
                if not b:
                    break
                enc.stdin.write(b)
        os.unlink(o)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit('encode failed')
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    probe_ = subprocess.run([FF, '-nostdin', '-i', out], capture_output=True, text=True).stderr
    md = re.search(r'Duration: (\d+):(\d+):([\d.]+)', probe_)
    got = int(md.group(1)) * 3600 + int(md.group(2)) * 60 + float(md.group(3)) if md else 0
    if got < E['duration'] - 0.5:                       # never ship a cut-off edit silently (a mix that ends early truncates it)
        raise SystemExit(f'{out}: rendered {got:.1f} s of {E["duration"]:.1f} s')
    print(f'{out}: {E["duration"]:.1f} s, {len(E["shots"])} shots, {len(E.get("transitions", []))} transitions, {len(E.get("texts", []))} texts')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('edl'); ap.add_argument('out')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--remix-without-music', metavar='RENDERED', help='re-mux RENDERED with the voice lines and effects only (site copies)')
    ap.add_argument('--remix', metavar='RENDERED', help='re-mux RENDERED with a fresh full mix from the EDL')
    a = ap.parse_args()
    E = json.load(open(a.edl))
    if a.remix:                                         # new audio mix (from the EDL) onto an already rendered video
        tmp = tempfile.mkdtemp(prefix='studio-')
        wav = os.path.join(tmp, 'mix.wav')
        mix_audio(E, wav, tmp)
        subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-i', a.remix, '-i', wav, '-map', '0:v', '-map', '1:a',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', a.out], check=True)
        print(a.out)
    elif a.remix_without_music:
        tmp = tempfile.mkdtemp(prefix='studio-')
        wav = os.path.join(tmp, 'mix.wav')
        mix_audio({**E, 'music': None}, wav, tmp)
        subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-i', a.remix_without_music, '-i', wav, '-map', '0:v', '-map', '1:a',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', a.out], check=True)
        print(a.out)
    else:
        render(E, a.out, a.workers, a.preview)
