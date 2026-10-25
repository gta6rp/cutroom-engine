"""A virtual camera operator for the 9:16 speaker layout.

Handheld IRL footage pans, people move, and two people share the frame. The operator:
  look   detects faces 10x a second on a small copy of the video (frontal and profile, both directions),
         picks who to follow with continuity (no jumping to a bigger face for one frame), and frames two people
         together when they fit side by side
  hold   keeps the camera still while the subject stays near the centre (a dead zone), so it isn't nervous
  glide  when the subject drifts out of the zone, the camera moves after them, smoothed forward and backward
         (no lag, no overshoot) and speed-capped
  cut    hard camera cuts in the source (shot changes) are cut, not panned across
  punch  a 1.15x push on the beat, eased in and out
then renders the crop path in one pass with sub-pixel moves (warpAffine), piping frames straight to x264.
"""
import os, subprocess
import numpy as np
import cv2

FPS = 30
LOOK_FPS = 10
SRC_W, SRC_H = 1920, 1080
OUT_W, OUT_H = 1080, 1920
CROP_W = 608                       # 9:16 window in the 1920x1080 source
CASC = cv2.data.haarcascades
FRONT = cv2.CascadeClassifier(os.path.join(CASC, 'haarcascade_frontalface_default.xml'))
PROFILE = cv2.CascadeClassifier(os.path.join(CASC, 'haarcascade_profileface.xml'))


def frames(ff, video, a, b, w, h, fps):
    """Yield BGR frames of [a, b] at fps, scaled to w x h."""
    p = subprocess.Popen([ff, '-nostdin', '-loglevel', 'error', '-ss', f'{a:.3f}', '-to', f'{b:.3f}', '-i', video,
                          '-vf', f'fps={fps},scale={w}:{h}:flags=bilinear', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close(); p.wait()


def faces_in(g):
    """Faces as (cx, cy, size) in the small frame: frontal first, then profiles facing either way."""
    out = [(x + w / 2, y + h / 2, w) for x, y, w, h in FRONT.detectMultiScale(g, 1.1, 5, minSize=(22, 22))]
    if not out:
        out = [(x + w / 2, y + h / 2, w) for x, y, w, h in PROFILE.detectMultiScale(g, 1.1, 5, minSize=(22, 22))]
        fl = cv2.flip(g, 1)
        out += [(g.shape[1] - (x + w / 2), y + h / 2, w) for x, y, w, h in PROFILE.detectMultiScale(fl, 1.1, 5, minSize=(22, 22))]
    return out


def look(ff, video, a, b):
    """Per-look-frame subject x (source px or None) and shot-cut flags."""
    sw, sh = 480, 270
    k = SRC_W / sw
    xs, cuts, prev_h, prev_x = [], [], None, None
    for im in frames(ff, video, a, b, sw, sh, LOOK_FPS):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hist = cv2.normalize(cv2.calcHist([g], [0], None, [32], [0, 256]), None).flatten()
        cut = prev_h is not None and cv2.compareHist(prev_h, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.32
        prev_h = hist
        fs = sorted(faces_in(g), key=lambda f: -f[2])
        x = None
        if fs:
            big = fs[0]
            if prev_x is not None and not cut:
                # continuity: stay with the face nearest the last one unless another is much bigger
                near = min(fs, key=lambda f: abs(f[0] * k - prev_x))
                big = near if near[2] * 1.8 >= fs[0][2] else fs[0]
            x = big[0] * k
            # two people who fit in the window together: frame both
            if len(fs) > 1:
                other = max((f for f in fs if f is not big), key=lambda f: f[2])
                if other[2] > 0.6 * big[2] and abs(other[0] - big[0]) * k < CROP_W * 0.78:
                    x = (other[0] + big[0]) / 2 * k
            prev_x = x
        xs.append(x)
        cuts.append(cut)
    return xs, cuts


YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yunet.onnx')
SIZES = []          # look_av: the followed face's size per look frame (source px), None when framing two together


def size_zoom(cuts, n_out, target=0.10, zmax=1.4):
    """Per shot (between cuts), a steady push so a small face (a wide room shot) reads on a phone: the median face of the
    shot at ~9% of the frame height, never more than zmax (the source is upscaled). 1.0 where nobody was measured."""
    if not SIZES:
        return np.ones(n_out)
    up = FPS / LOOK_FPS
    bounds = [0] + [i for i, c in enumerate(cuts) if c] + [len(SIZES)]
    zl = np.ones(len(SIZES))
    for s0, s1 in zip(bounds, bounds[1:]):
        sz = [v for v in SIZES[s0:s1] if v]
        if len(sz) >= max(3, 0.4 * (s1 - s0)):
            zl[s0:s1] = float(np.clip(target * SRC_H / np.median(sz), 1.0, zmax))
    return np.array([zl[min(len(zl) - 1, int(i / up))] for i in range(n_out)])


def envelope(ff, audio, a, dur, fps):
    """The audio's loudness, fps values a second, from a (seconds) for dur."""
    r = subprocess.run([ff, '-nostdin', '-loglevel', 'error', '-ss', f'{a:.3f}', '-t', f'{dur:.3f}', '-i', audio, '-vn', '-ac', '1', '-ar', '8000',
                        '-f', 's16le', '-'], capture_output=True)
    x = np.frombuffer(r.stdout, np.int16).astype(np.float32)
    n = int(8000 / fps)
    return np.sqrt(np.array([np.mean(x[k:k + n] ** 2) for k in range(0, len(x) - n + 1, n)]) + 1) if len(x) >= 4 * n else None


def look_av(ff, video, a, b, audio, a_audio):
    """look() for a group: when the people in frame don't fit one 9:16 window, follow whoever is talking (their mouth
    moves with the audio), cutting (not panning) to the next speaker, never faster than 1.5 s; the way clip pages
    re-frame a group conversation per speaker. Two people who fit side by side are framed together, as before."""
    sw, sh = 960, 540
    k = SRC_W / sw
    det = cv2.FaceDetectorYN.create(YUNET, '', (sw, sh), 0.6, 0.3, 50)
    tracks, per, cuts, prev_h = [], [], [], None
    for i, im in enumerate(frames(ff, video, a, b, sw, sh, LOOK_FPS)):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        hist = cv2.normalize(cv2.calcHist([g], [0], None, [32], [0, 256]), None).flatten()
        cut = prev_h is not None and cv2.compareHist(prev_h, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.32
        prev_h = hist
        cuts.append(cut)
        _, f = det.detect(im)
        here = []
        for r in ([] if f is None else [r for r in f if r[2] >= 14]):
            cx, cy, sz = r[0] + r[2] / 2, r[1] + r[3] / 2, r[2]
            best = None
            for t in tracks:
                if t['last'] >= i - 3 and not cut and abs(t['x'] - cx) < max(0.9 * sz, 24) and abs(t['y'] - cy) < 0.9 * sz and t not in [h[0] for h in here]:
                    d = abs(t['x'] - cx) + abs(t['y'] - cy)
                    if best is None or d < best[0]:
                        best = (d, t)
            t = best[1] if best else {'id': len(tracks), 'mo': {}, 'pos': {}, 'prev': None, 'last': -9}
            if not best:
                tracks.append(t)
            mx0, mx1 = min(r[10], r[12]) - 0.2 * sz, max(r[10], r[12]) + 0.2 * sz
            my0, my1 = (r[11] + r[13]) / 2 - 0.22 * sz, (r[11] + r[13]) / 2 + 0.3 * sz
            m = g[int(max(0, my0)): int(max(1, my1)), int(max(0, mx0)): int(max(1, mx1))]
            cur = cv2.resize(m, (24, 14)).astype(np.float32) if m.size else None
            if cur is not None and t['prev'] is not None and t['last'] == i - 1:
                t['mo'][i] = float(np.abs(cur - t['prev']).mean())
            t['prev'], t['last'] = cur, i
            t['x'], t['y'] = cx, cy
            t['pos'][i] = (cx * k, cy * k, sz * k)
            here.append((t, r))
        per.append([h[0] for h in here])
    n = len(per)
    env = envelope(ff, audio, a_audio, b - a, LOOK_FPS)
    if env is None or n < 10:
        return None
    ed = np.diff(env[:n + 1]) if len(env) > n else np.diff(np.pad(env, (0, n + 1 - len(env)), mode='edge'))

    def talk(t, i, w=7):
        idx = [j for j in range(max(1, i - w), min(n, i + w + 1)) if j in t['mo'] and j - 1 in t['mo']]
        if len(idx) < 5:
            return 0.0
        mo = np.array([t['mo'][j] for j in idx]); md = np.diff([t['mo'][j - 1] for j in idx] + [mo[-1]])[:len(idx)]
        md = mo - np.array([t['mo'][j - 1] for j in idx])
        e = ed[[j - 1 for j in idx]]
        if md.std() < 1e-6 or e.std() < 1e-6:
            return 0.0
        c = float(np.corrcoef(md, e)[0, 1])
        base = np.median(list(t['mo'].values())) + 0.5
        return max(0.0, c) * float(mo.mean() / base)

    xs, cur, since, streak = [], None, -99, 0
    SIZES.clear()
    for i in range(n):
        ps = [t for t in per[i]]
        if cuts[i]:
            cur, streak = None, 0
        if not ps:
            xs.append(None); SIZES.append(None); continue
        if len(ps) == 1:
            cur = ps[0]; xs.append(ps[0]['pos'][i][0]); SIZES.append(ps[0]['pos'][i][2]); continue
        ps.sort(key=lambda t: -t['pos'][i][2])
        A, B_ = ps[0], ps[1]
        if B_['pos'][i][2] > 0.6 * A['pos'][i][2] and abs(A['pos'][i][0] - B_['pos'][i][0]) < CROP_W * 0.78:
            xs.append((A['pos'][i][0] + B_['pos'][i][0]) / 2); cur = A; SIZES.append(None); continue   # two together: no push
        sc = {t['id']: talk(t, i) for t in ps}
        if cur is None or cur not in ps:
            cur = max(ps, key=lambda t: (sc[t['id']], t['pos'][i][2])); since = i
        else:
            cand = max(ps, key=lambda t: sc[t['id']])
            if cand is not cur and sc[cand['id']] > 0.25 and sc[cand['id']] > 1.4 * sc[cur['id']]:
                streak += 1
            else:
                streak = 0
            if streak >= 4 and i - since >= int(1.5 * LOOK_FPS):
                cur, since, streak = cand, i, 0
                cuts[i] = True                      # a cut to the speaker, not a pan across the room
        xs.append(cur['pos'][i][0]); SIZES.append(cur['pos'][i][2])
    return xs, cuts


def confirm(xs, cuts, win=3, near=130.0):
    """Keep a sighting only if another one within +/-win look-frames (same shot) lands within `near` px:
    one-off detections (a stranger walking past, a poster, a false positive) must not move the camera."""
    out = []
    for i, x in enumerate(xs):
        if x is None:
            out.append(None); continue
        ok = False
        for j in range(max(0, i - win), min(len(xs), i + win + 1)):
            if j == i or xs[j] is None:
                continue
            if any(cuts[k] for k in range(min(i, j) + 1, max(i, j) + 1)):
                continue
            if abs(xs[j] - x) < near:
                ok = True; break
        out.append(x if ok else None)
    return out


def path(xs, cuts, n_out, dead=0.16, sigma_s=0.42, vmax=750.0):
    """Crop-centre x for each output frame."""
    xs = confirm(xs, cuts)
    up = FPS / LOOK_FPS
    # shots: [start, end) in look frames
    bounds = [0] + [i for i, c in enumerate(cuts) if c] + [len(xs)]
    cx = np.full(n_out, SRC_W / 2.0)
    for s0, s1 in zip(bounds, bounds[1:]):
        seg = xs[s0:s1]
        known = [x for x in seg if x is not None]
        if not known:
            continue
        # hold through misses (back-fill the start with the first sighting)
        v, last = [], known[0]
        for x in seg:
            last = x if x is not None else last
            v.append(last)
        v = np.array(v, float)
        # drop single-frame jumps
        v = np.array([np.median(v[max(0, i - 2): i + 3]) for i in range(len(v))])
        # dead zone: the camera only follows once the subject leaves the middle of the frame
        dz = dead * CROP_W
        hold = np.empty_like(v); h = v[0]
        for i, x in enumerate(v):
            if x - h > dz:
                h = x - dz
            elif h - x > dz:
                h = x + dz
            hold[i] = h
        # to output rate, then smooth both ways (no lag) and cap the speed
        o0, o1 = int(round(s0 * up)), min(n_out, int(round(s1 * up)))
        if o1 <= o0:
            continue
        t_out = np.arange(o0, o1) / up - s0
        p = np.interp(t_out, np.arange(len(hold)), hold)
        sig = sigma_s * FPS
        r = int(3 * sig)
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); ker /= ker.sum()
        p = np.convolve(np.pad(p, r, mode='edge'), ker, 'valid')
        cap = vmax / FPS
        for i in range(1, len(p)):
            p[i] = p[i - 1] + np.clip(p[i] - p[i - 1], -cap, cap)
        for i in range(len(p) - 2, -1, -1):
            p[i] = p[i + 1] + np.clip(p[i] - p[i + 1], -cap, cap)
        cx[o0:o1] = p
    return cx


def zoom_curve(n_out, punches, a, z=1.15, attack=0.14, hold=1.1, release=0.3):
    """1.0 everywhere, easing up to z on each punch time (source seconds) and back down."""
    zc = np.ones(n_out)
    t = a + np.arange(n_out) / FPS
    ease = lambda u: u * u * (3 - 2 * u)
    for p in punches:
        up = np.clip((t - (p - attack)) / attack, 0, 1)
        down = np.clip(1 - (t - (p + hold)) / release, 0, 1)
        zc = np.maximum(zc, 1 + (z - 1) * ease(np.minimum(up, down)))
    return zc


def render(ff, video, a, b, audio, a_audio, out, punches=(), threads=None, out_h=1920, hide=(), push=True):
    """out_h: a shorter output (a panel of a split, e.g. 1080x1100) gets a wider window of the same height."""
    global CROP_W, OUT_H
    keep = (CROP_W, OUT_H)
    CROP_W, OUT_H = int(round(SRC_H * OUT_W / out_h / 2) * 2), out_h
    try:
        return _render(ff, video, a, b, audio, a_audio, out, punches, threads, hide, push)
    finally:
        CROP_W, OUT_H = keep


def _render(ff, video, a, b, audio, a_audio, out, punches=(), threads=None, hide=(), push=True):
    """Render [a, b] of video (with audio from a_audio onward) as a 1080x1920 speaker shot that follows the subject."""
    av = look_av(ff, video, a, b, audio, a_audio) if os.path.exists(YUNET) else None
    xs, cuts = av if av else look(ff, video, a, b)
    n_out = int(round((b - a) * FPS))
    cx = path(xs, cuts, n_out)
    zc = zoom_curve(n_out, punches, a) * (size_zoom(cuts, n_out) if av and push else 1.0)   # a physical moment keeps the bodies in
    enc = [ff, '-nostdin', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', '-',
           '-ss', f'{a_audio:.3f}', '-t', f'{b - a:.3f}', '-i', audio, '-map', '0:v', '-map', '1:a',
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-ar', '48000', '-ac', '2', '-b:a', '192k', '-shortest']
    if threads:
        enc += ['-threads', str(threads)]
    enc.append(out)
    p = subprocess.Popen(enc, stdin=subprocess.PIPE)
    i = 0
    for im in frames(ff, video, a, b, SRC_W, SRC_H, FPS):
        if i >= n_out:
            break
        if hide:                                         # stream overlays (chat, counters) blurred out before the crop
            im = im.copy()
            for x_, y_, w_, h_ in hide:
                roi = im[y_:y_ + h_, x_:x_ + w_]
                if roi.size:
                    im[y_:y_ + h_, x_:x_ + w_] = (cv2.GaussianBlur(roi, (0, 0), 22) * 0.8).astype(np.uint8)
        z = zc[i]
        cw, ch = CROP_W / z, SRC_H / z
        x0 = float(np.clip(cx[i] - cw / 2, 0, SRC_W - cw))
        y0 = float(np.clip(SRC_H * 0.42 - ch / 2, 0, SRC_H - ch)) if z > 1.001 else 0.0   # push in toward the upper middle, where faces are
        s = OUT_W / cw
        M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
        p.stdin.write(cv2.warpAffine(im, M, (OUT_W, OUT_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).tobytes())
        i += 1
    p.stdin.close()
    if p.wait():
        raise RuntimeError('vcam encode failed')
    moves = float(np.abs(np.diff(cx)).sum()) if len(cx) > 1 else 0.0
    # the path actually used, at 10 Hz, in source pixels: [t_out, crop_x, crop_y, crop_w, crop_h] (for edit-bay visuals)
    cam = []
    for k in range(0, n_out, FPS // LOOK_FPS):
        z = zc[k]; cw, ch = CROP_W / z, SRC_H / z
        x0 = float(np.clip(cx[k] - cw / 2, 0, SRC_W - cw)); y0 = float(np.clip(SRC_H * 0.42 - ch / 2, 0, SRC_H - ch)) if z > 1.001 else 0.0
        cam.append([round(k / FPS, 2), round(x0, 1), round(y0, 1), round(cw, 1), round(ch, 1)])
    return {'frames': i, 'cuts': int(sum(cuts)), 'seen': round(sum(x is not None for x in xs) / max(1, len(xs)), 2), 'travel_px': round(moves),
            'source': [SRC_W, SRC_H], 'camera': cam, 'punches': [round(p_ - a, 2) for p_ in punches if a <= p_ <= b]}
