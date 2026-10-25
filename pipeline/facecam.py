#!/usr/bin/env python3
"""Find a facecam overlay: a small webcam box laid over gameplay, a browser or a chat page (xQc, Jynxzi, Asmongold...).

A face that stays put, is small against the frame, and sits inside straight static borders is an overlay, not a camera
filming the room. Returns the cam box grown or trimmed to 16:9 (the top panel of the facecam-over-screen layout),
plus where the content is (the side away from the cam), or None for a full-frame camera.

usage: facecam.py <video> [a b]      prints the box and writes facecam-debug.jpg next to the video
"""
import json, os, sys
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
import cv2
import numpy as np

FRONT = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')


def frames(video, a, b, n=10):
    cap = cv2.VideoCapture(video)
    out = []
    for t in np.linspace(a + 0.3, b - 0.3, n):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if ok:
            out.append(img)
    cap.release()
    return out


YUNET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'yunet.onnx')
_yn = {}


def faces_yunet(img, scale):
    """OpenCV's YuNet (a small CNN): far better than Haar on headsets, side light, dark rooms."""
    h, w = img.shape[:2]
    W, H = int(w * scale), int(h * scale)
    key = (W, H)
    if key not in _yn:
        _yn[key] = cv2.FaceDetectorYN.create(YUNET, '', (W, H), 0.6, 0.3, 50)
    _, f = _yn[key].detect(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
    return [] if f is None else [(x / scale, y / scale, fw / scale, fh / scale) for x, y, fw, fh, *_ in f if fw >= 12]


def faces(img, scale):
    if os.path.exists(YUNET):
        try:
            return faces_yunet(img, scale)
        except cv2.error:
            pass
    return faces_haar(img, scale)


def faces_haar(img, scale):
    g = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    found = list(FRONT.detectMultiScale(g, 1.1, 6, minSize=(22, 22)))
    if not found:
        found = list(PROFILE.detectMultiScale(g, 1.1, 6, minSize=(22, 22))) + \
            [(g.shape[1] - x - w, y, w, h) for x, y, w, h in PROFILE.detectMultiScale(cv2.flip(g, 1), 1.1, 6, minSize=(22, 22))]
    return [(x / scale, y / scale, w / scale, h / scale) for x, y, w, h in found]


def static_edges(imgs):
    """Median edge strength over the samples: overlay borders stay, moving content washes out."""
    gx, gy = [], []
    for img in imgs:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx.append(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)))
        gy.append(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)))
    return np.median(gx, axis=0), np.median(gy, axis=0)


def border(profile, lo, hi, toward):
    """Strongest static line in [lo, hi), the nearest-to-face one when several are close in strength."""
    lo, hi = max(0, int(lo)), min(len(profile), int(hi))
    if hi - lo < 3:
        return None
    seg = profile[lo:hi]
    peak = float(seg.max())
    if peak < 18:                      # nothing straight and still: the cam runs to the frame edge
        return None
    idx = np.where(seg >= 0.7 * peak)[0] + lo
    return int(idx.min() if toward == 'right' else idx.max())


def box_for(near, ex, ey, W, H):
    fx, fy, fw, fh = (float(np.median([f[k] for f in near])) for k in range(4))
    cxs = [f[0] + f[2] / 2 for f in near]
    if fw > 0.13 * W or np.std(cxs) > 0.03 * W:
        return None                    # a big or wandering face: a camera filming the room, not an overlay
    # a border is a straight line along (nearly) the whole side of the box: use a low percentile along the line, so
    # static things inside the cam (shelves, a mic arm) and short UI lines beside it don't count
    rows = slice(int(max(0, fy - 1.2 * fh)), int(min(H, fy + 2.2 * fh)))
    cols = slice(int(max(0, fx - 1.2 * fw)), int(min(W, fx + 2.2 * fw)))
    colp = np.percentile(ex[rows], 30, axis=0)
    rowp = np.percentile(ey[:, cols], 30, axis=1)
    right = border(colp, fx + fw * 1.15, fx + fw * 5, 'right')
    left = border(colp, fx - fw * 4, fx - fw * 0.15, 'left')
    bottom = border(rowp, fy + fh * 1.25, fy + fh * 4, 'right')
    top = border(rowp, fy - fh * 3, fy - fh * 0.1, 'left')
    x0 = left if left is not None else max(0, int(fx - 2.5 * fw))
    x1 = right if right is not None else min(W, int(fx + 3.5 * fw))
    y0 = top if top is not None else max(0, int(fy - 1.6 * fh))
    y1 = bottom if bottom is not None else min(H, int(fy + 2.8 * fh))
    n_borders = sum(v is not None for v in (left, right, top, bottom))
    edge_touch = (x0 <= 6) + (x1 >= W - 6) + (y0 <= 6) + (y1 >= H - 6)
    if n_borders < 1 or n_borders + edge_touch < 3:
        return None                    # no box around the face
    bw, bh = x1 - x0, y1 - y0
    if bw * bh > 0.45 * W * H or bw < 2.6 * fw or bh < 1.8 * fh:
        return None
    outer = [int(x0), int(y0), int(x1), int(y1)]      # the whole cam box (gamecam.py re-cuts it to its own panel)
    # inside the box, to 16:9 around the face (the panel is 16:9; never stretch)
    cx, cy = fx + fw / 2, fy + fh / 2
    if bw / bh > 16 / 9:
        nw = bh * 16 / 9; x0 = min(max(x0, cx - nw / 2), x1 - nw); x1 = x0 + nw
    else:
        nh = bw * 9 / 16; y0 = min(max(y0, cy - nh * 0.45), y1 - nh); y1 = y0 + nh
    box = {k: int(round(v / 2) * 2) for k, v in (('x', x0), ('y', y0), ('w', x1 - x0), ('h', y1 - y0))}
    box['side'] = 'left' if cx < W / 2 else 'right'
    box['src'] = [W, H]
    box['content_x'] = [int(x1), W] if box['side'] == 'left' else [0, int(x0)]
    box['score'] = n_borders + 2 * edge_touch + len(near) / 10   # real cams sit against the frame edge; a face in the content floats
    box['face'] = (fx, fy, fw, fh)
    box['outer'] = outer
    return box


def content_crop(imgs, box, W, H, out_w=1080, out_low=1312):
    """Where the lower panel looks: the window beside the cam with the most motion (the clip or video being reacted
    to), leaning to the middle of the content when nothing moves. Source pixels, aspect of the lower panel."""
    ch = int(round(H * 0.83 / 2) * 2)
    cw = int(round(ch * out_w / out_low / 2) * 2)
    lo, hi = box['content_x']
    if hi - lo < cw:
        lo, hi = (0, W) if box['side'] == 'left' else (0, W)
    small = [cv2.cvtColor(cv2.resize(im, (W // 4, H // 4)), cv2.COLOR_BGR2GRAY).astype(np.float32) for im in imgs]
    diffs = [np.abs(small[i + 1] - small[i]) for i in range(len(small) - 1)]
    # per column: motion that runs most of the height (a game's camera, a video) counts; a scrolling chat or a small guest
    # cam moves only some rows, so it barely registers at the 25th percentile
    motion = np.mean([np.percentile(d, 25, axis=0) + 0.15 * d.mean(axis=0) for d in diffs], axis=0)
    motion = np.repeat(motion, 4)[:W]
    motion = motion * (1 - overlays(imgs, small, box, W))   # chat and guest cams move all the time but aren't the content
    edge = np.ones(W, np.float32); edge[: int(0.22 * W)] = 0.1; edge[int(0.78 * W):] = 0.1
    motion = motion * edge                              # overlays (chat, alerts, guest cams) hug the edges; content sits mid-frame
    # lean toward the middle of the whole frame (a game's crosshair and player, a centred video), as far as the content allows
    mid = min(max(W / 2, lo + cw / 2), hi - cw / 2)
    best, best_x = -1, int(mid - cw / 2)
    for x in range(lo, hi - cw + 1, 8):
        # scrolling chat is all motion: lean hard toward the middle of the content, where the game or the video is
        m = motion[x:x + cw].mean() * (1 - 0.7 * abs(x + cw / 2 - mid) / max(1, (hi - lo) / 2))
        if m > best:
            best, best_x = m, x
    col = column(imgs, lo, hi, cw, motion)
    if col is not None:                     # a web page (X, Reddit, a news site): frame its content column whole
        best_x = int(min(max(lo, col - cw // 2), hi - cw))
    y = (H - ch) // 2
    vb = video_box(small, lo, hi, W, H, out_w / out_low)
    if vb:                                  # a video playing inside a page: the player, not the page around it
        return vb
    return {'x': int(best_x), 'y': int(y), 'h': ch}


def video_box(small, lo, hi, W, H, asp):
    """A video playing inside a still page (an X post, a YouTube page, a Discord window): the rectangle where nearly all
    the motion is, padded and cut to the panel's shape (asp = width/height), as {x, y, h} in source pixels; None when
    the motion fills the screen (a game, a full-screen video) or there is no clear player."""
    if len(small) < 4:
        return None
    m = np.mean([np.abs(small[i + 1] - small[i]) for i in range(len(small) - 1)], axis=0)
    m = cv2.GaussianBlur(m, (0, 0), 3)
    hs, ws = m.shape
    m[:, : lo // 4] = 0; m[:, hi // 4:] = 0                # only the content side, not the cam
    if m.max() < 4:
        return None
    mask = (m > 0.35 * m.max()).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    x, y, w, h, area = st[k]
    share = float(m[lab == k].sum() / max(1e-6, m.sum()))
    frac = (w * h) / float(hs * (hi - lo) / 4)
    if share < 0.6 or frac > 0.5 or w < 0.1 * ws or h < 0.1 * hs:   # motion everywhere (a game) or no single player
        return None
    x, y, w, h = x * 4, y * 4, w * 4, h * 4
    cx, cy = x + w / 2, y + h / 2
    ch = max(h * 1.12, w * 1.12 / asp)
    ch = int(min(H * 0.95, max(ch, H * 0.35)) // 2 * 2); cw = int(round(ch * asp / 2) * 2)
    x0 = int(min(max(0, cx - cw / 2), W - cw)); y0 = int(min(max(0, cy - ch / 2), H - ch))
    return {'x': x0, 'y': y0, 'h': ch, 'player': [int(x), int(y), int(w), int(h)]}


def overlays(imgs, small, box, W):
    """Per source column, 1 where an overlay sits: a scrolling chat (a band whose next frame is itself shifted up) or
    another person's cam (a face outside the main facecam box)."""
    mask = np.zeros(W, np.float32)
    hs, ws = small[0].shape
    band = 16                                           # 64 source px
    for x0 in range(0, ws - band + 1, band):
        if 0.22 * ws < x0 + band / 2 < 0.78 * ws:     # chat overlays hug an edge; a feed scrolling mid-frame is the content
            continue
        votes = 0
        for a_, b_ in zip(small, small[1:]):
            A, B = a_[:, x0:x0 + band], b_[:, x0:x0 + band]
            still = float(np.abs(B - A).mean())
            if still < 2:
                continue
            shifted = min(float(np.abs(B[:-k] - A[k:]).mean()) for k in range(2, 16))
            if shifted < 0.55 * still:
                votes += 1
        if votes >= max(2, 0.3 * (len(small) - 1)):
            mask[x0 * 4: (x0 + band) * 4] = 1
    cx0, cx1 = box['x'], box['x'] + box['w']
    for img in imgs[:: max(1, len(imgs) // 4)]:
        for x, y, w, h in faces(img, 640 / img.shape[1]):
            fx = x + w / 2
            if not (cx0 - 20 <= fx <= cx1 + 20) and not (0.22 * W < fx < 0.78 * W):   # someone else's cam, in a corner
                mask[int(max(0, x - w)): int(min(W, x + 2 * w))] = 1
    return mask


def column(imgs, lo, hi, cw, motion):
    """Centre of a page column that fits the crop: two long static vertical borders 45-100% of the crop width apart
    (the feed of X, a Reddit thread, an article), the one with the most going on inside, near the middle. None when the
    content has no such borders (a game, a full video)."""
    half = [cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2)) for im in imgs[:8]]
    gx, _ = static_edges(half)
    h = gx.shape[0]
    frac = (gx[int(0.12 * h): int(0.9 * h)] > 30).mean(axis=0)          # how much of the height each column's edge runs
    lines, run = [], []
    for x in range(lo // 2, min(hi // 2, len(frac))):
        if frac[x] > 0.6:
            run.append(x)
        elif run:
            lines.append(2 * int(np.mean(run))); run = []
    if run:
        lines.append(2 * int(np.mean(run)))
    best, pick = 0.0, None
    mid = (lo + hi) / 2
    dens = (gx > 30).mean(axis=0)                                       # text and pictures inside the column
    mn = motion / max(1e-6, float(motion[lo:hi].max()))                 # what moves (the video being reacted to) counts most
    for i, L in enumerate(lines):
        for R in lines[i + 1:]:
            w = R - L
            if not 0.45 * cw <= w <= cw:
                continue
            inside = float(dens[L // 2: R // 2].mean()) + 1.5 * float(mn[L:R].mean())
            # the main column is the wide one (a feed or a thread), not a sidebar of boxes
            sc = inside * (w / cw) * (1 - 0.5 * abs((L + R) / 2 - mid) / max(1, (hi - lo) / 2))
            if sc > best:
                best, pick = sc, (L + R) // 2
    return pick


def screen_layout(video, a, b, out_w=1080, out_h=980):
    """No facecam: is this a screen (a trading terminal, a game, a page) rather than an IRL camera? If so, the crop for the
    centred layout: the window where things move (the chart, the play), leaning to the middle. None for camera footage."""
    imgs = frames(video, a, b)
    if len(imgs) < 4:
        return None
    H, W = imgs[0].shape[:2]
    big = 0
    for img in imgs:
        if any(h >= 0.12 * H for x, y, w, h in faces(img, 640 / W)):
            big += 1
    if big >= 0.3 * len(imgs):                          # people on camera: the speaker crop follows them instead
        return None
    ch = int(round(H * 0.9 / 2) * 2); cw = int(round(ch * out_w / out_h / 2) * 2)
    small = [cv2.cvtColor(cv2.resize(im, (W // 4, H // 4)), cv2.COLOR_BGR2GRAY).astype(np.float32) for im in imgs]
    diffs = [np.abs(small[i + 1] - small[i]) for i in range(len(small) - 1)]
    motion = np.repeat(np.mean([np.percentile(d, 25, axis=0) + 0.15 * d.mean(axis=0) for d in diffs], axis=0), 4)[:W]
    motion = motion * (1 - overlays(imgs, small, {'x': -10 ** 6, 'w': 0}, W))   # scrolling feeds and chats aren't the content
    edge = np.ones(W, np.float32); edge[: int(0.22 * W)] = 0.1; edge[int(0.78 * W):] = 0.1
    motion = motion * edge
    best, bx = -1.0, (W - cw) // 2
    for x in range(0, W - cw + 1, 16):
        m = motion[x:x + cw].mean() * (1 - 0.6 * abs(x + cw / 2 - W / 2) / (W / 2))
        if m > best:
            best, bx = m, x
    return {'x': int(bx), 'y': int((H - ch) // 2), 'h': ch}


def detect(video, a=0.0, b=None, debug=False):
    if b is None:
        cap = cv2.VideoCapture(video); b = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS)); cap.release()
    imgs = frames(video, a, b)
    if len(imgs) < 4:
        return None
    H, W = imgs[0].shape[:2]
    scale = 960 / W
    dets = [(i, f) for i, img in enumerate(imgs) for f in faces(img, scale)]
    if not dets:
        return None
    # faces that keep coming back to the same place (a react stream can show a second face in the content)
    clusters, used = [], set()
    for k, (_, (x, y, w, h)) in sorted(enumerate(dets), key=lambda kv: -kv[1][1][2]):
        if k in used:
            continue
        cx, cy = x + w / 2, y + h / 2
        near = [j for j, (_, f) in enumerate(dets) if j not in used and abs(f[0] + f[2] / 2 - cx) < 0.05 * W and abs(f[1] + f[3] / 2 - cy) < 0.07 * H]
        used |= set(near)
        hits = len({dets[j][0] for j in near})
        if hits >= 0.4 * len(imgs):
            clusters.append([dets[j][1] for j in near])
    if not clusters:
        return None
    ex, ey = static_edges(imgs)
    # a real facecam box is at least ~13% of the frame wide; smaller "cams" are faces in the content (a photo in a feed)
    boxes = [bx for c in clusters if (bx := box_for(c, ex, ey, W, H)) and bx['w'] >= 0.13 * W]
    if not boxes:
        return None
    box = max(boxes, key=lambda bx: bx['score'])
    fx, fy, fw, fh = box.pop('face')
    box['screen'] = content_crop(imgs, box, W, H)
    if debug:
        dbg = imgs[len(imgs) // 2].copy()
        cv2.rectangle(dbg, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (0, 214, 255), 4)
        cv2.rectangle(dbg, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (0, 255, 0), 2)
        sc = box['screen']; scw = int(round(sc['h'] * 1080 / 1312))
        cv2.rectangle(dbg, (sc['x'], sc['y']), (sc['x'] + scw, sc['y'] + sc['h']), (255, 120, 0), 4)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(video)), 'facecam-debug.jpg'), cv2.resize(dbg, (960, int(960 * H / W))))
    return box


if __name__ == '__main__':
    v = sys.argv[1]
    a, b = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (0.0, None)
    print(json.dumps(detect(v, a, b, debug=True)))
