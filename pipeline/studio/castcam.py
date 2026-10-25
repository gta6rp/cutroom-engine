#!/usr/bin/env python3
"""Who is on screen, and where the clean picture is, for every source of a story edit.

A story about two people drifts when its shots show someone else (a friend in a car for the last ten seconds of a Lacy
& Marlon edit), and looks cheap when the source's stream overlays come along (chat, a sub-goal counter, alerts). So:

cast(story, pool, k)   faces across every source (YuNet), grouped by appearance (OpenCV SFace embeddings): the k groups
                       with the most screen time are the cast. Nobody is identified; the groups are only "who recurs".
                       Cached in STORY/log/cast.json: {'cast': [group ids], 'per_src': {id: [[t, [group ids]], ...]}}
cast_at(info, src, t)  how many cast members were on screen within a second of t
clean_region(file)     [x, y, w, h] (fractions of the frame): the largest window of the frame's own shape that keeps
                       clear of overlays (text that sits near the frame's edges in every sample while the picture
                       changes), or None when the frame is clean. Cached in STORY/log/regions.json by storyedit.

usage: castcam.py STORY [--k 2]        cast.json + regions.json for a story folder (pool.json), with a contact sheet
"""
import json, os, sys
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
import cv2
import numpy as np

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
_det, _rec = {}, None


def _faces(img):
    """YuNet faces with landmarks on a 960-wide copy, and that copy."""
    global _rec
    h, w = img.shape[:2]; s = 960 / w
    im = cv2.resize(img, (960, int(h * s)))
    key = im.shape[:2]
    if key not in _det:
        _det[key] = cv2.FaceDetectorYN.create(os.path.join(MODELS, 'yunet.onnx'), '', (im.shape[1], im.shape[0]), 0.7, 0.3, 50)
    if _rec is None:
        _rec = cv2.FaceRecognizerSF.create(os.path.join(MODELS, 'sface.onnx'), '')
    _, f = _det[key].detect(im)
    return im, ([] if f is None else [r for r in f if r[2] >= 34])


def _embed(im, row):
    v = _rec.feature(_rec.alignCrop(im, row)).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def cast(story, pool, k=2, max_samples=500, sim=0.40):
    out_f = os.path.join(story, 'log', 'cast.json')
    if os.path.exists(out_f):
        return json.load(open(out_f))
    cents, counts, per_src, crops = [], [], {}, {}
    for it in pool:
        f = os.path.join(story, it['file'])
        c = cv2.VideoCapture(f); fps = c.get(5) or 30; dur = c.get(7) / fps
        step = max(1.5, dur / max_samples)
        rows = []
        t = 0.3
        while t < dur - 0.3:
            c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, img = c.read()
            if not ok:
                break
            im, fs = _faces(img)
            ids = []
            for r in fs:
                e = _embed(im, r)
                if cents:
                    sims = np.array(cents) @ e
                    j = int(np.argmax(sims))
                if cents and sims[j] >= sim:
                    counts[j] += 1
                    cents[j] = cents[j] * 0.9 + e * 0.1; cents[j] /= np.linalg.norm(cents[j])
                else:
                    cents.append(e); counts.append(1); j = len(cents) - 1
                ids.append(j)
                if j not in crops or (r[2] > crops[j][1]):
                    x, y, w, h = (int(v) for v in r[:4])
                    crops[j] = (im[max(0, y - h // 4): y + h + h // 4, max(0, x - w // 4): x + w + w // 4].copy(), r[2])
            rows.append([round(t, 1), sorted(set(ids))])
            t += step
        per_src[it['id']] = rows
    # groups the running centroids split in two get merged
    order = np.argsort(counts)[::-1]
    alias = {}
    for a_ in order:
        for b_ in order:
            if b_ == a_ or b_ in alias or a_ in alias:
                continue
            if counts[b_] <= counts[a_] and float(cents[a_] @ cents[b_]) >= sim + 0.05:
                alias[int(b_)] = int(a_)
    tot = {}
    for j, n in enumerate(counts):
        tot[alias.get(j, j)] = tot.get(alias.get(j, j), 0) + n
    top = [j for j, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:k]]
    for rows in per_src.values():
        for r in rows:
            r[1] = sorted({alias.get(j, j) for j in r[1]})
    info = {'cast': top, 'counts': {str(j): n for j, n in sorted(tot.items(), key=lambda kv: -kv[1])[:8]}, 'per_src': per_src}
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    json.dump(info, open(out_f, 'w'))
    sheet = [cv2.resize(crops[j][0], (160, 200)) for j in top + [j for j in sorted(tot, key=lambda j: -tot[j]) if j not in top][:4] if j in crops and crops[j][0].size]
    if sheet:
        cv2.imwrite(os.path.join(story, 'log', 'cast.jpg'), np.hstack(sheet))
    return info


def cast_at(info, src, t, near=1.0):
    rows = info['per_src'].get(src) or []
    cs = set(info['cast'])
    best = 0
    for tt, ids in rows:
        if abs(tt - t) <= near:
            best = max(best, len(cs & set(ids)))
    return best


def clean_region(f, n=40, max_zoom=1.45, t0=None, t1=None):
    """The largest window of the frame's shape clear of overlays, as fractions [x, y, w, h], or None if nothing to avoid."""
    c = cv2.VideoCapture(f); N = c.get(7); fps = c.get(5) or 30
    if N < 10:
        return None
    lo, hi = (0, N) if t0 is None else (max(0, int(t0 * fps)), min(N, int(t1 * fps)))   # one shot's stretch (a compilation moves its overlays)
    G, E = [], []
    for k in range(n):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(lo + (hi - lo) * (k + 0.5) / n)); ok, img = c.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2GRAY)
        G.append(g.astype(np.float32))
        e = cv2.dilate(cv2.Canny(g, 80, 160), np.ones((3, 3), np.uint8))
        E.append(cv2.resize((e > 0).astype(np.float32), (60, 34), interpolation=cv2.INTER_AREA))
    if len(G) < 8:
        return None
    G = np.array(G); E = np.array(E)
    busy = (E > 0.14).mean(axis=0)                                              # text-like in most samples
    chg = cv2.resize(np.abs(np.diff(G, axis=0)).mean(axis=0), (60, 34), interpolation=cv2.INTER_AREA)
    moving = float(np.median(chg))
    yy, xx = np.mgrid[0:34, 0:60]
    band = (xx < 18) | (xx >= 42) | (yy < 9) | (yy >= 28)                       # overlays hug the frame's edges
    chat = (busy >= 0.7) & (chg > max(4.0, 1.4 * moving)) & band                # text that keeps changing: a chat
    ui = (busy >= 0.9) & (chg < 0.5 * moving) & band if moving > 6 else np.zeros_like(band)   # still text over a moving picture
    mask = (chat | ui).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [stats[i][:4] for i in range(1, nlab) if stats[i][4] >= 6]          # >= 6 cells (~0.3% of the frame)
    if not boxes:
        return None
    occ = np.zeros((34, 60), np.float32)
    for x, y, w, h in boxes:
        occ[y:y + h, x:x + w] = 1
    best = None
    for s in np.arange(1.0, 1 / max_zoom - 1e-6, -0.025):
        w, h = int(round(60 * s)), int(round(34 * s))
        for y in range(0, 34 - h + 1):
            for x in range(0, 60 - w + 1):
                o = occ[y:y + h, x:x + w].mean()
                if o <= 0.01:
                    cost = abs(x + w / 2 - 30) + abs(y + h / 2 - 17) * 0.5
                    if best is None or cost < best[0]:
                        best = (cost, x, y, w, h)
        if best:
            break
    if not best:
        return None
    _, x, y, w, h = best
    if w >= 59:
        return None
    return [round(x / 60, 4), round(y / 34, 4), round(w / 60, 4), round(h / 34, 4)]


if __name__ == '__main__':
    story = sys.argv[1]; k = int(sys.argv[sys.argv.index('--k') + 1]) if '--k' in sys.argv else 2
    pool = json.load(open(os.path.join(story, 'pool.json')))
    info = cast(story, pool, k)
    print('cast groups', info['cast'], 'screen time', info['counts'])
    regs = {}
    for it in pool:
        regs[it['id']] = clean_region(os.path.join(story, it['file']))
        print(it['id'], regs[it['id']])
    json.dump(regs, open(os.path.join(story, 'log', 'regions.json'), 'w'), indent=1)


OVERLAY_RULES = """This is one frame of a livestream recording. List every stream OVERLAY drawn on top of the camera picture:
live chat, viewer / sub / follower counters, goal bars, donation or alert boxes, timers, channel logos and watermarks,
a small webcam box, burned-in captions from another page. NOT real things in the room (posters, TVs, signs, screens).
Give each as a generous box (cover the whole overlay) in fractions of the frame: [x0, y0, x1, y1], origin top-left.
Answer with JSON only: {"boxes": [{"kind": "chat|counter|alert|logo|webcam|caption|other", "box": [x0, y0, x1, y1]}]}"""


def overlay_boxes(f, t, cache=None):
    """The overlays on the frame at t (the editor reads it), cached by file and second."""
    key = f"{os.path.basename(f)}@{t:.0f}"
    if cache is not None and key in cache:
        return cache[key]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'live'))
    import judge as J, re
    c = cv2.VideoCapture(f); c.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000); ok, img = c.read()
    if not ok:
        return []
    h, w = img.shape[:2]; s = 960 / max(h, w)
    jpg = cv2.imencode('.jpg', cv2.resize(img, (int(w * s), int(h * s))), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    try:
        raw = J._ask(OVERLAY_RULES, [jpg], 90)
        boxes = [b for b in json.loads(re.search(r'\{.*\}', raw, re.S).group(0)).get('boxes') or []
                 if len(b.get('box') or []) == 4 and all(0 <= v <= 1 for v in b['box'])]
    except Exception:  # noqa: BLE001
        boxes = []
    if cache is not None:
        cache[key] = boxes
    return boxes


def clean_window(boxes, max_zoom=1.6, pad=0.01, faces=()):
    """The largest window of the frame's own shape that avoids every overlay box (a little padding), as [x, y, w, h]
    fractions, preferring the centre; overlays that can't be avoided within max_zoom are allowed in, smallest first.
    None when nothing needs avoiding."""
    bx = [[max(0, b['box'][0] - pad), max(0, b['box'][1] - pad), min(1, b['box'][2] + pad), min(1, b['box'][3] + pad)] for b in boxes]
    bx = [b for b in bx if (b[2] - b[0]) * (b[3] - b[1]) < 0.35]          # a "box" covering most of the frame is a misread
    if not bx:
        return None
    bx.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    for keep in range(len(bx), 0, -1):                                    # drop the smallest boxes if all can't be avoided
        avoid = bx[:keep]
        best = None
        for s in np.arange(1.0, 1 / max_zoom - 1e-6, -0.02):
            for y in np.arange(0, 1 - s + 1e-6, 0.01):
                for x in np.arange(0, 1 - s + 1e-6, 0.01):
                    if all(x >= b[2] or x + s <= b[0] or y >= b[3] or y + s <= b[1] for b in avoid):
                        # people stay in: every face whole inside the window (a face cut off costs more than any offset)
                        cut = sum(1 for f in faces if not (f[0] >= x and f[2] <= x + s and f[1] >= y and f[3] <= y + s))
                        cost = 10 * cut + abs(x + s / 2 - 0.5) + abs(y + s / 2 - 0.5) * 0.7
                        if best is None or cost < best[0]:
                            best = (cost, x, y, s)
            if best and best[0] < 10:                                     # nobody cut off at this size: take it
                break
        if best:
            _, x, y, s = best
            return None if s >= 0.995 else [round(float(x), 4), round(float(y), 4), round(float(s), 4), round(float(s), 4)]
    return None


def faces_frac(f, t):
    """Face boxes on the frame at t, as [x0, y0, x1, y1] fractions (with hair and chin room)."""
    c = cv2.VideoCapture(f); c.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000); ok, img = c.read()
    if not ok:
        return []
    im, fs = _faces(img)
    H, W = im.shape[:2]
    return [[max(0, (r[0] - 0.2 * r[2]) / W), max(0, (r[1] - 0.35 * r[3]) / H), min(1, (r[0] + 1.2 * r[2]) / W), min(1, (r[1] + 1.15 * r[3]) / H)]
            for r in fs if r[2] >= 0.03 * W]


def faces_over(f, a, b, n=4):
    """Face boxes (fractions, with hair and chin room) on n frames across [a, b]: one frame misses a face in profile."""
    out = []
    for t in np.linspace(a, b, n):
        out += faces_frac(f, float(t))
    return out


def regions_or_hide(story, shots, workers=4):
    """Like regions_for, but a crop that would cut a face on any of four frames across the shot is dropped and the
    overlays are blurred instead (shot['hide']): Kai's corner webcam sat right above his head, and the clean window
    started below it. Returns (cropped, blurred) counts."""
    from concurrent.futures import ThreadPoolExecutor
    cf = os.path.join(story, 'log', 'overlays.json')
    cache = json.load(open(cf)) if os.path.exists(cf) else {}
    def one(sh):
        t = sh['in'] + min(0.6, sh['dur'] / 2)
        span = sh['dur'] * (min(x[1] for x in sh['speed']) if isinstance(sh.get('speed'), list) else (sh.get('speed') or 1.0))
        bs = list(overlay_boxes(sh['src'], t, cache))
        if abs(span) > 1.5:                              # alerts pop in mid-shot: a second read near the end (AURA kept a chat box)
            bs += [b for b in overlay_boxes(sh['src'], sh['in'] + abs(span) - 0.4, cache)
                   if not any(max(abs(b['box'][i] - c['box'][i]) for i in range(4)) < 0.03 for c in bs)]
        if not bs:
            return None, None
        fs = faces_over(sh['src'], sh['in'], sh['in'] + max(0.2, abs(span)))
        r = clean_window(bs, faces=fs)
        if r and all(f[0] >= r[0] - 0.01 and f[2] <= r[0] + r[2] + 0.01 and f[1] >= r[1] - 0.01 and f[3] <= r[1] + r[3] + 0.01 for f in fs):
            return r, None
        boxes = [b['box'] for b in bs if (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1]) < 0.35]
        return None, boxes or None
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, shots))
    for sh, (r, hide) in zip(shots, res):
        if r:
            sh['region'] = r
        elif hide:
            sh['hide'] = hide
    json.dump(cache, open(cf, 'w'))
    return sum(1 for r, _ in res if r), sum(1 for _, h in res if h)


def regions_for(story, shots, workers=4):
    """shot['region'] for every shot whose source frame carries overlays (chat, counters, another page's captions):
    the largest clean window that keeps the people in. Overlay reads are cached in STORY/log/overlays.json."""
    from concurrent.futures import ThreadPoolExecutor
    cf = os.path.join(story, 'log', 'overlays.json')
    cache = json.load(open(cf)) if os.path.exists(cf) else {}
    def one(sh):
        t = sh['in'] + min(0.6, sh['dur'] / 2)
        bs = overlay_boxes(sh['src'], t, cache)
        return clean_window(bs, faces=faces_frac(sh['src'], t)) if bs else None
    with ThreadPoolExecutor(workers) as ex:
        regs = list(ex.map(one, shots))
    for sh, r in zip(shots, regs):
        if r:
            sh['region'] = r
    json.dump(cache, open(cf, 'w'))
    return sum(1 for r in regs if r)
