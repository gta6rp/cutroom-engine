#!/usr/bin/env python3
"""Which layout a cut gets, shared by live rooms (live.py) and cuts from past streams (clipcut.py).

  a game with facecams   -> gamesplit (gamecam.py): the game on top at near full height, the cam below, the hook on the
                            seam; with a duo or squad on cam, the lower panel cuts to whoever is talking or reacting.
  a still sit-down talk  -> fillersplit (half the time): the speaker camera over our own satisfying loop
  a small facecam        -> pip: the content in the middle over a blurred copy, the cam as a rounded inset
  facecam over a screen  -> hooksplit: the cam on top, the screen below, the hook on the seam. For traders the seam moves
                            with the moment: half and half while they explain something, a smaller cam when the moment is
                            a trade (more room for the chart), per the Top Floor Clippers video guide.
  no facecam, two people far apart -> letterbox: the whole frame with bars, the hook as plain text on the video.
  no facecam, a screen   -> center: the footage zoomed in the middle over a blurred copy, the hook above it.
  no facecam, a camera   -> speaker: the virtual camera follows the people (IRL).
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
# per-streamer layout notes (live/profiles.json), e.g. {"dvces": {"inset": [x, y, w, h]}}: a part of their stream layout
# worth keeping in view (a balance/PnL box), in 1920x1080 source pixels
try:
    PROFILES = json.load(open(os.path.join(HERE, 'live', 'profiles.json')))
except (OSError, ValueError):
    PROFILES = {}

# streamers whose clips are trading clips (memecoin / crypto traders)
TRADERS = {'dvces', 'jackduval', 'tjr', 'tjrtrades', 'orangie', 'cupsey', 'ansem', 'cented', 'frankdegods'}
TRADE_WORDS = re.compile(r"\b(buy|bought|buying|sell|sold|selling|ape[ds]?|aping|entry|entries|exit|chart|pump(ed|ing)?|dump(ed|ing)?|bond(ed|ing)?|"
                         r"rug(ged)?|mcap|market cap|supply|liquidity|sniped?|bag|bags|position|profit|pnl|sol|solana|\d+x|\d+k|"
                         r"up \d+|down \d+|million|mil|ticker|coin|token)\b", re.I)


def is_trade(text):
    """A trade moment: the talk is about buying, selling and the chart, not an explanation or a story."""
    return len(TRADE_WORDS.findall(text or '')) >= 4


IRL_CATS = {'irl', 'travel & outdoors', 'food & drink', 'sports', 'pools, hot tubs & bikinis', 'pools, hot tubs & beaches'}


def wide_scene(video, a, b):
    """Two or three sizeable faces (>= 11% of the frame height) far apart in most sampled frames: a conversation a 9:16
    crop can't hold. A crowd of small faces isn't one: the letterbox would show a strip of tiny people, so the speaker
    camera takes it (following whoever talks)."""
    import cv2, facecam
    cap = cv2.VideoCapture(video); wide = n = 0
    for t in [a + (b - a) * k / 5 for k in range(6)]:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000); ok, img = cap.read()
        if not ok:
            continue
        n += 1
        H, W = img.shape[:2]
        allf = [f for f in facecam.faces(img, 960 / W) if f[3] >= 0.045 * H]
        fs = [f for f in allf if f[3] >= 0.14 * H]
        if len(fs) == 2 and len(allf) <= 3:               # two big faces: a group or small faces get the speaker camera
            xs = [x + w / 2 for x, y, w, h in fs]
            wide += (max(xs) - min(xs)) > 0.38 * W
    cap.release()
    return n > 0 and wide >= n / 2


TALK_CATS = {'just chatting', 'talk shows & podcasts', 'podcasts'}


def still_camera(video, a, b, n=8):
    """A camera on a tripod (a sit-down conversation), not a handheld IRL walk: frame-to-frame change stays small."""
    import cv2, numpy as np
    cap = cv2.VideoCapture(video); g, ds = None, []
    for t in [a + (b - a) * k / n for k in range(n + 1)]:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t) * 1000); ok, img = cap.read()
        if not ok:
            continue
        x = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if g is not None:
            ds.append(float(np.median(np.abs(x - g))))      # the median ignores the people moving; a moving camera shifts everything
        g = x
    cap.release()
    return bool(ds) and float(np.median(ds)) < 6


CENTER_Y, CENTER_H = 500, 980                    # edit.py's centred panel


def ui_content(src, a, b, cam):
    """True when the 'content' beside a facecam is a flat dark page (the streamer's own chat, a dashboard, a browser
    tab), not a video or a game: a hooksplit then filled half the frame with chat (xQc, 2026-10-09). Flat = a large
    share of the panel sits in one dark grey level."""
    import cv2, numpy as np
    sc = (cam or {}).get('screen') or {}
    if not sc:
        return False
    cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS) or 30; W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    x0 = int(sc.get('x', 0)); y0 = int(sc.get('y', 0)); hh = int(sc.get('h', 0)); x1 = min(W, x0 + int(sc.get('w', W - x0)))
    flat = []
    for t in np.linspace(a, b, 5)[1:-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, img = cap.read()
        if not ok or hh < 50 or x1 - x0 < 50:
            continue
        g = cv2.cvtColor(img[y0:y0 + hh, x0:x1], cv2.COLOR_BGR2GRAY)
        hist = np.bincount((g // 4).ravel(), minlength=64)
        k = int(np.argmax(hist))
        flat.append(k * 4 < 64 and hist[max(0, k - 1):k + 2].sum() / g.size > 0.45)
    cap.release()
    return bool(flat) and sum(flat) >= max(1, len(flat) - 1)


def player_page(src, a, b, cam=None):
    """True when the motion sits in a player inside a still page (the streamer watching a video or another stream in a
    browser), not filling the screen like a game: Kick's category lags (xQc "in Age of Empires IV" while watching
    Mizkif's stream got a gamesplit of a blurred Twitch page)."""
    import cv2, numpy as np, facecam
    imgs = facecam.frames(src, a, b)
    if len(imgs) < 4:
        return False
    H, W = imgs[0].shape[:2]
    small = [cv2.cvtColor(cv2.resize(im, (W // 4, H // 4)), cv2.COLOR_BGR2GRAY).astype(np.float32) for im in imgs]
    if cam:                                             # the streamer's own cam moves too: not the player
        sx = (W / 4) / ((cam.get('src') or [W])[0])     # cam boxes are in source pixels
        x0, y0 = int(cam['x'] * sx) - 2, int(cam['y'] * sx) - 2
        for g in small:
            g[max(0, y0):y0 + int(cam['h'] * sx) + 4, max(0, x0):x0 + int(cam['w'] * sx) + 4] = 0
    vb = facecam.video_box(small, 0, W, W, H, 1080 / 1312)
    return bool(vb) and vb['h'] < 0.8 * H


# streams that burn in their own live captions (Marlon's purple word highlight): ours on top would double them
BURNED_CAPS = {'marlon'}


def _people(src, a, b, n=8):
    """(left, right) of everyone in [a, b] (faces, profiles, standing bodies) in 1920-wide pixels, and whether it's one
    close face throughout (one face, at least ~18% of the frame wide). Ported from video/clav/groupfill.py (experiment #2)."""
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    prof = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    ls, rs, close = [], [], 0
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((a + (b - a) * (k + 0.5) / n) * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(fr, (960, int(960 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
        faces = list(cas.detectMultiScale(g, 1.1, 4, minSize=(36, 36)))
        f = faces + list(prof.detectMultiScale(g, 1.1, 4, minSize=(36, 36)))
        f += [(960 - x - w, y, w, h) for x, y, w, h in prof.detectMultiScale(cv2.flip(g, 1), 1.1, 4, minSize=(36, 36))]
        f += [tuple(r) for r, wt in zip(*hog.detectMultiScale(g, winStride=(8, 8), scale=1.05)) if float(wt) > 0.6]
        if len(f):
            ls.append(min(x for x, y, w, h in f) * 2); rs.append(max(x + w for x, y, w, h in f) * 2)
        if len(faces) == 1 and faces[0][2] >= 0.18 * 960 and len(f) <= 2:
            close += 1
    cap.release()
    return ((int(min(ls)), int(max(rs))) if ls else (480, 1440)), close >= max(3, int(n * 0.75))   # plain ints: the spec is JSON


def irl_fill(src, a, b, slug):
    """Experiment #2 (the user, 2026-10-21: "we should not always crop too much, it makes the quality pixelated"), after
    @m3lgtheworld's 2.7M-view Marlon clip: the whole 16:9 stream at full width over a blurred fill, the hook in its white box
    on top, small word captions just under the picture. A 1.33x zoom (a 4:3 window) only on a 1080p source where everyone
    still fits; the tight 9:16 speaker crop only for one close face on a 1080p source."""
    import cv2
    cap = cv2.VideoCapture(src); h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0; cap.release()
    (l, r), one_close = _people(src, a, b)
    if one_close and h >= 1080:
        return 'speaker', {}
    ex = {'irl_fill': True, 'tighten': True, 'captions': 'none' if (slug or '').lower() in BURNED_CAPS else 'subs'}
    if h >= 1080 and r - l <= 1440 - 120:
        x = int(max(0, min(480, (l + r) // 2 - 720)))
        ex.update(screen={'x': x, 'y': 0, 'h': 1080, 'w': 1440}, center_h=810, center_y=560)
    else:
        ex.update(screen={'x': 0, 'y': 0, 'h': 1080, 'w': 1920}, center_h=608, center_y=650)
    ex['subs_y'] = ex['center_y'] + ex['center_h'] + 80
    return 'center', ex


def choose(src, a, b, slug, text, side, cam, category=None, title='', cid=''):
    """(layout, extra spec keys) for the stretch [a, b] of src. category/title: the stream's (Kick category, stream title)."""
    import facecam
    if (category or '').strip().lower() in IRL_CATS:
        side = cam = None                                # an IRL stream has no facecam overlay: anything found is a false hit
    if (side or cam) and (slug or '').lower() not in TRADERS:
        import gamecam
        if gamecam.is_game(category, title) and not player_page(src, a, b, cam):   # the category says game: the picture must agree
            cs = gamecam.cams(src, a, b) or gamecam.cams(src, a + (b - a) * 0.45, b)
            if cs:
                ov = gamecam.overlay_rects_span(src, a, b)
                cs = gamecam.refine_cams(cs, src, a + (b - a) * 0.5)   # the editor's webcam boxes (same reads, cached)
                return 'gamesplit', {'cams': cs, 'overlays': ov}
    if cam and not side and (slug or '').lower() not in TRADERS and cam['w'] < 0.22 * (cam.get('src') or [1920])[0]:
        # a small facecam (a corner box under ~22% of the screen): blown up to a full-width panel it goes soft, so it
        # rides as a rounded inset over the content instead (the 22.8M-view Jynxzi page's layout)
        return 'pip', {}
    if cam and not side and (slug or '').lower() not in TRADERS and ui_content(src, a, b, cam):
        # nothing worth showing beside the cam (their chat, a dashboard): the cam alone, full width, over a blurred copy
        ih = int(round(1080 * cam['h'] / max(1, cam['w']) / 2) * 2)
        return 'center', {'camcard': True, 'cap_y': int(CENTER_Y + (CENTER_H + ih) / 2 + 90)}
    if side or cam:
        extra = {}
        if (slug or '').lower() in TRADERS:
            extra['top_h'] = 608 if is_trade(text) else 960
        return 'hooksplit', extra
    # a sit-down conversation on a still camera: half the time the fillersplit (the speaker camera over our own satisfying
    # loop, the 11.2M-view Adin Ross format), half the time the plain speaker camera, so the two can be compared
    if (category or '').strip().lower() in TALK_CATS and still_camera(src, a, b):
        import zlib
        if zlib.crc32((cid or src).encode()) % 2 == 0:
            return 'fillersplit', {}
    scr = facecam.screen_layout(src, a, b)
    if scr:
        prof = PROFILES.get((slug or '').lower()) or {}
        return 'center', {'screen': scr, **({'inset': prof['inset']} if prof.get('inset') else {})}
    # a camera with no facecam (IRL): less crop, so the picture stays sharp (experiment #2); was letterbox / speaker
    try:
        return irl_fill(src, a, b, slug)
    except Exception:  # noqa: BLE001
        return ('letterbox', {}) if wide_scene(src, a, b) else ('speaker', {})
