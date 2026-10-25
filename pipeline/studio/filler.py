#!/usr/bin/env python3
"""Our own "satisfying" filler footage for the lower panel of a talking clip (the fillersplit layout).

Top clip pages put looping game footage under a talking-heads clip so the eye always has motion (an 11.2M-view Adin Ross
page uses a Roblox obby; research/references/STYLES.md). Someone else's gameplay is a copyright and reuse risk, so we
render our own: physics toys of the kind that go viral on their own, drawn procedurally, seeded so every clip gets a
different one.

  ring   balls bouncing inside a slowly turning ring with a gap; a ball that escapes respawns in the middle
  trails coloured balls bouncing in a box, leaving fading rainbow trails
  grow   one ball bouncing in a circle, a little bigger on every hit

usage: filler.py OUT.mp4 --kind ring --seconds 40 --w 1080 --h 820 [--seed 3]
"""
import argparse, math, os, subprocess
import numpy as np
import cv2

FF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.venv', 'bin', 'ffmpeg')
PALETTE = [(255, 87, 87), (255, 189, 46), (80, 220, 120), (60, 180, 255), (190, 110, 255), (255, 110, 200)]   # RGB


def hsv(h):
    c = cv2.cvtColor(np.uint8([[[int(h) % 180, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def render(out, kind='ring', seconds=40.0, w=1080, h=820, seed=3, fps=30):
    rng = np.random.default_rng(seed)
    p = subprocess.Popen([FF, '-nostdin', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                          '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    S = 2                                                  # supersampling for smooth edges
    W, H = w * S, h * S
    cx, cy = W / 2, H / 2
    R = min(W, H) * 0.42
    g = 2600.0 * S                                         # gravity, px/s^2
    dt = 1 / fps
    trail = np.zeros((H, W, 3), np.float32)
    if kind == 'trails':
        n = 6
        pos = rng.uniform([W * 0.2, H * 0.2], [W * 0.8, H * 0.5], (n, 2)); vel = rng.uniform(-900, 900, (n, 2)) * S
        rad = np.full(n, 26.0 * S)
    elif kind == 'grow':
        pos = np.array([[cx + 40 * S, cy - R * 0.5]]); vel = np.array([[700.0 * S, 0.0]]); rad = np.array([22.0 * S])
    else:
        n = 3
        pos = np.array([[cx + rng.uniform(-60, 60) * S, cy - R * 0.3] for _ in range(n)]); vel = rng.uniform(-700, 700, (n, 2)) * S
        rad = np.full(n, 24.0 * S)
    cols = [PALETTE[(seed + i) % len(PALETTE)][::-1] for i in range(len(pos))]
    ang, gap = 0.0, math.radians(38)
    hue = rng.uniform(0, 180)
    prev = None
    for f in range(int(seconds * fps)):
        for _ in range(4):                                 # physics substeps
            sdt = dt / 4
            vel[:, 1] += g * sdt
            pos += vel * sdt
            if kind == 'trails':
                for i in range(len(pos)):
                    for k, lim in ((0, W), (1, H)):
                        if pos[i, k] - rad[i] < 0:
                            pos[i, k] = rad[i]; vel[i, k] = abs(vel[i, k]) * 0.98
                        elif pos[i, k] + rad[i] > lim:
                            pos[i, k] = lim - rad[i]; vel[i, k] = -abs(vel[i, k]) * 0.98
                    if vel[i, 1] ** 2 < (300 * S) ** 2 and pos[i, 1] > H - rad[i] - 2:
                        vel[i, 1] = -rng.uniform(1500, 2100) * S       # keep them going
            else:
                for i in range(len(pos)):
                    d = pos[i] - (cx, cy); r = float(np.hypot(*d))
                    if r + rad[i] >= R:
                        a = (math.atan2(d[1], d[0]) - ang) % (2 * math.pi)
                        if kind == 'ring' and a < gap:          # through the gap: out, and a new ball in the middle
                            if r > R + rad[i] * 3:
                                pos[i] = (cx + rng.uniform(-50, 50) * S, cy - R * 0.25); vel[i] = rng.uniform(-600, 600, 2) * S
                                cols[i] = PALETTE[int(rng.integers(len(PALETTE)))][::-1]
                            continue
                        nrm = d / max(r, 1e-6)
                        vn = float(vel[i] @ nrm)
                        if vn > 0:
                            vel[i] -= 2 * vn * nrm
                            vel[i] *= 1.0 if kind == 'ring' else 1.0
                            tang = np.array([-nrm[1], nrm[0]])
                            vel[i] += tang * rng.uniform(-60, 60) * S       # a little spin so it never settles
                            if kind == 'grow':
                                rad[i] = min(R * 0.8, rad[i] * 1.045)
                                if rad[i] >= R * 0.79:
                                    rad[i] = 22.0 * S
                        pos[i] = (cx, cy) + nrm * (R - rad[i] - 0.5)
        ang += math.radians(28) * dt
        hue += 0.6
        # draw: dark background, fading trails, the ring, the balls
        trail *= 0.90 if kind != 'grow' else 0.80
        for i in range(len(pos)):
            col = hsv(hue + i * 30) if kind != 'ring' else cols[i]
            q = prev[i] if prev is not None and np.hypot(*(prev[i] - pos[i])) < R * 0.5 else pos[i]
            cv2.line(trail, (int(q[0]), int(q[1])), (int(pos[i, 0]), int(pos[i, 1])), col, int(2 * rad[i]), cv2.LINE_AA)   # a smooth trail
        prev = pos.copy()
        img = np.full((H, W, 3), (18, 14, 12), np.float32) + trail * 0.55
        if kind != 'trails':
            ring_col = hsv(hue)
            a0 = math.degrees(ang + gap) if kind == 'ring' else 0
            a1 = math.degrees(ang + 2 * math.pi) if kind == 'ring' else 360
            cv2.ellipse(img, (int(cx), int(cy)), (int(R), int(R)), 0, a0, a1, ring_col, int(9 * S), cv2.LINE_AA)
        for i in range(len(pos)):
            col = hsv(hue + i * 30) if kind != 'ring' else cols[i]
            cv2.circle(img, (int(pos[i, 0]), int(pos[i, 1])), int(rad[i]), col, -1, cv2.LINE_AA)
            cv2.circle(img, (int(pos[i, 0]), int(pos[i, 1])), int(rad[i]), (255, 255, 255), int(3 * S), cv2.LINE_AA)
        frame = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError('filler encode failed')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--kind', default='ring', choices=['ring', 'trails', 'grow'])
    ap.add_argument('--seconds', type=float, default=40); ap.add_argument('--w', type=int, default=1080); ap.add_argument('--h', type=int, default=820)
    ap.add_argument('--seed', type=int, default=3)
    a = ap.parse_args()
    print(render(a.out, a.kind, a.seconds, a.w, a.h, a.seed))
