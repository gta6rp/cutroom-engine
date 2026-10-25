#!/usr/bin/env python3
"""rankedit: a Top-N ranking (made by rankcut.py: spec.json + editor.json + src.mp4 + words.json) re-cut in the studio.

The list sits on black at the top (the title, then one row per entry, revealed as the countdown reaches it, the current
one in yellow); each entry fills a 4:5 frame below: the facecam on top and the game or page under it when the stream has
a cam overlay, else framed on the faces. Every entry opens on its number popping over the picture with a hit; the music
stays low and muffled under the talking and opens up on #1, with a riser into it. Word-pop captions, the credit mark.

usage: rankedit.py RANK_DIR [--song funk-do-bounce] [--workers 6]
Writes RANK_DIR/<id>-studio.mp4 (with music), <id>-studio-site.mp4 (no music), edit.json, cutsheet.webp.
"""
import argparse, json, os, re, subprocess, sys, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import studio  # noqa: E402
import hypecut as H  # noqa: E402
from storyedit import mask  # noqa: E402

W, HH = 1080, 1920
BOX = [0, 440, 1080, 1350]          # the picture: 4:5 under the list
YEL, WHITE, GREY = [255, 214, 10], [255, 255, 255], [150, 146, 140]
NUMCOL = {1: [255, 214, 10], 2: [220, 220, 228], 3: [205, 139, 62]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rank'); ap.add_argument('--song', default=''); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    R = os.path.abspath(a.rank)
    S = json.load(open(os.path.join(R, 'spec.json')))
    ed = json.load(open(os.path.join(R, 'editor.json')))
    src = os.path.join(R, 'src.mp4')
    segs = S['segments']
    labels = S.get('rank') or [it['label'] for it in ed['items']]
    n = len(segs)
    global BOX
    BOX = [0, 158 + n * 54 + 12, 1080, 1350]            # the picture starts right under the list
    W_all = json.load(open(os.path.join(R, 'words.json')))
    words = [w for sg in W_all.get('segments', []) for w in sg.get('words', [])] if isinstance(W_all, dict) else W_all
    title = (S.get('rank_title') or S['hook']).upper()
    credit = S.get('watermark') or S.get('credit', '')
    handle = 'cutroom' + re.sub(r'[^a-z0-9]', '', (S.get('credit', '').split('·')[0] or S['id']).lower())
    if not a.song:
        import zlib
        hype = [x['name'] for x in json.load(open(os.path.join(H.MUSIC, 'library.json'))) if 'hype' in x.get('mood', '')]
        a.song = hype[zlib.crc32(S['id'].encode()) % len(hype)]
    song = H.song(a.song)

    shots, voice, texts, trans, hits, sfx = [], [], [], [], [], []
    t = 0.0
    INTRO = 1.6
    # intro: the title over the #N entry's first frame, frozen and dark, then the countdown starts
    shots.append({'src': src, 'in': segs[0][0], 'at': 0.0, 'dur': INTRO, 'speed': 0.0, 'focus': 'auto', 'cam': [[0, 1.0, 0, 0, 0], [1, 1.08, 0, 0, 0]],
                  'look': ['bw'], 'box': BOX})
    texts.append({'t0': 0.15, 't1': INTRO, 'text': f'TOP {n}', 'style': 'numsm', 'anim': 'pop', 'y': BOX[1] + BOX[3] // 2, 'color': YEL})
    sfx.append({'at': 0.15, 'type': 'boom', 'gain': 0.5})
    t = INTRO
    starts = []
    for k, (sa, sb) in enumerate(segs):
        rank = n - k
        starts.append(t)
        dur = sb - sa
        lay = H.overlay_layout(src, sa, sb)
        sh = {'src': src, 'in': sa, 'at': round(t, 3), 'dur': round(dur, 3), 'speed': 1.0, 'focus': 'auto:both', 'cam': [[0, 1.0, 0, 0, 0], [1, 1.06, 0, 0, 0]],
              'look': [], 'box': BOX, 'face_fill': 0.16, 'headroom': 0.2}
        if lay:
            sh['camsplit'] = lay
        shots.append(sh)
        voice.append({'src': src, 'in': sa, 'out': sb, 'at': round(t, 3), 'duck': 0.18})
        # the number pops over the picture at the start of its entry
        texts.append({'t0': round(t, 3), 't1': round(t + 1.1, 3), 'text': f'#{rank}', 'style': 'num', 'anim': 'pop', 'y': BOX[1] + BOX[3] // 2 - 80,
                      'color': NUMCOL.get(rank, WHITE)})
        hits += [{'at': round(t, 3), 'type': 'flash', 'amt': 0.7 if rank > 1 else 1.0}, {'at': round(t, 3), 'type': 'punch', 'amt': 1.4}]
        sfx.append({'at': round(t, 3), 'type': 'boom', 'gain': 0.45 if rank > 1 else 0.6})
        if k:
            trans.append({'at': round(t, 3), 'type': ['whip', 'zoom', 'glitch', 'whip'][k % 4], 'dur': 0.3, **({'dir': 'left'} if k % 2 else {})})
            sfx.append({'at': round(t, 3), 'type': 'whoosh', 'gain': 0.35, 'dur': 0.4})
        # word-pop captions near the bottom of the picture
        ws = [w for w in words if sa - 0.05 <= w['s'] < sb - 0.1]
        groups, cur = [], []
        for w in ws:
            if cur and w['s'] - cur[-1]['s'] > 0.55:
                groups.append(cur); cur = []
            cur.append(w)
            if len(cur) == 2 or w['w'][-1:] in '.,?!':
                groups.append(cur); cur = []
        if cur:
            groups.append(cur)
        for j, g in enumerate(groups):
            t0 = t + g[0]['s'] - sa
            if t0 < t + 1.1:                              # the number owns the first second
                continue
            t1 = t + (groups[j + 1][0]['s'] - sa if j + 1 < len(groups) else min(dur, g[-1]['e'] - sa + 0.25))
            texts.append({'t0': round(t0, 3), 't1': round(max(t1, t0 + 0.25), 3), 'text': mask(' '.join(w['w'].strip('.,!') for w in g)),
                          'style': 'hype', 'anim': 'pop', 'y': BOX[1] + BOX[3] - 230, 'color': WHITE})
        t += dur
    END = t + 0.8
    shots.append({'src': None, 'at': round(t, 3), 'dur': 0.8})
    trans.append({'at': round(t, 3), 'type': 'flash', 'dur': 0.2})
    # the list on black: title, then the rows; a row shows once the countdown reaches it, yellow while it plays
    from PIL import ImageFont                           # the title on one line: the largest size that fits
    tst = 'ranktitle34'
    for st in ('ranktitle', 'ranktitle46', 'ranktitle40', 'ranktitle34'):
        size, fname, stroke, _ = studio.STYLES[st]
        if ImageFont.truetype(os.path.join(studio.FONTS, fname), size).getlength(title) + 2 * stroke <= W * 0.88:
            tst = st; break
    texts.append({'t0': 0.0, 't1': round(END, 3), 'text': title, 'style': tst, 'anim': 'none', 'y': 78, 'color': WHITE})
    for k in range(n):
        rank = n - k
        y = 158 + (rank - 1) * 54
        lab = f'{rank}. {mask(labels[k])}'
        hidden = f'{rank}.'
        texts.append({'t0': 0.0, 't1': round(starts[k], 3), 'text': hidden, 'style': 'rankrow', 'anim': 'none', 'y': y, 'color': GREY})
        texts.append({'t0': round(starts[k], 3), 't1': round(starts[k] + segs[k][1] - segs[k][0], 3), 'text': lab, 'style': 'rankrow', 'anim': 'pop', 'y': y, 'color': YEL})
        texts.append({'t0': round(starts[k] + segs[k][1] - segs[k][0], 3), 't1': round(END, 3), 'text': lab, 'style': 'rankrow', 'anim': 'none', 'y': y, 'color': WHITE})
    if os.environ.get('CUTROOM_ONSCREEN_CREDIT') == '1':   # no watermark by default (the user, 2026-10-21: not a program rule)
        texts.append({'t0': 0.0, 't1': round(t, 3), 'text': f'{credit} · @{handle}', 'style': 'mark', 'anim': 'none', 'y': 1855, 'opacity': 0.75})
    texts.append({'t0': round(t + 0.05, 3), 't1': round(END, 3), 'text': f'@{handle}', 'style': 'caption', 'anim': 'fade', 'y': BOX[1] + BOX[3] // 2})
    # music: low and muffled under the countdown, the drop exactly when #1 starts (a riser into it)
    one = starts[-1]
    drop_song = next((d for d in song['drops'] if d >= one), song['drops'][-1])
    mstart, mat = round(drop_song - one, 3), 0.0
    if mstart < 0:
        mat, mstart = -mstart, 0.0
    sfx.append({'at': round(one, 3), 'type': 'riser', 'dur': 2.2, 'gain': 0.3})
    E = {'size': [W, HH], 'fps': 30, 'duration': round(END, 3),
         'music': {'src': song['path'], 'start': mstart, 'at': mat, 'gain': 0.7, 'drop_gain': 0.45, 'fade_in': 0.4, 'fade_out': 0.6, 'muffle_until': round(one, 3), 'levels': {'pre': -20, 'post': -12}},
         'voice': voice, 'shots': shots, 'transitions': trans, 'hits': hits, 'sfx': sfx, 'texts': texts, 'finish': {'grain': 0.015, 'vignette': 0.15}}
    for x, y in zip(shots, shots[1:]):
        assert abs(x['at'] + x['dur'] - y['at']) < 0.02, (x['at'], x['dur'], y['at'])
    cid = f"{S['id']}-studio"
    json.dump(E, open(os.path.join(R, 'edit.json'), 'w'), indent=1, ensure_ascii=False)
    mp4 = os.path.join(R, f'{cid}.mp4')
    studio.render(E, mp4, a.workers)
    td = tempfile.mkdtemp(prefix='rankedit-'); wav = os.path.join(td, 'site.wav')
    studio.mix_audio({**E, 'music': None}, wav, td)
    subprocess.run([studio.FF, '-nostdin', '-y', '-loglevel', 'error', '-i', mp4, '-i', wav, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
                    '-b:a', '160k', '-shortest', '-movflags', '+faststart', os.path.join(R, f'{cid}-site.mp4')], check=True)
    shutil.rmtree(td, ignore_errors=True)
    subprocess.run([sys.executable, os.path.join(HERE, 'cutsheet.py'), os.path.join(R, 'edit.json'), mp4, os.path.join(R, 'cutsheet.webp'),
                    '--title', f"{title.title()}: how the AI cut it"], capture_output=True)
    print(f"{mp4}: {E['duration']:.1f} s, {n} entries, song {a.song}")


if __name__ == '__main__':
    main()
