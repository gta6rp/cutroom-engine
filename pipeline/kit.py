#!/usr/bin/env python3
"""Put a finished studio edit (hypecut / storyedit / rankedit) into the post kit (posting/queue/<id>), the way live.py's
to_kit does for live cuts: the owner downloads it on the phone and posts it in the app. Nothing is posted from here.

The kit item carries two copies: <id>.mp4 with our music baked in, and <id>-nomusic.mp4 (the site copy: voices only) for
posting on the official TikTok sound instead (a post on the sound joins its trend; post.json 'sound' says which sound and
where our cut starts in it). A cover frame, the critic's notes (pipeline/critic.py) and the edit's own caption come along.

usage: kit.py EDIT_DIR/<id>.json [--account @cutroomn3on] [--note "..."] [--program "..."]
"""
import argparse, json, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QUEUE = os.path.join(ROOT, 'posting', 'queue')
FF = os.path.join(ROOT, '.venv', 'bin', 'ffmpeg')


def add(meta_path, account=None, note=None, program=None):
    m = json.load(open(meta_path))
    d = os.path.dirname(os.path.abspath(meta_path)); cid = m['id']
    held = json.load(open(os.path.join(ROOT, 'portfolio', 'held', 'holds.json'))) if os.path.exists(os.path.join(ROOT, 'portfolio', 'held', 'holds.json')) else {}
    if cid in held or cid.removesuffix('-edit') in held:
        raise SystemExit(f'{cid} is held (portfolio/held/holds.json): {held.get(cid) or held.get(cid.removesuffix("-edit"))}')
    mp4, site = os.path.join(d, f'{cid}.mp4'), os.path.join(d, f'{cid}-site.mp4')
    if not os.path.exists(mp4):
        raise SystemExit(f'no {mp4}')
    q = os.path.join(QUEUE, cid); os.makedirs(q, exist_ok=True)
    shutil.copy2(mp4, os.path.join(q, f'{cid}.mp4'))
    if os.path.exists(site):
        shutil.copy2(site, os.path.join(q, f'{cid}-nomusic.mp4'))
    # cover: a frame a third of the way in (past the hook card, before the payoff)
    subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-ss', f"{m.get('duration', 9) / 3:.2f}", '-i', mp4, '-frames:v', '1', '-q:v', '3',
                    os.path.join(q, f'{cid}.jpg')], check=False)
    snd = m.get('sound') or {}
    post = {'account': account or (f"@{m['handle']}" if m.get('handle') else None), 'platform': 'tiktok', 'file': f'{cid}.mp4',
            'file_nomusic': f'{cid}-nomusic.mp4' if os.path.exists(site) else None, 'cover': f'{cid}.jpg',
            'caption': m.get('post_caption') or m.get('title'), 'hook': m.get('title'), 'kind': m.get('kind'), 'duration': m.get('duration'),
            'credit': m.get('credit'), 'music': m.get('music'),
            'sound': ({**snd, 'how': 'to post on the official sound: use the no-music copy, add this sound in the TikTok app and start it '
                       f"at {snd.get('start', 0):.0f} s so its drop lands where ours does"} if snd else None),
            'program': program, 'note': note, 'made_by': m.get('made_by'), 'status': 'ready'}
    json.dump({k: v for k, v in post.items() if v is not None}, open(os.path.join(q, 'post.json'), 'w'), indent=1, ensure_ascii=False)
    sys.path.insert(0, HERE)
    import critic
    try:
        c = critic.kit_notes(q, {'hook': m.get('title'), 'credit': m.get('credit'), 'kind': 'edit'})
        print(f"critic {c.get('score')}/10: {c.get('best')}")
    except Exception as e:  # noqa: BLE001
        print('critic failed:', e)
    return q


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('meta'); ap.add_argument('--account'); ap.add_argument('--note'); ap.add_argument('--program')
    a = ap.parse_args()
    print(add(a.meta, a.account, a.note, a.program))
