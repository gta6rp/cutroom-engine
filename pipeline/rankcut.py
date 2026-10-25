#!/usr/bin/env python3
"""A ranking ("TOP 5 ... THIS WEEK") from several moments of one streamer: the clips their viewers made most this week.

Each source clip is downloaded (clipcut.fetch), normalised to 1920x1080 30 fps and joined into one file, then
re-heard on the GPU ASR. One editor call (same headless Claude as live/judge.py) reads every moment (transcript +
two frames each), drops the weak or unsafe ones, picks the 8-14 s that carry each, names each in 2-5 words, orders
them #5 -> #1 and writes the list title. Labels are checked against what was said or shown, like live hooks.
edit.py renders the 'rank' layout: title bar, the stream whole, the list filling in as each moment plays.

A spec with "preset" (title, items, caption) comes from catalog.py, which already scored a deep pool: no editor call.

usage: rankcut.py rank.json [--judge sonnet] [--force]
  rank.json: {"id", "streamer", "slug", "platform", "credit", "program", "parts": [clip jobs as in clipcut.py]}
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'live'))
import clipcut as C  # noqa: E402

PY = C.PY

RULES = """You are the editor of a short-form clipping page. Below are several moments from one streamer, each a clip their
viewers made. Build a countdown video: "TOP N" moments, #N plays first, #1 last.

For each moment decide if it belongs (funny, wild, emotional or hype, and makes sense without context). Hard no:
slurs, sexual content, gambling (also gambling on screen: slots, a casino or poker table, cards and chips, a betting site, a
gambling broadcast being watched, even if the joke is about something else), minors, real allegations or tragedies. Keep 3 to 5 moments.
For each kept moment give: the 8-14 second window that carries it (seconds from that moment's own start), a label of
2-5 words that names it (use words that were said or things the frames show: no invented details), and its rank.
Then a list title of 3-7 words in caps style like "XQC'S WILDEST MOMENTS THIS WEEK" (no emojis), and a post caption
(one line plus 3 hashtags).

Reply with JSON only:
{"title": "...", "caption": "...", "items": [{"part": <index>, "rank": <1..N, 1 = best>, "label": "...", "span": [start, end], "evidence": "words said or 'frame: ...'"}], "dropped": [{"part": <index>, "why": "..."}]}

MOMENTS:
"""


def norm(src, out):
    """1920x1080 30 fps, letterboxed if needed, so the parts join without a re-scale in the renderer."""
    C.ff('-i', src, '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30',
         '-c:v', 'libx264', '-preset', 'veryfast', '-crf', 18, '-c:a', 'aac', '-ar', 48000, '-ac', 2, '-b:a', '192k', out)


def main(spec_path, judge_mode='sonnet', force=False):
    import rehear, judge as J
    R = json.load(open(spec_path))
    d = os.path.join(C.OUT, R['id']); os.makedirs(d, exist_ok=True)
    parts = []
    for k, job in enumerate(R['parts']):
        pd = os.path.join(d, f'p{k}'); os.makedirs(pd, exist_ok=True)
        src, aud = C.fetch(job, pd)
        n = os.path.join(pd, 'norm.mp4')
        if not os.path.exists(n):
            norm(src, n)
        parts.append({'job': job, 'file': n, 'dur': C.duration(n)})
    # join
    joined, jaud = os.path.join(d, 'src.mp4'), os.path.join(d, 'src.m4a')
    off = 0.0
    for p in parts:
        p['off'] = off; off += p['dur']
    if force or not os.path.exists(joined):
        lst = os.path.join(d, 'parts.txt')
        open(lst, 'w').write(''.join(f"file '{p['file']}'\n" for p in parts))
        C.ff('-f', 'concat', '-safe', 0, '-i', lst, '-c', 'copy', joined)
        C.ff('-i', joined, '-vn', '-c:a', 'copy', jaud)
    # hear every part
    wf = os.path.join(d, 'words.json')
    if force or not os.path.exists(wf):
        words = []
        for p in parts:
            for t in range(0, int(p['dur']) + 1, 30):
                if p['dur'] - t > 0.5:
                    words += rehear.hear_safe(jaud, p['off'] + t, p['off'] + min(p['dur'], t + 30))
        json.dump({'segments': [{'start': 0, 'end': off, 'text': ' '.join(w['w'] for w in words), 'words': words}]}, open(wf, 'w'))
    words = json.load(open(wf))['segments'][0]['words']
    # one editor call over all moments
    ej = os.path.join(d, 'editor.json')
    if R.get('preset') and (force or not os.path.exists(ej)):
        json.dump(R['preset'], open(ej, 'w'), indent=1, ensure_ascii=False)   # picked from the catalog (catalog.py): no second read
    if force or not os.path.exists(ej):
        moments, frames = [], []
        for k, p in enumerate(parts):
            ws = [w for w in words if p['off'] <= w['s'] < p['off'] + p['dur']]
            moments.append({'part': k, 'clip_title': p['job'].get('title'), 'viewer_clip_views': p['job'].get('views'), 'duration': round(p['dur'], 1),
                            'transcript': [{'t': round(w['s'] - p['off'], 1), 'w': w['w']} for w in ws]})
            for f in (0.35, 0.75):
                try:
                    frames.append(C.ff('-ss', f"{p['off'] + p['dur'] * f:.2f}", '-i', joined, '-frames:v', 1, '-vf', 'scale=400:-2', '-q:v', 6, '-f', 'image2pipe', '-c:v', 'mjpeg', '-'))
                except RuntimeError:
                    pass
        import lore
        prompt = RULES + lore.prompt_block(R.get('slug') or R['streamer']) + '\n' + json.dumps({'streamer': R['streamer'], 'moments': moments}, ensure_ascii=False) + \
            f"\n\n(Frames: two per moment, in order: moment 0 at 35% and 75%, then moment 1, ...)"
        raw = J._ask(prompt, frames, 90, J.CLAUDE_MODELS.get(judge_mode, 'sonnet'))
        v = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
        json.dump(v, open(ej, 'w'), indent=1, ensure_ascii=False)
    v = json.load(open(ej))
    items = sorted(v['items'], key=lambda it: -it['rank'])        # #N first, #1 last
    segs, labels = [], []
    for it in items:
        p = parts[it['part']]
        a, b = (float(x) for x in it['span'])
        a, b = max(0.4, a), min(p['dur'] - 0.1, b)
        if b - a < 4:
            a, b = max(0.4, p['dur'] - 12), p['dur'] - 0.1
        segs.append([round(p['off'] + a, 2), round(p['off'] + b, 2)])
        lab = J.mask(re.sub(r'[“”"#]', '', it['label'])).strip()
        ws = [w['w'] for w in words if p['off'] + a - 1 <= w['s'] <= p['off'] + b + 1]
        pay = {'lines': [{'t': 0, 'text': ' '.join(ws)}], 'chat_top': [], 'title': p['job'].get('title') or '', 'streamer': R['streamer']}
        extra = J.unsupported(lab, str(it.get('evidence') or ''), pay)
        if len(extra) > 1:                                  # a label the moment doesn't back up: use its own words
            q = J.quote_hook(pay, 6).strip('“”')
            print(f"  label '{lab}' says {extra}: using '{q or lab}'")
            lab = q or lab
        labels.append(lab)
    title = re.sub(r'[^\w\s\'!?.-]', '', v['title']).upper().strip()
    spec = {'id': R['id'], 'video': 'src.mp4', 'audio': 'src.m4a', 'words': 'words.json', 'out_dir': '.', 'layout': 'rank',
            'segments': segs, 'rank': labels, 'rank_title': title, 'hook': title.title(), 'captions': 'subs', 'subs_y': 1600,
            'watermark': R['credit'], 'mark_y': 1790, 'credit': f"{R['streamer']} · {R['credit']}", 'program': R.get('program'),
            'source_url': R['parts'][0].get('source_url'), 'post_caption': v.get('caption')}
    json.dump(spec, open(os.path.join(d, 'spec.json'), 'w'), indent=1, ensure_ascii=False)
    r = subprocess.run(['nice', '-n', '10', PY, os.path.join(HERE, 'edit.py'), os.path.join(d, 'spec.json')], capture_output=True, text=True,
                       env={**os.environ, 'CUTROOM_FF_THREADS': '4'})
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    print(f"{R['id']}: {title} | " + ' / '.join(f"{len(labels) - i}. {l}" for i, l in enumerate(labels)), flush=True)
    for x in v.get('dropped', []):
        print(f"  dropped part {x['part']}: {x['why']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('spec')
    ap.add_argument('--judge', default='sonnet')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    main(a.spec, a.judge, a.force)
