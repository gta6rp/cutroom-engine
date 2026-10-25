#!/usr/bin/env python3
"""A story edit: one arc from a streamer's lore (research/lore), told in its own clips, with date cards between beats.

Each beat is a clip of the moment (Kick/Twitch/YouTube URL, downloaded via clipcut.fetch), normalised and joined. One
editor call reads every beat (transcript + two frames each) and picks the 4-10 s that carry it plus a short on-screen
caption (from what was said or shown, checked like live hooks). The edit opens on the story's title card, then
date card -> beat, date card -> beat ..., auto-framed on whoever talks, the title held on screen as a hook box.
Tone sets the cut: 'sad' slows nothing and adds no music (sounds go on in-app), but uses longer holds and quieter
cards; 'happy'/'hype' cut tighter. Never invents events: beats come from the sourced lore file.

usage: storycut.py story.json [--judge sonnet] [--force]
  story.json: {"id", "title", "tone": "happy|sad|funny|drama", "streamer", "credit", "program",
               "beats": [{"date": "JUNE 2025", "what": "they meet on stream", "url": "...", "source_url": "..."}]}
"""
import argparse, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'live'))
import clipcut as C  # noqa: E402
from rankcut import norm  # noqa: E402

RULES = """You are the editor of a short-form clipping page making a STORY EDIT: one storyline of a streamer, told in clips
from different dates. Below is the story (title, tone) and each beat: what happened (from sourced research), its date,
and the clip's transcript (seconds from that clip's start). Two frames per beat follow, in order.

For each beat pick the 4-10 seconds that carry it best (the line or reaction that tells this part of the story), and
write an on-screen caption of 3-8 words that says what this beat is, from words said or things the frames show. If a
beat's clip doesn't show what the research says, say so (fit: false) instead of forcing it. Hard no: slurs, sexual
content, gambling on screen, minors, real allegations.

Reply with JSON only: {"beats": [{"i": <index>, "span": [a, b], "caption": "...", "fit": true, "evidence": "..."}],
"post_caption": "one line + 3 hashtags"}

STORY:
"""


def main(path, judge_mode='sonnet', force=False):
    import rehear, judge as J
    R = json.load(open(path))
    d = os.path.join(C.OUT, R['id']); os.makedirs(d, exist_ok=True)
    beats = []
    for k, b in enumerate(R['beats']):
        bd = os.path.join(d, f'b{k}'); os.makedirs(bd, exist_ok=True)
        src, _ = C.fetch({'url': b['url'], 'local': b.get('local')}, bd)
        n = os.path.join(bd, 'norm.mp4')
        if not os.path.exists(n):
            norm(src, n)
        beats.append({**b, 'file': n, 'dur': C.duration(n)})
    off = 0.0
    for b in beats:
        b['off'] = off; off += b['dur']
    joined, jaud = os.path.join(d, 'src.mp4'), os.path.join(d, 'src.m4a')
    if force or not os.path.exists(joined):
        open(os.path.join(d, 'parts.txt'), 'w').write(''.join(f"file '{b['file']}'\n" for b in beats))
        C.ff('-f', 'concat', '-safe', 0, '-i', os.path.join(d, 'parts.txt'), '-c', 'copy', joined)
        C.ff('-i', joined, '-vn', '-c:a', 'copy', jaud)
    wf = os.path.join(d, 'words.json')
    if force or not os.path.exists(wf):
        words = []
        for b in beats:
            for t in range(0, int(b['dur']) + 1, 30):
                if b['dur'] - t > 0.5:
                    words += rehear.hear_safe(jaud, b['off'] + t, b['off'] + min(b['dur'], t + 30))
        json.dump({'segments': [{'start': 0, 'end': off, 'text': ' '.join(w['w'] for w in words), 'words': words}]}, open(wf, 'w'))
    words = json.load(open(wf))['segments'][0]['words']
    ej = os.path.join(d, 'editor.json')
    if force or not os.path.exists(ej):
        info, frames = [], []
        for k, b in enumerate(beats):
            ws = [w for w in words if b['off'] <= w['s'] < b['off'] + b['dur']]
            info.append({'i': k, 'date': b['date'], 'what_happened': b['what'], 'duration': round(b['dur'], 1),
                         'transcript': ' '.join(f"[{w['s'] - b['off']:.0f}] {w['w']}" if j % 10 == 0 else w['w'] for j, w in enumerate(ws))[:1500]})
            for f in (0.35, 0.7):
                try:
                    frames.append(C.ff('-ss', f"{b['off'] + b['dur'] * f:.2f}", '-i', joined, '-frames:v', 1, '-vf', 'scale=360:-2', '-q:v', 6, '-f', 'image2pipe', '-c:v', 'mjpeg', '-'))
                except RuntimeError:
                    pass
        import lore
        prompt = RULES + lore.prompt_block(R['streamer']) + '\n' + json.dumps({'title': R['title'], 'tone': R.get('tone'), 'streamer': R['streamer'], 'beats': info}, ensure_ascii=False)
        raw = J._ask(prompt, frames, 120, J.CLAUDE_MODELS.get(judge_mode, 'sonnet'))
        json.dump(json.loads(re.search(r'\{.*\}', raw, re.S).group(0)), open(ej, 'w'), indent=1, ensure_ascii=False)
    v = json.load(open(ej))
    picks = {x['i']: x for x in v['beats'] if x.get('fit', True)}
    segs, cards, caps = [], {-1: R['title']}, []
    for k, b in enumerate(beats):
        x = picks.get(k)
        if not x:
            print(f"  beat {k} ({b['date']}): skipped, the clip doesn't show it"); continue
        a, e = (float(t) for t in x['span'])
        a, e = max(0.3, a), min(b['dur'] - 0.1, e)
        if segs:
            cards[len(segs) - 1] = b['date']          # date card before this beat
        else:
            cards[-1] = f"{R['title']}\\N{b['date']}"  # title + first date open the edit
        segs.append([round(b['off'] + a, 2), round(b['off'] + e, 2)])
        caps.append(J.mask(re.sub(r'[“”"#]', '', x.get('caption') or '')))
    if len(segs) < 2:
        raise SystemExit('fewer than two beats fit: not a story')
    import facecam
    seg_cams = [facecam.detect(joined, a0, b0) for a0, b0 in segs]   # a facecam over a screen gets the split, not the moving crop
    spec = {'id': R['id'], 'video': 'src.mp4', 'audio': 'src.m4a', 'words': 'words.json', 'out_dir': '.', 'layout': 'speaker', 'seg_cams': seg_cams,
            'segments': segs, 'cards': {str(k): t for k, t in cards.items()}, 'card_s': 1.4 if R.get('tone') == 'sad' else 1.0,
            'hook': R['title'].title() if R['title'].isupper() else R['title'], 'hook_full': True, 'captions': 'subs',
            'credit': R['credit'], 'program': R.get('program'), 'source_url': beats[0].get('source_url'), 'post_caption': v.get('post_caption'),
            'seg_titles': caps}
    json.dump(spec, open(os.path.join(d, 'spec.json'), 'w'), indent=1, ensure_ascii=False)
    r = subprocess.run(['nice', '-n', '10', C.PY, os.path.join(HERE, 'edit.py'), os.path.join(d, 'spec.json')], capture_output=True, text=True,
                       env={**os.environ, 'CUTROOM_FF_THREADS': '4'})
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    print(f"{R['id']}: {len(segs)} beats | " + ' / '.join(caps), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('story')
    ap.add_argument('--judge', default='sonnet')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    main(a.story, a.judge, a.force)
