#!/usr/bin/env python3
"""The critic: the editor watches one of OUR finished cuts the way it studied the top pages' posts
(research/references/STYLE-GUIDE.md) and says what a top clip page would fix before posting it.

critique(mp4, spec=None) -> {"score": 1-10, "post": bool, "issues": [{"what", "fix", "severity": 1-3}], "best": "..."}
usage: critic.py CUT.mp4 [CUT.mp4 ...]          prints each critique; with --report, a tally of the issues
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'live'))
GUIDE = os.path.join(os.path.dirname(HERE), 'config', 'STYLE-GUIDE.md')

RULES = """You are the head editor of a top TikTok clip page (millions of views a post). You get one of your junior editor's
finished vertical cuts: 10 frames in order, the transcript, its hook and layout. Judge it against what the top pages do
(the house style guide below) and against plain craft. Look hard at: text covering a face or the action; the crop (heads
cut off, the subject off-centre, empty space, the wrong person in frame); leftover stream overlays (chat, counters,
alerts, a second copy of a cam); a frame too dark or blurry to read on a phone; dead air before the moment starts; a
hook that is vague, wrong, too long or doesn't match the picture; captions that lag, cover things or say nothing; an
ending that trails off. Don't invent problems: if it's good, say so.
Answer with JSON only:
{"score": 1-10 (would you post it as is: 8+ yes),
 "issues": [{"what": "the problem, specific (when, where on screen)", "fix": "what to change", "severity": 1-3}],
 "best": "the one thing that works best"}"""


def frames(path, n=10):
    import cv2
    c = cv2.VideoCapture(path); N = c.get(7); out = []
    for k in range(n):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + 0.5) / n)); ok, f = c.read()
        if ok:
            h, w = f.shape[:2]; s = 640 / max(h, w)
            out.append(cv2.imencode('.jpg', cv2.resize(f, (int(w * s), int(h * s))), [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes())
    return out


def critique(mp4, spec=None):
    import judge as J
    d = os.path.dirname(os.path.abspath(mp4))
    spec = spec or (json.load(open(os.path.join(d, 'spec.json'))) if os.path.exists(os.path.join(d, 'spec.json')) else {})
    words = ''
    wf = os.path.join(d, spec.get('words') or 'words.json')
    if os.path.exists(wf):
        W = json.load(open(wf))
        ws = [w for s in W.get('segments', []) for w in s.get('words', [])] if isinstance(W, dict) else W
        a, b = (spec.get('segments') or [[0, 1e9]])[0]
        words = ' '.join(w['w'] for w in ws if a - 0.5 <= w['s'] <= b + 0.5)[:1500]
        sheet = mp4[:-4] + '.json'                      # edit.py's sheet: the source pieces actually in the cut, in order
        tl = [t for t in (json.load(open(sheet)).get('timeline') or []) if 'src' in t] if os.path.exists(sheet) else []
        if tl:
            words = ' '.join(w['w'] for t in tl for w in ws if t['src'][0] - 0.05 <= w['s'] < t['src'][1])[:1500]
    ej = next((os.path.join(d, f) for f in ('edit.json', 'edit-mini.json', 'edit-cine.json') if os.path.exists(os.path.join(d, f))), None)
    if ej and os.path.exists(wf):                       # a studio edit: what its voice lines actually say, in edit order (the
        E = json.load(open(ej))                         # whole source's transcript made the critic fault tails the cut never had)
        said = []
        for v in sorted(E.get('voice') or [], key=lambda v: v['at']):
            said += [w['w'] for w in ws if v['in'] - 0.05 <= w['s'] < v['out']]
        words = ' '.join(said)[:1500] + '  [after the last line: music only]'
    guide = open(GUIDE).read()[:6000] if os.path.exists(GUIDE) else ''
    kind = spec.get('kind') or ('edit' if re.search(r'(edit|story|cine|hype|funny|wild|mafiathon|fncs|v2)', os.path.basename(mp4)) and not spec.get('layout') else 'clip')
    as_edit = ('\n\nTHIS IS AN EDIT, not a raw clip (a story edit, hype edit or ranking cut to music): judge it as an edit. It needs no '
               'hook box (a title or date card can open it), music is expected; judge story clarity for a stranger, the cutting, the '
               'picture, text placement and the ending.') if kind == 'edit' else ''
    prompt = (RULES + as_edit + f"\n\nHOUSE STYLE GUIDE:\n{guide}\n\nTHE CUT: kind {kind}, layout {spec.get('layout')}, hook {spec.get('hook')!r}, "
              f"streamer credit {spec.get('credit')}\nTRANSCRIPT: {words or '(none)'}")
    fr = frames(mp4)
    v = None
    for extra in ('', '\n\nYour last answer was not valid JSON (escape quotes inside strings). Answer again, JSON only.'):
        raw = J._ask(prompt + extra, fr, 150)
        m = re.search(r'\{.*\}', raw, re.S)
        try:
            v = json.loads(m.group(0)) if m else None
        except ValueError:
            v = None
        if v:
            break
    v = v or {'score': None, 'issues': [], 'best': '(the critic could not answer)'}
    v['post'] = (v.get('score') or 0) >= 8
    v['file'] = os.path.basename(mp4); v['layout'] = spec.get('layout')
    json.dump(v, open(mp4[:-4] + '.critic.json', 'w'), indent=1, ensure_ascii=False)
    return v


def kit_notes(qdir, spec=None):
    """Critique a post-kit item (posting/queue/<id>) and keep the notes in its post.json for the owner to read."""
    pj = os.path.join(qdir, 'post.json')
    p = json.load(open(pj))
    v = critique(os.path.join(qdir, p['file']), spec or {'hook': p.get('hook'), 'credit': p.get('credit')})
    p = json.load(open(pj))                               # re-read: the owner may have marked it posted meanwhile
    p['critic'] = {'score': v.get('score'), 'best': v.get('best'),
                   'fixes': [i for i in sorted(v.get('issues') or [], key=lambda i: -(i.get('severity') or 0))][:3]}
    json.dump(p, open(pj, 'w'), indent=1, ensure_ascii=False)
    return p['critic']


if __name__ == '__main__':
    if sys.argv[1:2] == ['--queue']:
        Q = os.path.join(os.path.dirname(HERE), 'posting', 'queue')
        todo = [os.path.join(Q, d) for d in sorted(os.listdir(Q)) if os.path.exists(os.path.join(Q, d, 'post.json'))
                and 'critic' not in json.load(open(os.path.join(Q, d, 'post.json')))]
        with ThreadPoolExecutor(3) as ex:
            for d, c in zip(todo, ex.map(kit_notes, todo)):
                print(f"{c.get('score')}/10 {os.path.basename(d)}: {(c['fixes'][0]['what'] if c.get('fixes') else '')[:110]}")
        sys.exit()
    files = [f for f in sys.argv[1:] if f.endswith('.mp4')]
    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(critique, files))
    for v in res:
        print(f"{v.get('score')}/10 {v.get('layout') or '-':11s} {v['file']}: best: {(v.get('best') or '')[:90]}")
        for i in v.get('issues') or []:
            print(f"   [{i.get('severity')}] {i.get('what', '')[:150]} -> {i.get('fix', '')[:110]}")
