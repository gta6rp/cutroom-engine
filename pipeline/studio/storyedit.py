#!/usr/bin/env python3
"""storyedit: a storyline from the lore library -> a music-cut story edit (the Jason x Sakura v2 recipe, automated).

Input: a story folder with pool.json, the footage for one storyline:
  [{"id": "gift", "file": "src/gift.mp4", "date": "2026-04-07", "note": "N3on gives her a pink Chanel bag (Kick clip)"}, ...]
1. Shot log (one editor call per source, per stretch of a long one): quotable lines with word times and who says them,
   and the usable visual moments (faces, emotion, action) with where the people are in the frame.
2. Assembly (one editor call, text only): the arc from the lore library + every shot log + the song's shape -> a plan:
   a cold-open line before the drop, then acts (a date, a mood, the montage shots, one to three lines), a recap, the
   last line and the shot to hold on.
3. Build (this file): the plan on the song's beat grid. Music muffled under the cold open, the drop on the title, one
   cut per beat in happy montages and two in sad ones, lines held over the music with typed captions in the speaker's
   colour, black act breaks with a date into a black-and-white turn, a fast recap, the last shot held, fade out.
   studio.py renders it; the site copy has no music.

usage: storyedit.py STORY_DIR --name "N3on & Alabama" --streamer N3on --song snowfall [--credit kick.com/n3on]
                    [--handle cutroomn3on] [--arc "N3on x Alabama"] [--plan plan.json] [--workers 6]
"""
import argparse, io, json, math, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
ROOT = os.path.dirname(PIPE)
sys.path.insert(0, HERE); sys.path.insert(0, PIPE); sys.path.insert(0, os.path.join(PIPE, 'live'))
import studio  # noqa: E402
import hypecut as H  # noqa: E402

COL = {'him': [255, 255, 255], 'her': [255, 176, 210], 'other': [255, 224, 150], 'song': [255, 224, 150]}

LOG_RULES = """You log footage for a fan story edit about a streamer storyline (the TikTok kind: their moments cut to a song,
their own words over the music). The images are frame sheets from ONE source, each frame labelled with its source time
and how many faces are visible. The transcript has word@start-second times.

List what an editor could use:
- lines: up to 6 short quotable lines (1.2-4 s) that tell the story or land emotionally (affection, a joke, a confession,
  a goodbye). a/b = the start of the first word and the end of the last word (from the word times; end = start of the
  next word). speaker: "him" (the streamer), "her" (the other person in the storyline), "other", or "unknown".
- shots: up to 10 visual moments worth a beat in a montage: faces, smiles, a hug, a kiss on the cheek, a gift, walking
  together, a reaction. t = the source time to start from. focus: where the people of the moment are in the frame:
  "left", "right", "both" (two people close together), "center". mood: happy, warm, sad, funny. quality 1-5 (5 = a
  great close, clear, emotional frame; 1 = dark, blurry, faces away).
Name people only from the cast list and only when you're sure; never guess identity from looks. Skip anything sexual.
Answer with JSON only: {"summary": "one line", "lines": [...], "shots": [...]}"""

PLAN_RULES = """You are the editor of a TikTok fan-edit page. Plan ONE story edit of this storyline, cut to the song below,
from the shot logs (each source has an id and a date). Rules (from what works in top story edits):
- cold_open: one line that needs no context, before the music drops (1.5-4 s). It should make people stay.
- acts: 2-4, in date order. Each has: date (e.g. "APRIL 2026"), mood (happy, warm or sad), shots (the montage, 4-12
  shots: happy montages cut every beat, so use more; sad ones every two beats), lines (1-3 of the logged lines, in the
  order they're said; they're held over the music). The first act starts on the drop with the title.
- recap: 4-6 of the best shots from earlier acts, a fast flashback before the end.
- final: the kindest or most telling closing line (logged), said over the recap; hold: the shot to end on.
- Total length 30-45 s at the song's tempo. Use each source moment once (a recap may repeat). Prefer quality 4-5 shots.
- Only use lines and shots from the logs, with their exact times and source ids. Don't invent facts or dates; dates come
  from the sources.
- title: 1-3 words for the drop (a ship name, their names, or the storyline's word), plus the post caption.
Answer with JSON only:
{"title": "...", "caption": "...", "cold_open": {"src": id, "a": s, "b": s, "speaker": "him|her|other"},
 "acts": [{"date": "...", "mood": "happy", "shots": [{"src": id, "t": s, "focus": "both"}],
           "lines": [{"src": id, "a": s, "b": s, "speaker": "her"}]}],
 "recap": [{"src": id, "t": s, "focus": "both"}], "final": {"src": id, "a": s, "b": s, "speaker": "him"},
 "hold": {"src": id, "t": s, "focus": "both"}}"""


def dur_of(f):
    import cv2
    c = cv2.VideoCapture(f)
    return c.get(7) / (c.get(5) or 30)


def shot_log(story, item, cast):
    """One source -> its lines and shots (cached in log/<id>.json). Long sources are logged in stretches."""
    import judge as J
    out_f = os.path.join(story, 'log', f"{item['id']}.json")
    if os.path.exists(out_f):
        return json.load(open(out_f))
    os.makedirs(os.path.dirname(out_f), exist_ok=True)
    f = os.path.join(story, item['file'])
    end = H.video_end(f, dur_of(f))
    W = H.hear(f, 0, end, os.path.join(story, 'log', f"{item['id']}.words.json"))
    span = 48 if end > 60 else end                    # long sources: a stretch of ~48 frames per call
    step = max(1.0, end / 48) if end <= 60 else max(1.0, min(8.0, end / 160))
    lines, shots, summ = [], [], []
    t = 0.0
    while t < end - 0.5:
        b = min(end, t + span * step)
        imgs, _ = H.sheets(f, t, b, step)
        ws = [w for w in W if t <= w['s'] < b]
        prompt = (LOG_RULES + f"\n\nCAST: {cast}\nSOURCE: {item['id']}, {item.get('date', '')}: {item.get('note', '')}\nSTRETCH: {t:.0f}-{b:.0f} s"
                  + "\nTRANSCRIPT (word@start seconds):\n" + (H.transcript_lines(ws) or '(no speech)'))
        try:
            raw = J._ask(prompt, imgs[:8], timeout=180)
            v = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
            lines += [x for x in v.get('lines') or [] if t - 1 <= x.get('a', -1) < x.get('b', -1) <= b + 1]
            shots += [x for x in v.get('shots') or [] if t - 1 <= x.get('t', -1) <= b]
            summ.append(v.get('summary') or '')
        except Exception as e:  # noqa: BLE001
            print(f"log {item['id']} {t:.0f}-{b:.0f}: {e}", file=sys.stderr)
        t = b
    log = {'id': item['id'], 'date': item.get('date'), 'note': item.get('note'), 'duration': round(end, 1), 'summary': ' '.join(summ)[:400],
           'lines': lines, 'shots': shots}
    json.dump(log, open(out_f, 'w'), indent=1, ensure_ascii=False)
    return log


def plan_story(logs, arc, song, title_hint):
    import judge as J
    period = (song['beats'][-1] - song['beats'][0]) / max(1, len(song['beats']) - 1)
    period = period * 2 if period < 0.45 else period
    prompt = (PLAN_RULES + f"\n\nSTORYLINE: {title_hint}\n{arc}\n\nSONG: {song['credit']}, {song['mood']}, a beat every {period:.2f} s; "
              f"happy montage shots last one beat ({period:.2f} s), sad ones two. Lines are held for their own length.\n\nSHOT LOGS:\n"
              + json.dumps([{k: l[k] for k in ('id', 'date', 'note', 'summary', 'lines', 'shots')} for l in logs], ensure_ascii=False))
    last = None
    ids = {l['id']: l for l in logs}
    for _ in range(2):
        raw = J._ask(prompt + (f"\n\nYOUR LAST PLAN HAD A PROBLEM: {last}. Answer again." if last else ''), (), timeout=240)
        try:
            p = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
            refs = [p['cold_open'], p['final'], p['hold']] + p['recap'] + [x for a in p['acts'] for x in a['shots'] + a['lines']]
            bad = [r for r in refs if r.get('src') not in ids]
            last = f"unknown source ids {[r.get('src') for r in bad]}" if bad else (None if p['acts'] else 'no acts')
        except Exception as e:  # noqa: BLE001
            p, last = None, f'invalid JSON: {e}'
        if p and not last:
            return p
    raise SystemExit(f'story plan unusable: {last}')


FOCUS = {'left': 'auto:left', 'right': 'auto:right', 'both': 'auto:both', 'center': 'auto'}
BLEEP = [(r'f+u+c+k', 'f**k'), (r'sh+i+t', 'sh*t'), (r'b+i+t+c+h', 'b*tch'), (r'n+i+g+g\w*', 'n***'), (r'\bf+a+g+\w*', 'f**'), (r'c+u+n+t', 'c**t'),
         (r'p+u+s+s+y', 'p***y'), (r'd+i+c+k', 'd*ck'), (r'r+e+t+a+r+d\w*', 'r****d')]


def mask(t):
    """On-screen words are masked the way the live cuts mask them."""
    for pat, rep in BLEEP:
        t = re.sub(pat, rep, t, flags=re.I)
    return t


def bright(f, t):
    import cv2
    c = cv2.VideoCapture(f); c.set(cv2.CAP_PROP_POS_MSEC, (t + 0.3) * 1000); ok, fr = c.read()
    return float(fr.mean()) if ok else 0.0


def faces_at(f, t):
    import cv2
    c = cv2.VideoCapture(f); c.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = c.read()
    if not ok:
        return 0
    sm = cv2.resize(fr, (640, int(fr.shape[0] * 640 / fr.shape[1])))
    det = cv2.FaceDetectorYN.create(studio.YUNET, '', (sm.shape[1], sm.shape[0]), 0.6, 0.3, 50)
    _, r = det.detect(sm)
    return 0 if r is None else len(r)


def fit(p, files, song, target=46.0):
    """Hold the plan to the length budget (the editor tends to over-fill), drop near-black shots, and end on the pair."""
    period = (song['beats'][-1] - song['beats'][0]) / max(1, len(song['beats']) - 1)
    if period < 0.45:
        period *= 2
    for a in p['acts']:
        a['shots'] = [s for s in a.get('shots') or [] if bright(files[s['src']], s['t']) > 40]
        a['lines'] = (a.get('lines') or [])[:2]
    p['recap'] = [s for s in p.get('recap') or [] if bright(files[s['src']], s['t']) > 40][:5]

    def length():
        n = 0.0
        for i, a in enumerate(p['acts']):
            per = 2 if a.get('mood') == 'sad' else 1
            n += sum(per if (a.get('mood') == 'sad' or j < 8) else 2 for j in range(len(a['shots']))) * period
            n += sum(max(2, math.ceil((l['b'] - l['a'] + 0.4) / period)) * period for l in a['lines'])
            if i and (a.get('mood') == 'sad' or p['acts'][i - 1].get('mood') == 'sad'):
                n += 2 * period
        return n + max(3.0, p['cold_open']['b'] - p['cold_open']['a'] + 0.6) + len(p['recap']) * period + 4.6

    while length() > target:
        big = max(p['acts'], key=lambda a: len(a['shots']))
        if len(big['shots']) > 4:
            big['shots'].pop()
        elif any(len(a['lines']) > 1 for a in p['acts']):
            next(a for a in p['acts'] if len(a['lines']) > 1)['lines'].pop()
        else:
            break
    # the title lands on the first montage shot: make it the brightest of the first three
    s0 = p['acts'][0]['shots']
    if len(s0) > 1:
        k = max(range(min(3, len(s0))), key=lambda i: bright(files[s0[i]['src']], s0[i]['t']))
        s0.insert(0, s0.pop(k))
    # end on the two of them: the brightest two-shot in the edit (both faces visible)
    hd = p.get('hold')
    if True:                                            # the editor's pick often lands on one person; the pair wins
        cands = [x for a in p['acts'] for x in a['shots'] if x.get('focus') == 'both'] + [x for x in p['recap'] if x.get('focus') == 'both']
        cands = [x for x in cands if faces_at(files[x['src']], x['t'] + 0.5) >= 2]
        if cands:
            p['hold'] = max(cands, key=lambda x: bright(files[x['src']], x['t']))
    # a line whose own picture has no face (the camera on the ceiling) plays over a two-shot of the act instead
    for a in p['acts']:
        two = [x for x in a['shots'] if x.get('focus') == 'both'] or a['shots']
        for ln in a['lines']:
            if faces_at(files[ln['src']], (ln['a'] + ln['b']) / 2) == 0 and two:
                ln['cutaway'] = two[len(a['lines']) % len(two)]
    return p


CAST = set()


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def names(t):
    """Snap near-miss spellings of the cast's names to the real ones ("lacey" -> "lacy")."""
    def fix(m):
        w = m.group(0)
        c = next((n for n in CAST if len(w) >= 4 and w[0] == n[0] and w != n and lev(w, n) <= 1), None)
        return c or w
    return re.sub(r"[a-z0-9]+", fix, t)


def build(p, story, pool, song, credit, handle):
    files = {x['id']: os.path.join(story, x['file']) for x in pool}
    shots, layers, voice, texts, trans, hits = [], [], [], [], [], []
    PUSH = [[0, 1.02, 0, 0, 0], [1, 1.12, 0, 0, 0]]; PULL = [[0, 1.14, 0, 0, 0], [1, 1.02, 0, 0, 0]]

    def shot(src, at, end, i, focus='both', look=None, speed=1.0, cam=None, into=None):
        # faces first: a single person is framed close (the face high in the frame), a pair a little wider
        shots.append({'src': files[src], 'in': round(max(0, i), 3), 'at': round(at, 3), 'dur': round(end - at, 3), 'speed': speed,
                      'focus': FOCUS.get(focus, 'auto:both'), 'cam': cam or PUSH, 'look': look or [], 'headroom': 0.1,
                      'face_fill': 0.21 if focus == 'both' else 0.31})
        if into:
            trans.append({'at': round(at, 3), 'type': into, 'dur': {'flash': 0.3, 'dissolve': 0.3, 'black': 0.3}.get(into, 0.36)})

    def say(ln, at, duck=0.3):
        voice.append({'src': files[ln['src']], 'in': ln['a'], 'out': ln['b'], 'at': round(at, 3), 'duck': duck})

    def text(t0, t1, txt, style='lyric', anim='type', y=1400, color=COL['him']):
        texts.append({'t0': round(t0, 3), 't1': round(t1, 3), 'text': txt, 'style': style, 'anim': anim, 'y': y, 'color': color})

    def words(ln):
        # the shot logger's own text for this line when it has it (it reads context: "MCS" heard, "FNCS" meant)
        try:
            logged = json.load(open(os.path.join(story, 'log', f"{ln['src']}.json")))['lines']
            hit = next((x for x in logged if abs(x['a'] - ln['a']) < 0.35 and abs(x['b'] - ln['b']) < 0.6 and x.get('text')), None)
        except (OSError, ValueError, KeyError):
            hit = None
        if hit:
            return mask(names(hit['text'].lower().strip(' .,')))
        W = json.load(open(os.path.join(story, 'log', f"{ln['src']}.words.json")))
        ws = [w['w'] for w in W if ln['a'] - 0.05 <= w['s'] < ln['b'] - 0.05]
        out = []
        for w in ws:                                   # "ah, ah, ah, ah" -> "ah,"; "okay okay okay" -> "okay"
            if out and re.sub(r'\W', '', w.lower()) == re.sub(r'\W', '', out[-1].lower()):
                continue
            out.append(w)
        return mask(names(' '.join(out).lower().strip(' .,')))

    p = fit(p, files, song)
    # cold open (music muffled), then the drop
    co = p['cold_open']; L = co['b'] - co['a']
    DROP = round(max(3.0, L + 0.6), 3)
    drop_song = H.best_drop(song, DROP) if any(d >= DROP for d in song['drops']) else song['drops'][0]
    start = round(drop_song - DROP, 3)
    B = [round(b - start, 3) for b in song['beats'] if b >= drop_song - 0.02]
    if len(B) > 2 and (B[-1] - B[0]) / (len(B) - 1) < 0.45:      # a fast (double-time) grid: a story cuts on every other beat
        B = B[::2]
    say(co, 0.3, duck=0.6)
    shot(co['src'], 0.0, DROP, co['a'] - 0.3, focus='both', cam=[[0, 1.04, 0, 0, 0], [1, 1.2, 0, 0, 0]])
    text(0.4, DROP - 0.1, words(co), color=COL.get(co.get('speaker'), COL['other']))
    acts = p['acts']
    first_date = acts[0].get('date', '')
    if first_date:
        text(0.2, DROP - 0.1, first_date, 'date', 'fade', y=300)
    k = 0                                                  # beat index
    sad_story = any(a.get('mood') == 'sad' for a in acts)
    for ai, act in enumerate(acts):
        mood = act.get('mood', 'happy')
        look = ['bw'] if mood == 'sad' else []
        if ai > 0:
            # a black card with the date is an act break for a sad turn (or the turn back from one); a happy story keeps moving
            if mood == 'sad' or acts[ai - 1].get('mood') == 'sad':
                trans.append({'at': B[k], 'type': 'black', 'dur': 0.3})
                shots.append({'src': None, 'at': B[k], 'dur': round(B[k + 2] - B[k], 3)})
                text(B[k] + 0.1, B[k + 2] - 0.05, act.get('date', ''), 'date', 'fade', y=960)
                k += 2
            elif act.get('date') and act['date'] != acts[ai - 1].get('date'):
                text(B[k] + 0.05, B[k + 3], act['date'].lower(), 'lyric', 'fade', y=300)
        else:
            trans.append({'at': B[0], 'type': 'flash', 'dur': 0.3})
            hits.append({'at': B[0], 'type': 'shake', 'amt': 1.0})
            text(B[0] + 0.05, B[2] - 0.05, p['title'].upper(), 'title', 'pop', y=960)
        per = 2 if mood == 'sad' else 1
        for j, s in enumerate(act.get('shots') or []):
            n = per if (mood == 'sad' or j < 8) else 2
            into = (['zoom', 'whip', 'flash'][j % 3] if j % 3 == 0 and mood != 'sad' else ('dissolve' if mood == 'sad' else None))
            shot(s['src'], B[k], B[k + n], s['t'], s.get('focus'), look, speed=0.5 if mood == 'sad' else 1.0, cam=PUSH if j % 2 == 0 else PULL, into=into)
            if mood != 'sad' and not into:
                hits.append({'at': B[k] - 1 / 30, 'type': 'pulse', 'amt': 0.9})
            k += n
        for ln in act.get('lines') or []:
            Ld = ln['b'] - ln['a']
            n = max(2, math.ceil((Ld + 0.4) / (B[k + 1] - B[k])))
            say(ln, B[k] + 0.12, duck=0.28)
            cw = ln.get('cutaway')
            if cw:
                shot(cw['src'], B[k], B[k + n], cw['t'], cw.get('focus'), look + (['glow'] if mood != 'sad' else []), speed=0.6, into='dissolve')
            else:
                shot(ln['src'], B[k], B[k + n], ln['a'] - 0.12, 'both', look + (['glow'] if mood != 'sad' else []), into='dissolve')
            text(B[k] + 0.2, B[k + n] - 0.05, words(ln), color=COL.get(ln.get('speaker'), COL['other']))
            k += n
    # recap + the last line + hold
    fin = p.get('final')
    rc = p.get('recap') or []
    rlook = ['bw'] if sad_story else ['warm']
    if fin:
        say(fin, B[k] + 0.1, duck=0.3)
    r0 = k
    for j, s in enumerate(rc[:6]):
        shot(s['src'], B[k], B[k + 1], s['t'], s.get('focus'), rlook, cam=PUSH if j % 2 == 0 else PULL)
        hits.append({'at': B[k] - 1 / 30, 'type': 'pulse', 'amt': 1.2})
        k += 1
    if rc:
        trans.append({'at': B[r0], 'type': 'flash', 'dur': 0.3})
    hd = p.get('hold') or (rc[0] if rc else None)
    end = B[k] + 4.6
    if fin:
        end = max(end, B[r0] + 0.1 + (fin['b'] - fin['a']) + 2.0)
    shot(hd['src'], B[k], end, hd['t'], hd.get('focus'), rlook + ['glow'], speed=0.5, cam=[[0, 1.06, 0, 0, 0], [1, 1.3, 0, 0, 0]], into='dissolve')
    if fin:
        text(B[r0] + 0.2, B[r0] + 0.2 + (fin['b'] - fin['a']) + 0.6, words(fin), color=COL.get(fin.get('speaker'), COL['other']))
    text(end - 2.4, end - 0.2, p['title'].lower(), 'lyric', 'fade', y=960)
    if os.environ.get('CUTROOM_ONSCREEN_CREDIT') == '1':   # no watermark by default (the user, 2026-10-21: not a program rule)
        texts.append({'t0': 0.0, 't1': round(end - 2.6, 3), 'text': f'{credit} · @{handle}', 'style': 'mark', 'anim': 'none', 'y': 1855, 'opacity': 0.7})
    sfx = [{'at': B[0], 'type': 'riser', 'dur': min(2.6, DROP - 0.2), 'gain': 0.3}, {'at': B[0], 'type': 'boom', 'gain': 0.5}]
    sfx += [{'at': t['at'], 'type': 'whoosh', 'gain': 0.3} for t in trans if t['type'] in ('whip', 'zoom')]
    if os.environ.get('CUTROOM_RETRO_FX') != '1':
        sfx = []                                         # no synthesized riser/boom/whoosh (they read as cheap; STYLE-GUIDE.md)
    E = {'size': [1080, 1920], 'fps': 30, 'duration': round(end, 3),
         'music': {'src': song['path'], 'start': start, 'gain': 0.9, 'drop_gain': 0.55, 'fade_in': 0.6, 'fade_out': 2.5, 'muffle_until': DROP, 'levels': {'pre': -20, 'post': -13}},
         'voice': voice, 'shots': sorted(shots, key=lambda x: x['at']), 'layers': layers, 'transitions': trans, 'hits': hits, 'sfx': sfx,
         'texts': texts, 'finish': {'grain': 0.03, 'vignette': 0.3, 'fade_out': 1.4}}
    sh = E['shots']
    for x, y in zip(sh, sh[1:]):
        assert abs(x['at'] + x['dur'] - y['at']) < 0.02, (x['at'], x['dur'], y['at'])
    return E


CINE_RULES = """You are the editor of a page that makes cinematic story edits of streamers, like the best fan edits of TV
characters (a Harvey Specter edit): the story is told in the people's OWN WORDS, line after line, like one continuous
monologue or conversation, while the picture cuts between shots of them. Then a short montage builds to the end.

From the shot logs (each source has an id and a date), plan ONE edit:
- lines: 8-14 logged lines, in the order they should be heard. Strung back to back they must tell the storyline so a
  stranger follows it: who they are to each other, the turn, how it ends. Open on a line that introduces them, the way
  the best edits do ("This is Harvey Specter. He's our best closer."): one of them naming or describing the other, or
  saying what they are to each other. Never open on a line that needs context. Use exact logged times and source ids.
  Prefer lines that sound like statements and confessions ("she truly was my best friend") over filler and questions.
- For each line, cover: 0-2 shots (src, t, focus) to cut to while the line keeps playing: reactions, the other person,
  the place. A line about someone shows that someone.
- montage: 8-12 of the best visual shots for the ending, in story order, faces and emotion, quality 4-5.
- Stay on the story's people: when a logged shot or line has cast_on_screen, prefer 2 (both of them) for covers, the
  montage and the hold; never build a stretch around someone else.
- final: the line to end on (logged), the kindest or most telling; hold: the shot to end on.
- title: 1-3 words shown once, big, when the montage starts (a ship name, their names, the storyline's word).
- intro: one plain line (max 9 words) that tells a stranger who these people are to each other, shown small over the
  first seconds, by their roles ("his best friend", "his little bro", "his biggest fan", "his dad"); add the year when
  the story spans time ("2024: Lacy and Marlon, roommates"). Only facts from the storyline or the logs.
- caption: the post caption as the 100-300M-view story pages write it: who + the change or the turn + where it lands
  ("Speed made her a Ronaldo fan 🥹", "Ray has changed...", "Kai's dad finally sees him live 🥹"). One sentence.
Total dialogue 25-40 s. Only use logged lines and shots. Never invent.
Answer with JSON only:
{"title": "...", "intro": "...", "caption": "...", "lines": [{"src": id, "a": s, "b": s, "speaker": "him|her|other", "cover": [{"src": id, "t": s, "focus": "both"}]}],
 "montage": [{"src": id, "t": s, "focus": "both"}], "final": {"src": id, "a": s, "b": s, "speaker": "him"}, "hold": {"src": id, "t": s, "focus": "both"}}"""


def plan_cine(logs, arc, title_hint):
    import judge as J
    prompt = (CINE_RULES + f"\n\nSTORYLINE: {title_hint}\n{arc}\n\nSHOT LOGS:\n"
              + json.dumps([{k: l[k] for k in ('id', 'date', 'note', 'summary', 'lines', 'shots')} for l in logs], ensure_ascii=False))
    ids = {l['id'] for l in logs}
    last = None
    for _ in range(2):
        raw = J._ask(prompt + (f"\n\nYOUR LAST PLAN HAD A PROBLEM: {last}. Answer again." if last else ''), (), timeout=240)
        try:
            p = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
            refs = p['lines'] + p['montage'] + [p['final'], p['hold']] + [c for l in p['lines'] for c in l.get('cover') or []]
            bad = [r.get('src') for r in refs if r.get('src') not in ids]
            last = f'unknown source ids {bad}' if bad else (None if len(p['lines']) >= 5 else 'too few lines')
        except Exception as e:  # noqa: BLE001
            p, last = None, f'invalid JSON: {e}'
        if p and not last:
            return p
    raise SystemExit(f'cine plan unusable: {last}')


CASTINFO = {}


def usable(f, t):
    """A shot a stranger can read on a phone: not dark, not motion-blurred (the critic's most common note on our stories)."""
    import cv2
    c = cv2.VideoCapture(f); c.set(cv2.CAP_PROP_POS_MSEC, (t + 0.4) * 1000); ok, fr = c.read()
    if not ok:
        return False
    g = cv2.cvtColor(cv2.resize(fr, (640, 360)), cv2.COLOR_BGR2GRAY)
    return float(g.mean()) >= 45 and float(cv2.Laplacian(g, cv2.CV_64F).var()) >= 35


def on_cast(x):
    """A planned shot shows the story's people (castcam.py), or we can't tell (no cast info)."""
    if not CASTINFO:
        return True
    import castcam
    return castcam.cast_at(CASTINFO, x['src'], x['t'] + 0.5) >= 1


def build_cine(p, story, pool, song, credit, handle, size=(1920, 1080)):
    """The reference recipe (research/references): 16:9, their own lines back to back with the picture cutting under them
    (J-cuts), a montage that builds to the song's drop, a held last shot. A steady music bed lightly ducked under the voices,
    no effects; small white caps; a film grade."""
    files = {x['id']: os.path.join(story, x['file']) for x in pool}
    W_, H_ = size
    shots, voice, texts = [], [], []
    GRADE = ['film']
    cap_y = int(H_ * 0.86)

    def shot(src, at, dur, i, focus='both', speed=1.0, look=None):
        if dur < 0.2:
            return
        shots.append({'src': files[src], 'in': round(max(0, i), 3), 'at': round(at, 3), 'dur': round(dur, 3), 'speed': speed,
                      'focus': {'left': 'auto:left', 'right': 'auto:right', 'both': 'auto:both'}.get(focus, 'auto'),
                      'cam': [[0, 1.0, 0, 0, 0], [1, 1.035, 0, 0, 0]], 'look': look or GRADE, 'headroom': 0.25, 'balance': True})

    def words_of(ln):
        Wf = os.path.join(story, 'log', f"{ln['src']}.words.json")
        ws = [w for w in (json.load(open(Wf)) if os.path.exists(Wf) else []) if ln['a'] - 0.05 <= w['s'] < ln['b'] - 0.05]
        return ws

    def captions(ln, at):
        ws = words_of(ln)
        groups, cur = [], []
        for w in ws:
            if cur and (len(cur) >= 4 or w['s'] - cur[-1]['s'] > 0.6):
                groups.append(cur); cur = []
            cur.append(w)
        if cur:
            groups.append(cur)
        for k, g in enumerate(groups):
            t0 = at + g[0]['s'] - ln['a']
            t1 = at + (groups[k + 1][0]['s'] - ln['a'] if k + 1 < len(groups) else (ln['b'] - ln['a']))
            texts.append({'t0': round(t0, 3), 't1': round(max(t1, t0 + 0.3), 3), 'text': mask(names(' '.join(w['w'].strip('.,') for w in g))),
                          'style': 'cinecap', 'anim': 'none', 'y': cap_y, 'color': [255, 255, 255]})

    if p.get('intro'):                                  # who they are, for a stranger, over the first seconds
        texts.append({'t0': 0.15, 't1': 3.6, 'text': mask(p['intro'])[:70], 'style': 'cinecap', 'anim': 'fade', 'y': int(H_ * 0.12), 'color': [255, 255, 255]})
    t = 0.0
    for ln in p['lines']:
        L = ln['b'] - ln['a']
        voice.append({'src': files[ln['src']], 'in': ln['a'], 'out': ln['b'], 'at': round(t + 0.05, 3), 'duck': 0.6})
        captions(ln, t + 0.05)
        cover = [c for c in (ln.get('cover') or []) if c.get('src') in files and on_cast(c) and usable(files[c['src']], c['t'])][:2]
        seg = L + 0.3                                     # the line plus a breath
        own = seg if not cover else max(1.0, min(seg * 0.5, 2.2))
        if cover and not usable(files[ln['src']], ln['a']):   # the speaker's own frame is dark or smeared: the cover takes the line
            own = 0.0
        shot(ln['src'], t, own, ln['a'] - 0.05, 'both')
        rest = seg - own
        for j, c in enumerate(cover):
            d = rest / len(cover)
            shot(c['src'], t + own + j * d, d, c['t'], c.get('focus'))
        t += seg
    # the montage, on the beat, building: two beats a shot, then one; the song's drop lands where it starts
    drop_song = next((d for d in song['drops'] if d >= t), song['drops'][-1])
    mstart = drop_song - t
    mat = 0.0
    if mstart < 0:
        mat, mstart = -mstart, 0.0
    B = [b - mstart + mat for b in song['beats'] if b >= drop_song - 0.02]
    if len(B) > 2 and (B[-1] - B[0]) / (len(B) - 1) < 0.45:
        B = B[::2]
    mont = [m for m in p.get('montage') or [] if m.get('src') in files and on_cast(m) and usable(files[m['src']], m['t'])][:12]
    k = 0
    m0 = B[0]
    if B[0] > t:                                          # fill to the drop with the last line's shot
        shots[-1]['dur'] = round(B[0] - shots[-1]['at'], 3)
    for j, m in enumerate(mont):
        n = 2 if j < 4 else 1
        if k + n >= len(B):
            break
        shot(m['src'], B[k], B[k + n] - B[k], m['t'], m.get('focus'))
        k += n
    texts.append({'t0': round(m0 + 0.05, 3), 't1': round(min(B[min(k, len(B) - 1)], m0 + 1.6), 3), 'text': (p.get('title') or '').upper(),
                  'style': 'cinetitle', 'anim': 'fade', 'y': H_ // 2, 'color': [255, 255, 255]})
    # the last line over the held shot, then out
    t = B[k] if k < len(B) else shots[-1]['at'] + shots[-1]['dur']
    fin, hd = p.get('final'), p.get('hold') if p.get('hold') and on_cast(p['hold']) else (mont[-1] if mont else None)
    hold = 3.5
    if fin and fin.get('src') in files:
        voice.append({'src': files[fin['src']], 'in': fin['a'], 'out': fin['b'], 'at': round(t + 0.3, 3), 'duck': 0.6})
        captions(fin, t + 0.3)
        hold = max(hold, fin['b'] - fin['a'] + 1.6)
    if hd and hd.get('src') in files:
        shot(hd['src'], t, hold, hd['t'], hd.get('focus'), speed=0.85)
    end = t + hold
    if os.environ.get('CUTROOM_ONSCREEN_CREDIT') == '1':   # no watermark by default (the user, 2026-10-21: not a program rule)
        texts.append({'t0': 0.0, 't1': round(end, 3), 'text': f'{credit} · @{handle}', 'style': 'mark', 'anim': 'none', 'y': H_ - 36, 'opacity': 0.55})
    shots.sort(key=lambda x: x['at'])
    for x, y in zip(shots, shots[1:]):                    # close any gap from rounding
        x['dur'] = round(y['at'] - x['at'], 3)
    E = {'size': [W_, H_], 'fps': 30, 'duration': round(end, 3),
         'music': {'src': song['path'], 'start': round(mstart, 3), 'at': round(mat, 3), 'gain': 0.7, 'fade_in': 1.2, 'fade_out': 2.5, 'levels': {'post': -17}},
         'voice': voice, 'shots': shots, 'transitions': [], 'hits': [], 'sfx': [], 'texts': texts,
         'finish': {'grain': 0.02, 'vignette': 0.25, 'fade_out': 1.2}}
    return E


MINI_RULES = """You are the editor of a page that makes vertical MINI-STORY edits of streamers, the kind that reach 100-300M views
(IShowSpeed fan pages: "Speed made his day", "The ending", "We made the right person famous"): a three-beat story under
45 s, told with the real moments cut as fast jump cuts (picture and sound together), captioned, with a soft music bed.

It tells ONE event, like those pages do: one encounter, one bit, one argument that resolves. Take every segment from ONE
source (two only if both show the same event). Choose the source whose logged lines hold a complete setup, turn and
payoff; the viewers' view count of each source tells you which moments people already found worth clipping.
From the shot logs, plan ONE edit:
- beats: exactly 3, in order: "setup" (the promise or the conflict, understood by a stranger in the first 3 s: a line
  that names the situation), "turn" (what happens), "payoff" (the emotional landing: the reveal, the hug, the laugh).
  Each beat has 2-6 segments in order. A segment is a stretch of one source where something is said or happens:
  {"src": id, "a": s, "b": s}, 1.2-6 s long, using exact logged times (a line's a/b, or a shot's t to t+2.5).
- hold: the final reaction to hold under the label (a segment, 2-4 s).
- label: the payoff label shown over the hold, 1-3 words in asterisks plus one emoji, the way those pages write it
  ("*FORGIVES FAN* ❤️‍🩹", "*NEW RONALDO FAN* 🥶", "*SHE WAS A KEEPER*", "*BEST FRIENDS* 🫂").
- intro: the premise in max 8 words, shown big over the first 3 s, so a stranger knows what they're watching
  ("Marlon asks her on a trip... as friends?", "his best friend Marlon"). Never empty.
- caption: the post caption: who + the change or the turn + where it lands, one sentence + emoji.
- mood: one of wholesome, funny, hype, sad (it picks the music bed).
- title: 1-4 words.
Total 25-45 s. Prefer moments with clear faces and real reactions; skip filler. Only logged lines and shots. Never invent.
Answer with JSON only:
{"title": "...", "caption": "...", "mood": "...", "intro": "...", "label": "...",
 "beats": [{"name": "setup", "segments": [{"src": id, "a": s, "b": s}]}, {"name": "turn", "segments": [...]}, {"name": "payoff", "segments": [...]}],
 "hold": {"src": id, "a": s, "b": s}}"""

MOOD_SONG = {'wholesome': 'until-i-found-you', 'sad': 'snowfall', 'funny': 'funk-do-bounce', 'hype': 'montagem-coral'}


def plan_mini(logs, arc, title_hint):
    import judge as J
    prompt = (MINI_RULES + f"\n\nSTORYLINE: {title_hint}\n{arc}\n\nSHOT LOGS:\n"
              + json.dumps([{**{k: l[k] for k in ('id', 'date', 'note', 'summary', 'lines', 'shots')}, 'views': l.get('views')} for l in logs], ensure_ascii=False))
    ids = {l['id'] for l in logs}
    last = None
    for _ in range(2):
        raw = J._ask(prompt + (f"\n\nYOUR LAST PLAN HAD A PROBLEM: {last}. Answer again." if last else ''), (), timeout=240)
        try:
            p = json.loads(re.search(r'\{.*\}', raw, re.S).group(0))
            segs = [x for b in p['beats'] for x in b['segments']] + [p['hold']]
            bad = [x.get('src') for x in segs if x.get('src') not in ids]
            n_src = len({x.get('src') for x in segs})
            last = (f'unknown source ids {bad}' if bad else 'it must tell ONE event: segments from one source (two at most)' if n_src > 2
                    else None if len(p['beats']) == 3 and len(segs) >= 6 else 'needs 3 beats and 6+ segments')
        except Exception as e:  # noqa: BLE001
            p, last = None, f'invalid JSON: {e}'
        if p and not last:
            return p
    raise SystemExit(f'mini plan unusable: {last}')


def build_mini(p, story, pool, song, credit, handle, size=(1080, 1920)):
    """The vertical mini-story (STYLE-GUIDE.md, the 20-323M-view story edits): jump cuts of the real moments with their own
    sound, a face-tracked vertical crop, phrase captions in big caps, a smear only where the beat changes, the payoff
    label with an emoji over a held reaction, a soft music bed under the speech (sidechain-ducked in the studio mix)."""
    files = {x['id']: os.path.join(story, x['file']) for x in pool}
    W_, H_ = size
    shots, voice, texts, trans = [], [], [], []
    look = {'wholesome': ['warm', 'glow'], 'sad': ['sad'], 'funny': ['film'], 'hype': ['film']}.get(p.get('mood'), ['film'])

    keyw = {re.sub(r"[^a-z0-9]", '', w.lower())[:5] for w in f"{p.get('caption', '')} {p.get('label', '')} {p.get('title', '')}".split()}
    keyw = {w for w in keyw if len(w) >= 4 and w not in ('that', 'this', 'with', 'they', 'them', 'then', 'from', 'what', 'when', 'just', 'like', 'have',
                                                          'dont', 'want', 'more', 'than', 'into', 'your', 'youre', 'were', 'will', 'been', 'turns', 'gets')}

    def accent(wd):
        t = re.sub(r"[^A-Za-z0-9]", '', wd)
        return bool(t) and (any(ch.isdigit() for ch in t) or t.lower()[:5] in keyw or t.lower() in CAST)

    def words(src, a, b):
        Wf = os.path.join(story, 'log', f"{src}.words.json")
        return [w for w in (json.load(open(Wf)) if os.path.exists(Wf) else []) if a - 0.05 <= w['s'] < b - 0.05]

    def seg(x, at, speed=1.0, hold=None):
        a, b = float(x['a']), float(x['b'])
        dur = hold or (b - a) / speed
        push = [[0, 1.0, 0, 0, 0], [1, 1.18, 0, 0, 0]] if hold else [[0, 1.0, 0, 0, 0], [1, 1.06, 0, 0, 0]]   # the payoff: a slow push in
        shots.append({'src': files[x['src']], 'in': round(max(0, a), 3), 'at': round(at, 3), 'dur': round(dur, 3), 'speed': speed,
                      'focus': 'auto:both', 'cam': push, 'look': look, 'headroom': 0.12, 'face_fill': 0.21, 'balance': True})
        if speed == 1.0:
            voice.append({'src': files[x['src']], 'in': a, 'out': b, 'at': round(at, 3), 'duck': 0.5})
            ws = words(x['src'], a, b)
            grp, cur = [], []
            for w in ws:
                if cur and (len(cur) >= 3 or w['s'] - cur[-1]['s'] > 0.5):
                    grp.append(cur); cur = []
                cur.append(w)
            if cur:
                grp.append(cur)
            for k, g in enumerate(grp):
                t0 = at + g[0]['s'] - a
                t1 = at + (grp[k + 1][0]['s'] - a if k + 1 < len(grp) else min(b - a, g[-1]['e'] - a + 0.25))
                line = mask(names(' '.join(w['w'].strip('.,') for w in g))).upper()
                wds = line.split(' ')
                cand = [i for i, wd in enumerate(wds) if accent(wd)]
                if cand:                                    # one key word per line in yellow: the longest (the rarest, usually)
                    i = max(cand, key=lambda i: len(re.sub(r'[^A-Za-z0-9]', '', wds[i])))
                    wds[i] = '§' + wds[i]
                line = ' '.join(wds)
                texts.append({'t0': round(t0, 3), 't1': round(max(t1, t0 + 0.3), 3), 'text': line,
                              'style': 'hype', 'anim': 'pop', 'y': int(H_ * 0.62), 'color': [255, 255, 255]})
        return dur

    t = 0.0
    for bi, beat in enumerate(p['beats']):
        for si, x in enumerate(beat['segments']):
            if x.get('src') not in files or float(x['b']) - float(x['a']) < 0.6:
                continue
            if bi > 0 and si == 0 and shots:
                trans.append({'at': round(t, 3), 'type': 'auto', 'dur': 0.32})     # the beat changes: a motion-matched move
            t += seg(x, t)
    hd = p.get('hold')
    hold_at = t
    if hd and hd.get('src') in files:
        trans.append({'at': round(t, 3), 'type': 'dissolve', 'dur': 0.4})
        hold_t = 3.2
        seg(hd, t, speed=0.85, hold=hold_t)
        texts.append({'t0': round(t + 0.35, 3), 't1': round(t + hold_t, 3), 'text': p.get('label') or '', 'style': 'stage', 'anim': 'pop',
                      'y': int(H_ * 0.5), 'color': [255, 214, 10]})
        t += hold_t
    if p.get('intro'):
        texts.append({'t0': 0.1, 't1': 3.0, 'text': mask(p['intro']), 'style': 'hook', 'anim': 'pop', 'y': 300, 'color': [255, 255, 255]})
    if os.environ.get('CUTROOM_ONSCREEN_CREDIT') == '1':   # no watermark by default (the user, 2026-10-21: not a program rule)
        texts.append({'t0': 0.0, 't1': round(t, 3), 'text': f'{credit} · @{handle}', 'style': 'mark', 'anim': 'none', 'y': H_ - 64, 'opacity': 0.6})
    shots.sort(key=lambda x: x['at'])
    for x, y in zip(shots, shots[1:]):
        x['dur'] = round(y['at'] - x['at'], 3)
    # the song's drop lands on the payoff (the hold), and the bed swells into it
    pay = hold_at if hd and hd.get('src') in files else max(0.0, t - 3.0)
    drop = H.best_drop(song, pay) if any(d >= pay for d in song.get('drops') or []) else None   # no hole in the run-up
    start = round(drop - pay, 3) if drop is not None else 0.0
    return {'size': [W_, H_], 'fps': 30, 'duration': round(t, 3),
            'music': {'src': song['path'], 'start': start, 'gain': 0.7, 'fade_in': 0.8, 'fade_out': 1.5, 'swell_at': round(max(0.0, pay - 0.6), 3), 'levels': {'post': -17}},
            'voice': voice, 'shots': shots, 'transitions': trans, 'hits': [], 'sfx': [], 'texts': texts,
            'finish': {'grain': 0.02, 'vignette': 0.2, 'fade_out': 0.6}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('story'); ap.add_argument('--name', required=True); ap.add_argument('--streamer', default='')
    ap.add_argument('--song', default='snowfall'); ap.add_argument('--credit', default=''); ap.add_argument('--handle', default='cutroom')
    ap.add_argument('--arc', default=''); ap.add_argument('--cast', default=''); ap.add_argument('--plan'); ap.add_argument('--id')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--mode', default='beat', choices=['beat', 'cine', 'mini'], help='beat: vertical, cut to the beat; cine: the reference recipe (16:9, dialogue-led); mini: the vertical three-beat mini-story (jump cuts, captions, a payoff label)')
    ap.add_argument('--cast-k', type=int, default=0, help='the story is about this many people (2 for a duo): only shots with them on screen')
    a = ap.parse_args()
    pool = json.load(open(os.path.join(a.story, 'pool.json')))
    import lore
    s = lore.find(a.streamer)
    arc = next((x for x in (s or {}).get('arcs') or [] if a.arc and a.arc.lower() in (x.get('title') or '').lower()), None)
    arc_txt = json.dumps({k: arc.get(k) for k in ('title', 'summary', 'timeline', 'beats', 'edit_angles') if k in arc}, ensure_ascii=False)[:3000] if arc else ''
    cast = a.cast or ', '.join((lore.brief(a.streamer) or {}).get('cast') or [])
    CAST.update(w for w in re.findall(r'[a-z0-9]+', (cast + ' ' + a.streamer).lower()) if len(w) >= 4 and w not in ('with', 'the', 'his', 'her', 'duo', 'partner'))
    logs = [shot_log(a.story, it, f"{a.streamer} (him); {cast}") for it in pool]
    if a.cast_k:                                     # who recurs on screen: shots without the cast don't make the edit
        import castcam
        CASTINFO.update(castcam.cast(a.story, pool, a.cast_k))
        logs = [dict(l, shots=[dict(x, cast_on_screen=castcam.cast_at(CASTINFO, l['id'], x['t'] + 0.5)) for x in l['shots']],
                     lines=[dict(x, cast_on_screen=castcam.cast_at(CASTINFO, l['id'], (x['a'] + x['b']) / 2)) for x in l['lines']]) for l in logs]
        logs = [dict(l, shots=[x for x in l['shots'] if x['cast_on_screen']], lines=[x for x in l['lines'] if x['cast_on_screen']]) for l in logs]
        print('cast on screen:', sum(len(l['shots']) for l in logs), 'shots and', sum(len(l['lines']) for l in logs), 'lines kept')
    song = H.song(a.song)
    if a.mode == 'mini':
        vmap = {it['id']: it.get('views') for it in pool}
        logs = [dict(l, views=vmap.get(l['id'])) for l in logs]
        plan = json.load(open(a.plan)) if a.plan else plan_mini(logs, arc_txt, a.name)
        json.dump(plan, open(os.path.join(a.story, 'plan-mini.json'), 'w'), indent=1, ensure_ascii=False)
        if a.song == 'snowfall' and plan.get('mood') in MOOD_SONG:          # the plan's mood picks the bed unless one was asked for
            song = H.song(H.pick_song(plan['mood'], a.name))                 # the popular TikTok sounds for that mood first
        E = build_mini(plan, a.story, pool, song, a.credit, a.handle)
    elif a.mode == 'cine':
        plan = json.load(open(a.plan)) if a.plan else plan_cine(logs, arc_txt, a.name)
        json.dump(plan, open(os.path.join(a.story, 'plan-cine.json'), 'w'), indent=1, ensure_ascii=False)
        E = build_cine(plan, a.story, pool, song, a.credit, a.handle)
    else:
        plan = json.load(open(a.plan)) if a.plan else plan_story(logs, arc_txt, song, a.name)
        json.dump(plan, open(os.path.join(a.story, 'plan.json'), 'w'), indent=1, ensure_ascii=False)
        E = build(plan, a.story, pool, song, a.credit, a.handle)
    import castcam                                   # stream overlays (chat, counters, other pages' captions) cropped out
    n = castcam.regions_for(a.story, E['shots'])
    print(f'{n} of {len(E["shots"])} shots re-framed to keep clear of overlays')
    E['mutes'] = {os.path.join(a.story, it['file']): studio.slur_mutes(json.load(open(wf))) for it in pool    # slurs muted in the voice lines
                  if os.path.exists(wf := os.path.join(a.story, 'log', f"{it['id']}.words.json"))}
    json.dump(E, open(os.path.join(a.story, {'cine': 'edit-cine.json', 'mini': 'edit-mini.json'}.get(a.mode, 'edit.json')), 'w'), indent=1, ensure_ascii=False)
    cid = a.id or os.path.basename(os.path.abspath(a.story)) + {'cine': '-cine', 'mini': '-mini'}.get(a.mode, '')
    mp4 = os.path.join(a.story, f'{cid}.mp4')
    studio.render(E, mp4, a.workers)
    import tempfile, shutil
    site = os.path.join(a.story, f'{cid}-site.mp4'); td = tempfile.mkdtemp(prefix='storyedit-'); tmp = os.path.join(td, 'site.wav')
    studio.mix_audio({**E, 'music': None}, tmp, td)
    subprocess.run([studio.FF, '-nostdin', '-y', '-loglevel', 'error', '-i', mp4, '-i', tmp, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac',
                    '-b:a', '160k', '-shortest', '-movflags', '+faststart', site], check=True)
    shutil.rmtree(td, ignore_errors=True)
    json.dump({'id': cid, 'title': f"{a.name}: {plan['title']}", 'kind': 'story', 'duration': E['duration'], 'handle': a.handle, 'credit': a.credit,
               'music': f"{song['credit']}: baked into the TikTok copy; the site copy has none", 'post_caption': plan.get('caption') or a.name,
               'sound': {'name': song['credit'], 'tiktok': song.get('tiktok'), 'start': round((E.get('music') or {}).get('start', 0), 1),
                         'at': round((E.get('music') or {}).get('at', 0), 1)},
               'made_by': 'storyedit (shot logs + plan) + studio'}, open(os.path.join(a.story, f'{cid}.json'), 'w'), indent=1, ensure_ascii=False)
    print(f"{mp4}: {E['duration']:.1f} s · {plan['title']}")


if __name__ == '__main__':
    main()
