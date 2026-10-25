#!/usr/bin/env python3
"""The semantic scan: the room reads its own transcript like a clip-page editor would, every minute, and flags moments
by what makes them work, whether or not chat reacted. Chat speed and laughter only catch the loud moments; a lot of what
goes viral is a two-second mood switch, a line that contradicts the last one, an awkward pause, a relatable dynamic,
which chat answers late or not at all.

scan(lines, frames, streamer, ...) -> [{a, b, peak, kind, why, framing, score}], stream seconds.
usage (offline, on a recorded session): semantic.py RUN_DIR [--step 60 --window 150]
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RULES = """You are the editor of a TikTok clip page for this streamer, watching the live stream as it happens. You get the last
couple of minutes of the transcript (stream seconds, who said what as best the transcript knows) and a few frames.
Find the moments a top clip page would post, judged by what is funny or gripping, not by how loud it was:
- mood_switch: someone flips from laughing to scolding (or the reverse) in seconds
- contradiction: a line the next moment proves wrong, or someone doing the thing they just said they'd never do
- unexpected: an answer or reaction nobody expected; deadpan; a guest's line that lands
- roast / comeback: a clean burn, a quick comeback
- awkward: a silence, a rejection, a cringe the people in it feel
- relatable: a dynamic everyone recognizes (couples, friends, parents, being broke, a bad date)
- wholesome / emotional: a real moment of care, pride, or someone tearing up
- chaos / shock: something happens on camera that people will replay
For each moment give: a, b (stream seconds: start a beat before the setup, end just after the payoff; 8-40 s), peak (the
second the payoff lands), kind (from the list), why (one line: what makes it work), framing (the caption a top clip page
would put on it, lowercase, the way those pages write: "the moment n3on realized how quick latinas' moods change 😭",
"POV: you told her twice already", "he really thought that would work 💀"; with known people in it, name them: who did
what to whom, in front of whom, "Lacy Sends Marlon A Sus Message In Front Of Darla 😂💀". Frame the pattern; never invent facts. Light,
relatable cultural humor is fine; no slurs, hate, dehumanizing stereotypes, or cruelty about bodies or faith), score
(1-10: how likely a stranger stops scrolling).
Only moments whose payoff is inside the window. Skip ordinary talk, reading chat, and anything sexual, with minors in a
bad light, or mocking a member of the public. If nothing qualifies, return an empty list.
Answer with JSON only: {"moments": [{"a": s, "b": s, "peak": s, "kind": "...", "why": "...", "framing": "...", "score": n}]}"""


def scan(lines, frames=(), streamer='', cast='', title='', editor='sonnet', timeout=60):
    import judge as J
    txt = '\n'.join(f"[{l['start']:.1f}] {l['text']}" for l in lines)
    prompt = (RULES + f"\n\nSTREAMER: {streamer}" + (f"\nREGULAR CAST: {cast}" if cast else '') + (f"\nSTREAM TITLE: {title}" if title else '')
              + f"\nTRANSCRIPT:\n{txt}")
    try:
        import lore
        prompt += lore.prompt_block(streamer)
    except Exception:  # noqa: BLE001
        pass
    raw = J._ask(prompt, frames, timeout, J.CLAUDE_MODELS.get(editor, 'sonnet') if hasattr(J, 'CLAUDE_MODELS') else 'sonnet')
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0)).get('moments') or []
    except ValueError:
        return []
    lo = min((l['start'] for l in lines), default=0); hi = max((l['end'] for l in lines), default=0)
    good = []
    for x in out:
        try:
            a, b, pk = float(x['a']), float(x['b']), float(x.get('peak', x['b'] - 2))
        except (KeyError, TypeError, ValueError):
            continue
        if not (lo - 5 <= a < b <= hi + 2) or not (6 <= b - a <= 50):
            continue
        good.append({'a': round(a, 1), 'b': round(b, 1), 'peak': round(min(max(pk, a), b), 1), 'kind': str(x.get('kind', ''))[:20],
                     'why': str(x.get('why', ''))[:160], 'framing': str(x.get('framing', ''))[:120], 'score': float(x.get('score') or 0)})
    return good


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--step', type=float, default=60); ap.add_argument('--window', type=float, default=150)
    ap.add_argument('--min', type=float, default=6)
    a = ap.parse_args()
    lines = [json.loads(l) for l in open(os.path.join(a.run, 'transcript.jsonl'))]
    st = json.load(open(os.path.join(a.run, 'state.json')))
    name = (st.get('meta') or {}).get('name') or ''
    end = max(l['end'] for l in lines)
    found = []
    t = a.window
    while t <= end + a.step:
        win = [l for l in lines if t - a.window <= l['start'] < t]
        if win:
            for m in scan(win, (), name, title=(st.get('meta') or {}).get('title', '')):
                if m['score'] >= a.min and not any(abs(m['peak'] - f['peak']) < 12 for f in found):
                    found.append(m)
        t += a.step
    chat = [(c['peak'], c.get('stage'), c.get('hook')) for c in st.get('clips', [])]
    print(f'{len(found)} semantic moments (score >= {a.min}); chat-triggered moments: {len(chat)}')
    for m in sorted(found, key=lambda m: m['peak']):
        near = [c for c in chat if abs(c[0] - m['peak']) < 20]
        print(f"{m['peak']:7.1f}  {m['score']:.0f}  {m['kind']:13s} {m['framing'][:70]:70s} | {'ALSO chat: ' + (near[0][2] or '')[:40] if near else 'chat missed it'}")
    json.dump(found, open(os.path.join(a.run, 'semantic.json'), 'w'), indent=1, ensure_ascii=False)
