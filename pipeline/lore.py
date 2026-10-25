#!/usr/bin/env python3
"""The lore library (research/lore/streamers.json) for the editors: what a streamer's world is, and what not to touch.

flags(who)   storylines marked as allegations / minors / private people / unsafe, as short "avoid" lines the editor
             reads with every moment of that streamer (a moment about one of them is a hard no)
brief(who)   regular cast and running jokes, so hooks can name people and land callbacks

usage: lore.py <slug-or-name>      prints both
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), 'config', 'lore.json')
_cache = None


def _all():
    global _cache
    if _cache is None:
        _cache = json.load(open(SRC)) if os.path.exists(SRC) else []
    return _cache


def find(who):
    k = re.sub(r'[^a-z0-9]', '', (who or '').lower())
    if not k:
        return None
    for s in _all():
        names = [s.get('id') or '', s.get('name') or ''] + list((s.get('handles') or {}).values())
        if any(k == re.sub(r'[^a-z0-9]', '', str(n).lower().split('(')[0]) or k == re.sub(r'[^a-z0-9]', '', str(n).lower()) for n in names if n):
            return s
    return None


def flags(who):
    s = find(who)
    if not s:
        return []
    out = []
    for a in s.get('arcs') or []:
        sens = (a.get('sensitivity') or 'none').lower()
        if sens != 'none' or not a.get('edit_angles'):
            who_ = ', '.join(a.get('participants') or [])[:80]
            out.append(f"{a.get('title')}{f' ({who_})' if who_ else ''}: {sens if sens != 'none' else 'not for edits'}")
    if s.get('brand_safety'):
        out.append(f"brand safety: {str(s['brand_safety'])[:160]}")
    return out


def brief(who):
    s = find(who)
    if not s:
        return {}
    cast = s.get('cast') or []
    cast = [c if isinstance(c, str) else c.get('name', '') for c in cast][:12]
    jokes = s.get('running_jokes') or []
    jokes = [j if isinstance(j, str) else j.get('joke') or j.get('text') or '' for j in jokes][:8]
    return {'who': (s.get('who') or '')[:300], 'cast': cast, 'running_jokes': jokes}


CAST_DIR = os.path.join(os.path.dirname(HERE), 'config', 'cast')


def cast_file(who):
    """research/lore/cast/<id>.json: who's who on this streamer's streams (public collaborators, how to tell them apart),
    so a friend's line is never captioned as the streamer's (a Clavicular collab, 2026-10-17)."""
    s = find(who)
    ids = [x for x in {(s or {}).get('id'), re.sub(r'[^a-z0-9_]', '', (who or '').lower())} if x]
    for i in ids:
        f = os.path.join(CAST_DIR, f'{i}.json')
        if os.path.exists(f):
            try:
                return json.load(open(f))
            except ValueError:
                return None
    return None


def prompt_block(who):
    """Text appended to any editor prompt about this streamer."""
    f, b = flags(who), brief(who)
    out = ''
    if f:
        out += '\nAVOID (hard no, score 1, if the moment is about any of these storylines of this streamer):\n- ' + '\n- '.join(f)
    c = cast_file(who)
    if c:
        st = c.get('streamer') or {}
        out += f"\nTHE STREAMER: {st.get('name') or who}. How to recognize them: {str(st.get('how_to_recognize') or '')[:300]}"
        people = [f"{p.get('name')} ({p.get('relationship') or 'collaborator'}; {str(p.get('how_to_recognize') or '')[:140]})"
                  for p in (c.get('cast') or [])[:14] if p.get('name')]
        if people:
            out += ('\nWHO ELSE SHOWS UP ON THEIR STREAMS (they talk too; never caption their lines as the streamer\'s; name '
                    'one in a hook only when they are clearly the one on screen):\n- ' + '\n- '.join(people))
    elif b.get('cast'):
        out += '\nTHEIR REGULAR CAST (name them in hooks only when they are clearly in the moment): ' + ', '.join(b['cast'])
    return out


if __name__ == '__main__':
    w = sys.argv[1]
    print(json.dumps({'flags': flags(w), 'brief': brief(w)}, indent=1, ensure_ascii=False))
