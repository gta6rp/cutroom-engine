#!/usr/bin/env python3
"""Write REPORT.md for a live run: what the room watched, every moment with the editor's call, and how fast it cut.

usage: report.py <run_dir> [--from-service http://127.0.0.1:8820]
"""
import json, os, statistics, sys, urllib.request


def hms(t):
    t = max(0, int(t or 0))
    return f'{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}'


def mmss(t):
    t = max(0, round(t or 0))
    return f'{t // 60}:{t % 60:02d}'


def main(run, service=None):
    s = json.load(urllib.request.urlopen(service + '/state')) if service else json.load(open(os.path.join(run, 'state.json')))
    m = s['meta']
    clips = s['clips']
    ready = [c for c in clips if c['stage'] in ('ready', 'posting', 'posted')]
    dropped = [c for c in clips if c['stage'] == 'dropped']
    lat = [c['after_moment_s'] for c in ready if c.get('after_moment_s')]
    js = [c['judge_s'] for c in clips if c.get('judge_s')]
    rate = f" · ${m['rate']:g} per 100K views" if m.get('rate') else ''
    out = [f"# Live run {s['run']}", '',
           f"Watched **kick.com/{m['slug']}** ({m.get('title') or ''}) for **{hms(s['T'])}** of stream.",
           f"Program: {m.get('program')}{rate}.", '',
           '| | |', '|---|---|',
           f"| Words transcribed | {s.get('words', 0):,} ({s.get('model')}, {s.get('tx_rtf')}× real time on CPU) |",
           f"| Chat messages read | {s.get('chat_total', 0):,} |",
           f"| Chat lag | {s.get('chat_lag')} s (stream delay {s.get('edge_lag', '?')} s + player buffer + typing) |",
           f"| Moments flagged | {len(s['moments'])} |",
           f"| Kept by the editor | {len(ready)} |",
           f"| Dropped by the editor | {len(dropped)} |",
           f"| Moment → finished clip | {'median ' + mmss(statistics.median(lat)) + ', fastest ' + mmss(min(lat)) if lat else '—'} |",
           f"| Editor time | {'median ' + str(round(statistics.median(js), 1)) + ' s' if js else '—'} |", '',
           '## Moments', '']
    mom = {x['id']: x for x in s['moments']}
    for c in sorted(clips, key=lambda c: c['peak']):
        x = mom.get(c['moment'], {})
        sig = c.get('signals') or {}
        bits = [f"chat ×{sig.get('chat_x')}", f"{sig.get('laughs', 0)} laughs"] + ([f"{sig['clip_it']}× clip it"] if sig.get('clip_it') else []) + (['operator'] if sig.get('forced') else [])
        out.append(f"### {c['id']} · {hms(c['peak'])} · {c['stage']}")
        out.append(f"- signals: {', '.join(bits)}, score {sig.get('score')}")
        if x.get('chat_top'):
            out.append(f"- chat said: {', '.join(f'“{t}” ×{n}' for t, n in x['chat_top'][:4])}")
        out.append(f"- editor ({c.get('judge')}, {c.get('judge_s', '?')} s, {c.get('frames', 0)} frames): {c.get('why') or ''}")
        if c['stage'] in ('ready', 'posting', 'posted'):
            out.append(f"- **hook:** {c.get('hook')}  ·  caption: {c.get('caption')}")
            out.append(f"- {c.get('duration', 0):.0f} s {c.get('layout')} cut, rendered in {c.get('render_s')} s, ready {mmss(c.get('after_moment_s'))} after it happened → `{c.get('file')}`")
        out.append('')
    open(os.path.join(run, 'REPORT.md'), 'w').write('\n'.join(out))
    print('\n'.join(out[:16]))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[a.index('--from-service') + 1] if '--from-service' in a else None)
