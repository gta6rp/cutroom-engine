#!/usr/bin/env python3
"""The daily minimum: every room aims to put out at least DAILY_MIN cuts a day (default 3; CUTROOM_DAILY_MIN).

A cut costs little, and a room that posts nothing earns nothing, so when a room is behind it lowers its bar: it keeps
moments the editor scored RESCUE_MIN or better (default 4/10) instead of only its usual pick, and it goes back for the best
moment it passed on. Moments dropped for safety reasons are never rescued, whatever their score.

Counts are per streamer per local day in quota.json, shared by live rooms (live.py) and the offline agent (fleet.py).
usage: quota.py [slug]      prints today's counts
"""
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, 'quota.json')
DAILY_MIN = int(os.environ.get('CUTROOM_DAILY_MIN', '3'))
RESCUE_MIN = float(os.environ.get('CUTROOM_RESCUE_MIN', '6'))
# reasons a drop is final: the editor's hard-no categories and the lore library's flags
UNSAFE = re.compile(r'brand|safety|sexual|slur|gambl|casino|minor|allegation|hard no|private person|humiliat|harass|racis|culture.war|violence|weapon',
                    re.I)


def today():
    return time.strftime('%Y-%m-%d')


def _load():
    try:
        return json.load(open(F))
    except (OSError, ValueError):
        return {}


def count(slug):
    return int(_load().get(slug, {}).get(today(), 0))


def add(slug, n=1):
    d = _load()
    s = d.setdefault(slug, {})
    s[today()] = int(s.get(today(), 0)) + n
    for k in sorted(s)[:-14]:                             # two weeks of history is plenty
        s.pop(k, None)
    tmp = F + '.tmp'
    json.dump(d, open(tmp, 'w'), indent=1)
    os.replace(tmp, F)


def behind(slug, progress=1.0):
    """Behind pace: fewer cuts today than the daily minimum scaled by how far along the watching is (0..1)."""
    return count(slug) < DAILY_MIN * max(0.0, min(1.0, progress))


def rescuable(score, why):
    """A dropped moment the room may still cut when it's behind: good enough, and not dropped for safety."""
    return score is not None and score >= RESCUE_MIN and not UNSAFE.search(why or '')


def score_of(why):
    m = re.match(r'\s*(\d+(?:\.\d+)?)/10', why or '')
    return float(m.group(1)) if m else None


if __name__ == '__main__':
    d = _load()
    for slug, days in sorted(d.items()):
        if len(sys.argv) < 2 or sys.argv[1] == slug:
            print(slug, days.get(today(), 0), f'of {DAILY_MIN} today')
