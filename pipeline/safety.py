#!/usr/bin/env python3
"""Brand safety on the picture, not just the words. A clip's title and transcript can be harmless while the screen shows
something else: xQc's "Trump's grilled cheese" cut was political military news posts about Trump on his screen, plus a
slur written in a tweet (flagged 2026-10-11 by the marketing/fx session, whose own check is the same idea). The editor
(Claude) looks at 8 frames of the FINISHED cut, on-screen text included, and its transcript before it goes anywhere.

check(video, words_text='') -> {'safe': bool, 'why': str}   (cached next to the video as <video>.safety.json)
It fails closed: if the check can't run, the cut is held, never posted unchecked.
The SOUND is checked too (audio_gate): the finished file is transcribed again from its own soundtrack and any slur still
audible is bleeped in place, then heard again. The cut's transcript only bleeps the slurs it caught; one it missed (or a
slur in a song's lyrics) played on. The fx session's show-season lab found such a line, dropped from the captions
while its audio kept playing (2026-10-11).
usage: safety.py CUT.mp4 [CUT.mp4 ...]
"""
import json, os, re, sys
os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'live'))

RULES = """You check a short vertical clip before a streamer-clipping page posts it. Look at every frame closely, including
on-screen text: tweets, news, browser pages, chat, captions, signs. Read the transcript too. The clip is UNSAFE if any of:
- a slur or hate symbol anywhere: written in a tweet, chat or caption, or said as the point of the clip
- gambling: slots, casino, betting sites, odds, crypto casinos
- sexual content or nudity; minors shown in a bad light
- graphic violence, weapons used on people, real tragedies or deaths
- real allegations about named people (crimes, abuse, cheating)
Political topics, politicians, news and opinions are fine (the user, 2026-10-21: political streamers like Asmongold
are clipped like anyone else); the rules above still apply to them. Otherwise it is safe. Answer with JSON only: {"safe": true, "why": "one line naming what you saw"}"""


SLUR_HEARD = re.compile(r'\bn+i+g+(a|er|ah|uh|az)s?\b|\bf+a+g+(s|got|gots)?\b|\br+e+t+a+r+d+(ed|s)?\b|\bk+i+k+e+s?\b|'
                        r'\bc+h+i+n+k+s?\b|\bs+p+i+c+s?\b|\bt+r+a+n+n+(y|ies)\b|\bw+e+t+b+a+c+k+s?\b', re.I)   # fx/audio_audit.py's list
FF = os.path.join(os.path.dirname(HERE), '.venv', 'bin', 'ffmpeg')
_ASR = {}


def heard(video):
    """[(start, end, word)]: what the finished file's own soundtrack says (faster-whisper small.en, CPU int8: no GPU job).
    The wav is decoded here: the venv's PyAV doesn't match faster-whisper's own loader."""
    import subprocess, tempfile, wave
    import numpy as np
    if 'm' not in _ASR:
        from faster_whisper import WhisperModel
        _ASR['m'] = WhisperModel('small.en', device='cpu', compute_type='int8', cpu_threads=4)
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, 'a.wav')
        subprocess.run([FF, '-nostdin', '-y', '-v', 'error', '-i', video, '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav], check=True, capture_output=True)
        with wave.open(wav) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    segs, _ = _ASR['m'].transcribe(pcm, word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    return [(round(w.start, 2), round(w.end, 2), w.word.strip()) for sg in segs for w in (sg.words or [])]


def audio_gate(video):
    """Bleep any slur still audible in the finished file (in place: the picture is copied, the sound re-encoded), then
    listen again. -> {'safe', 'why', 'bleeped': n}. Raises if it can't listen (the caller fails closed)."""
    import shutil, subprocess, tempfile
    slurs = lambda: [(s0, s1) for s0, s1, w in heard(video) if SLUR_HEARD.search(re.sub(r'[^\w]', '', w))]
    hits = slurs()
    if not hits:
        return {'safe': True, 'why': '', 'bleeped': 0}
    sys.path.insert(0, HERE)
    from edit import audio_chain
    spans = [(max(0.0, s0 - 0.05), max(s1, s0 + 0.25) + 0.08) for s0, s1 in hits]
    td = tempfile.mkdtemp(prefix='hush-'); out = os.path.join(td, 'o' + os.path.splitext(video)[1])
    subprocess.run([FF, '-nostdin', '-y', '-loglevel', 'error', '-i', video, '-filter_complex',
                    audio_chain(spans, 3600).replace('loudnorm=I=-14:TP=-1.5:LRA=11', 'anull'), '-map', '0:v', '-map', '[a]',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', out], check=True, capture_output=True)
    shutil.move(out, video); shutil.rmtree(td, ignore_errors=True)
    left = slurs()
    if left:
        return {'safe': False, 'why': f'a slur is still audible at {left[0][0]:.1f} s after bleeping', 'bleeped': len(hits)}
    return {'safe': True, 'why': f'bleeped {len(hits)} slur{"s" if len(hits) > 1 else ""} the transcript missed', 'bleeped': len(hits)}


# betting branding on the SET is easy to miss in a general look (Kick's IRL events and sponsored streams: "Stake" printed
# on every rope and corner pad of a boxing ring passed the general review every time, and one dedicated look missed it
# once; the fx session, 2026-10-12). So a look for nothing else, twice, the frames in two orders, either one counting
BRAND = """Look ONLY for gambling or betting branding in these frames: sportsbook or casino names and logos (Stake, Roobet,
Rollbit, 1win, BC.Game, Shuffle, Gamdom, Duelbits, DraftKings, FanDuel, bet365, and any other betting or casino brand),
slot machines, roulette, poker tables, odds boards, crypto casinos. Check ring ropes, corner pads, banners, backdrops,
shirts, hats, phone cases, laptop and phone screens, overlays and on-screen text. The streaming platform's own logo
(Kick, Twitch, YouTube) is NOT betting. Answer with JSON only: {"betting": true or false, "what": "the brand and where, or empty"}"""


def brand_look(J, imgs):
    raw = J._ask(BRAND, imgs, 90)
    m = re.search(r'\{.*\}', raw, re.S)
    d = json.loads(m.group(0)) if m else {}
    if not isinstance(d.get('betting'), bool):
        raise ValueError('no brand verdict')
    return d


def frames(video, n=8, w=480):
    import cv2
    c = cv2.VideoCapture(video); N = int(c.get(7) or 0); out = []
    for k in range(n):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(N * (k + 0.5) / n)); ok, f = c.read()
        if ok:
            h0, w0 = f.shape[:2]
            out.append(cv2.imencode('.jpg', cv2.resize(f, (w, int(w * h0 / w0))), [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes())
    return out


def source_frames(video, n=6, w=480):
    """Frames of the same moment from the source, without our captions or hook: text our captions cover is still on
    screen for some of the cut (a "FRIENDS NOT F**DS" shirt passed once when the captions hid the word in all 8 frames,
    and was held on the next cut of the same stream). Needs the cut's spec.json beside it (edit.py's clip folders)."""
    import cv2
    d = os.path.dirname(os.path.abspath(video))
    try:
        S = json.load(open(os.path.join(d, 'spec.json')))
        src = S.get('video') or (S.get('videos') or [{}])[0].get('file')
        off = 0.0 if S.get('video') else float((S.get('videos') or [{}])[0].get('offset') or 0)
        src = src if os.path.isabs(src) else os.path.join(d, src)
        spans = [(a - off, b - off) for a, b in S.get('segments') or []]
    except Exception:  # noqa: BLE001
        return []
    tot = sum(b - a for a, b in spans)
    if not os.path.exists(src) or tot <= 0:
        return []
    c = cv2.VideoCapture(src); out = []
    for k in range(n):
        t = tot * (k + 0.5) / n
        for a, b in spans:
            if t <= b - a:
                c.set(cv2.CAP_PROP_POS_MSEC, (a + t) * 1000); ok, f = c.read()
                if ok:
                    h0, w0 = f.shape[:2]
                    out.append(cv2.imencode('.jpg', cv2.resize(f, (w, int(w * h0 / w0))), [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes())
                break
            t -= b - a
    return out


def check(video, words_text=''):
    cache = video + '.safety.json'
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(video):
        return json.load(open(cache))
    import judge as J
    v = {'safe': False, 'why': 'safety check unavailable: held until it can run'}
    try:
        au = audio_gate(video)                          # the sound first: it may re-write the file's audio
        if not au['safe']:
            json.dump(au, open(cache, 'w'))
            return au
        imgs = frames(video)
        raw_src = source_frames(video) if imgs else []
        if imgs:
            note = (f"\n\nThe first {len(imgs)} images are the finished cut; the last {len(raw_src)} are the same moment from the source, without "
                    "our captions: text our captions cover (a shirt, a sign, a screen) is still in the clip, so judge it too.") if raw_src else ''
            from concurrent.futures import ThreadPoolExecutor
            allf = imgs + raw_src
            with ThreadPoolExecutor(3) as ex:          # the general look and the two betting-brand looks at once
                gen = ex.submit(J._ask, RULES + note + f"\n\nTRANSCRIPT: {(words_text or '(none)')[:1500]}", allf, 120)
                looks = [ex.submit(brand_look, J, allf), ex.submit(brand_look, J, allf[::-1])]
                raw = gen.result(); brands = [f.result() for f in looks]
            m = re.search(r'\{.*\}', raw, re.S)
            d = json.loads(m.group(0)) if m else {}
            bet = next((b for b in brands if b['betting']), None)
            if bet:
                v = {'safe': False, 'why': f"betting branding on screen: {str(bet.get('what') or '')[:160]}"}
            elif isinstance(d.get('safe'), bool):
                v = {'safe': d['safe'], 'why': str(d.get('why') or '')[:200], **({'bleeped': au['bleeped'], 'sound': au['why']} if au['bleeped'] else {})}
    except Exception as e:  # noqa: BLE001
        v = {'safe': False, 'why': f'safety check failed ({str(e)[:80]}): held'}
    if not v['why'].startswith('safety check'):       # only real verdicts are cached; a failed check is retried
        json.dump(v, open(cache, 'w'))
    return v


if __name__ == '__main__':
    for f in sys.argv[1:]:
        print(f, check(f))
