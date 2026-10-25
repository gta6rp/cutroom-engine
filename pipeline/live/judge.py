"""The editor pass for a live moment: keep or drop it, and write the hook and the post caption.

Uses Claude (headless `claude -p`, no tools) with only what the room actually heard: the transcript lines and what chat
repeated. Falls back to a rule-based editor if Claude is off, slow or returns something unusable. Nothing is invented:
the hook must come from the moment itself.
"""
import base64, shutil, time, difflib, json, os, re, subprocess, tempfile

RULES = """You are the editor of cutroom, a clipping channel that cuts live-stream moments into TikToks for the streamer's official clipping program.
You get a ~30 s moment that the room flagged from a live stream: the transcript (seconds from clip start), what chat spammed right after,
and up to 4 frames in order around the peak (about 8 s before to 5 s after). IRL moments are often visual: use the frames.
Decide if it is worth posting, and write the on-screen hook and the post caption.

Score it 1-10 for how well it would do as a TikTok clip of this streamer:
  8-10  a clear beat a stranger gets at once: setup + payoff, a big reaction, a confrontation, a surprise, a line people will quote
  6-7   a strong chat reaction plus a quotable or funny line, even if some context is missing: the hook can supply what a stranger needs
  3-5   mild: small talk with a chuckle, a reaction to something we can't see or hear
  1-2   nothing: music, crowd noise, dead air, menus, travel
Layout: when the frames show the streamer's facecam in a box over a game, a browser or a video, the clip puts that cam full width on top
and the game or page below. Judge those moments on the streamer's face, words and chat: a dark, static or menu game screen is not a
reason to score low on its own. Talk streams: a sharp, quotable take or a story with a punchline counts as a beat; rambling doesn't.
Hard no (score 1, whatever chat did): the joke depends on humiliating a member of the public who isn't part of the show; a slur
that IS the moment (the joke, or said at someone; one said in passing is muted in the cut and masked in captions, so judge
the moment without it); sexual content
or nudity; minors in a bad light; violence; gambling on screen (slots, a casino or poker table, cards and chips, a betting site,
a gambling stream or broadcast being watched), even when the joke is about something else. Chat laughing a lot (laughs, chat ×) is a real signal: when the words carry the moment, lean higher.

Hook rules: max 10 words (the top pages' median), plain English, no hashtags, no quote marks, no usernames. Make the viewer need the payoff.
Frame it the way the top clip pages do: name the pattern that makes it funny or relatable, not only what happens.
"the moment n3on realized how fast her mood changes 😭" (924K views) beats "Yera tells N3on where to put his phone".
Frames that work: "the moment X realized ...", "POV: ...", "when she ...", "he really thought ... 💀", "X has no idea ...".
Lowercase is fine for these. With two or more known people in it, the biggest pages (4-10M views) name who did what to whom,
and in front of whom, in Title Case: "Lacy Sends Marlon A Sus Message In Front Of Darla 😂💀", "Braeden Leaked Lacy's Cologne
Collection 😭💀": the names, one small action, and its social cost (a witness, a leak, getting caught). One or two key
words may go in CAPS for punch, the way 11-22M-view pages write them ("No way Camilla just EXPOSED N3on after she CAUGHT
him", "jynxzi reveals how many DAYS he has on r6 💀😭"); never a whole hook in caps. Light, relatable cultural humor is fine, the kind people share about their own group
("how quick latinas' moods change 😭"). The line: no slurs, no hate, no dehumanizing stereotypes, nothing cruel about
someone's body or faith, nothing that punches down at the person in the clip.
If the MOMENT has a "reading" (the room's own read of why it works, a suggested framing and the second the payoff lands,
reading_peak_t), use it: check it against the transcript and frames, keep what holds, fix what doesn't.
WHO IS SPEAKING: a stream is not one person. IRL and collab streams have friends, guests, crew and strangers on mic and on
camera, and the transcript doesn't say who talks. If the MOMENT has "streamer_photo": true, the FIRST image is the streamer's
own profile picture (not part of the moment): compare faces with it. Credit a line or an action to the streamer only when the
frames show it's them, or the transcript makes it unambiguous (they're addressed by name, it's clearly their own facecam).
Otherwise don't put it in the streamer's mouth: write the hook about what happened ("His friend…", "A guest…", or a regular
cast member's name only if they're clearly the one on screen). Return "speaker": who says or does the payoff: "streamer",
a cast member's name, "someone else" or "unclear".
Lines can carry "voice": "streamer" means that line matches the streamer's own voice (learned from their past streams),
"other" means a different voice, "?" means too short to tell. Weigh it with the frames: a payoff line in an "other" voice is
someone else's line, never the streamer's.
Who said it changes how you WRITE the hook, never the score: an unclear speaker is not a flaw (and don't mention it in "why").
Score the moment itself.
The hooks that win name who's in it (the streamer, or a guest from their regular cast who is clearly there) and one concrete
detail of the moment ("N3on asks Yera what deodorant she uses"), not a vague tease ("You won't believe this").
Emoji: end the hook with one emoji (two at most) when it matches the moment's tone, the way clip pages do:
😭 funny/painful, 💀 savage or cringe, 🥹 wholesome, ❤️‍🩹 touching, 😳 shocking, 🔥 hype, 😂 silly. Never mid-sentence. Skip it when nothing fits.
Only use facts that are in the transcript, chat or clearly visible in the frames. Never invent names, numbers or events. Don't spoil the punchline.
Promise only what the viewer will see or hear: if the payoff happens off screen or too dark to see (a mob in a black cave),
hook the reaction instead ("Jynxzi's face when his hardcore run ends 💀"), not the unseen event.
Caption: one short line in the stream's own words or a reaction, then the streamer's tag, e.g. #n3on #kick #irl. Max 150 characters.
Keep captions clean: no calls to action ("follow for more", "link in bio"), no hype or price claims ("going to moon", "easy money"),
no $TICKER cashtags (write the coin's name plainly).
Trim: optional [start, end] in seconds from clip start if the beat is tighter than the window (keep 12–45 s, start before the setup).
Put the payoff late: the peak should land at about 75-85% of the trimmed clip and the clip should end about 1 s after it.
Payoff: the second (same clock as the transcript times) where the payoff lands: the line, the hit, the reveal, the reaction.
The cut ends a few seconds after it and starts at most ~25 s before it, so name the real payoff, not the setup.
Context: the MOMENT may carry "before": the transcript of the minute BEFORE the window (t = seconds before the window starts).
Most clips work short. But when the setup is itself interesting and a stranger needs it to feel the payoff (a story building,
a call-out and its reason, someone explaining their situation, the question that sets up the answer), return "context": how
many seconds before the window the clip should start (10-60, at a sentence start). The clip then runs from there through the
payoff, up to 75 s. Never for a dull walk-up, small talk or dead air: then null. Context never changes the score.

Evidence: also return "evidence", the exact words from the transcript or chat that the hook rests on, copied verbatim (max 12 words),
or "frame: <what is plainly visible>" if the hook describes something seen. The hook may paraphrase the evidence but must not add claims it
doesn't make: no ages, prices, records, firsts, superlatives or motives that aren't in it.

Visible: describe what is plainly visible in the frames in at most 15 plain words (people, place, objects, actions). The hook may use these words.

Big: 1-3 single words from the transcript, copied exactly, that carry the moment (the reveal, the punch word, the reaction:
"KNIFE", "DRENCHED", "INSANE"). Each one is shown huge on screen as it's said, so pick words a stranger gets the story from,
in the order they're said. Never a swear, a slur, a filler or a name of someone who isn't clearly there. [] if none fits.

Reply with JSON only: {"score": 1-10, "keep": true if score >= 7, "why": "<10 words>", "hook": "...", "caption": "...", "evidence": "...", "visible": "...", "trim": [s, e] or null, "payoff": s or null, "context": s or null, "big": ["word", ...], "speaker": "streamer|<name>|someone else|unclear"}
Always write the hook, caption and evidence, even for a low score."""


KEEP_MIN = float(os.environ.get('CUTROOM_KEEP_MIN', '7'))   # 6s were half of what we kept; the user: selection "isn't that amazing"
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['score', 'keep', 'why', 'hook', 'caption', 'evidence', 'visible', 'trim', 'payoff', 'context', 'big', 'speaker'],
          'properties': {'score': {'type': 'number'}, 'keep': {'type': 'boolean'}, 'why': {'type': 'string'}, 'hook': {'type': 'string'}, 'caption': {'type': 'string'},
                         'evidence': {'type': 'string'}, 'visible': {'type': 'string'}, 'trim': {'type': ['array', 'null'], 'items': {'type': 'number'}},
                         'payoff': {'type': ['number', 'null']}, 'context': {'type': ['number', 'null']}, 'big': {'type': 'array', 'items': {'type': 'string'}}, 'speaker': {'type': 'string'}}}
CLAUDE_MODELS = {'sonnet': 'sonnet', 'haiku': 'haiku', 'claude': 'sonnet'}


def _ask_codex(prompt, frames=()):
    """Codex (kept warm in codex_editor), frames as local images, reply constrained to SCHEMA."""
    import codex_editor
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, f in enumerate(frames):
            paths.append(os.path.join(d, f'f{i}.jpg'))
            open(paths[-1], 'wb').write(f)
        return codex_editor.shared().ask(prompt, paths, SCHEMA, effort='low')


# the Claude CLI by full path: services run with systemd's short PATH, and a CLI update on 2026-10-15 left `claude` only in
# ~/.local/bin, so every editor call failed ("No such file or directory: 'claude'") and the live rooms cut nothing for 31 h
CLAUDE_BIN = os.environ.get('CLAUDE_BIN') or shutil.which('claude') or os.path.expanduser('~/.local/bin/claude')


# when the Claude plan hits its session limit ("You've hit your session limit · resets ..."), every editor, safety and
# critic call fails and the rooms cut nothing (2026-10-17, the evening before launch). The first call that sees it leaves a
# marker; for 10 minutes every process sends its calls to Codex (the ChatGPT login: a separate quota) instead
LIMIT_F = os.path.join(tempfile.gettempdir(), 'cutroom-claude-limited')
LIMIT_RE = re.compile(r"hit your (session|usage|weekly) limit|usage limit|rate.?limit|limit reached|out of (extra )?usage", re.I)


def claude_limited():
    return os.path.exists(LIMIT_F) and time.time() - os.path.getmtime(LIMIT_F) < 600


def _ask_codex_text(prompt, frames=()):
    """The same call through Codex, free text (the callers pull the JSON out themselves)."""
    import codex_editor
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, f in enumerate(frames):
            paths.append(os.path.join(d, f'f{i}.jpg'))
            open(paths[-1], 'wb').write(f)
        return codex_editor.shared().ask(prompt + '\n\nReply with the JSON only.', paths, None, effort='low')


def _ask(prompt, frames=(), timeout=40, model='sonnet'):
    """One headless Claude turn, no tools. Frames (JPEG bytes) go in as images ahead of the text. Falls over to Codex while
    the Claude plan is at its usage limit."""
    if claude_limited():
        return _ask_codex_text(prompt, frames)
    content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': base64.b64encode(f).decode()}} for f in frames]
    content.append({'type': 'text', 'text': prompt})
    msg = {'type': 'user', 'message': {'role': 'user', 'content': content}}
    r = subprocess.run([CLAUDE_BIN, '-p', '--model', model, '--tools', '', '--no-session-persistence', '--strict-mcp-config',
                        '--input-format', 'stream-json', '--output-format', 'stream-json', '--verbose'],
                       input=json.dumps(msg) + '\n', capture_output=True, text=True, timeout=timeout, cwd='/tmp',
                       env={**os.environ, 'MAX_THINKING_TOKENS': '0'})   # no extended thinking: 3-4 s instead of 15-40 s, same calls
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get('type') == 'result':
            res = d.get('result') or ''
            if LIMIT_RE.search(res) and '{' not in res:
                open(LIMIT_F, 'w').write(res[:200])
                return _ask_codex_text(prompt, frames)
            return res
    raise ValueError(f'no result from editor: {r.stderr[:200]}')


def _parse(raw, editor, max_words=11):
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        raise ValueError(f'no JSON from editor: {raw[:200]}')
    v = json.loads(m.group(0))
    if isinstance(v.get('score'), (int, float)):
        v['keep'] = v['score'] >= KEEP_MIN          # the threshold is ours to tune; the editor only scores
    hook = re.sub(r'["“”#]', '', str(v.get('hook') or '')).strip()
    words = [w for w in hook.split() if re.search(r'[A-Za-z0-9]', w)]
    if not isinstance(v.get('keep'), bool) or (v['keep'] and not (1 <= len(words) <= max_words)):   # top pages' median hook is 10 words (HOOK_STYLE may allow more)
        raise ValueError(f'unusable editor reply: {v}')
    trim = v.get('trim')
    if trim and not (isinstance(trim, list) and len(trim) == 2 and all(isinstance(x, (int, float)) for x in trim) and 10 <= trim[1] - trim[0] <= 50):
        trim = None
    sc = v.get('score')
    po = v.get('payoff')
    po = float(po) if isinstance(po, (int, float)) and po >= 0 else None
    out = {'payoff': po, 'keep': v['keep'], 'score': sc, 'why': (f'{sc:g}/10 · ' if isinstance(sc, (int, float)) else '') + str(v.get('why', ''))[:120], 'hook': hook, 'caption': str(v.get('caption') or '')[:150], 'trim': trim,
           'evidence': str(v.get('evidence') or '')[:160], 'visible': str(v.get('visible') or '')[:160], 'by': editor if editor == 'codex' else f'claude-{CLAUDE_MODELS.get(editor, "sonnet")}'}
    out['hook'], out['caption'] = mask(out['hook']), mask(out['caption'])
    big = [re.sub(r"[^\w']", '', x) for x in (v.get('big') or []) if isinstance(x, str) and len(x.split()) == 1]
    out['big'] = [x for x in big if x and mask(x) == x][:3]     # one word each; a word mask() stars never goes huge
    out['speaker'] = str(v.get('speaker') or 'unclear')[:40]
    cx = v.get('context')                              # seconds of setup to add before the window (the long, full-context cut)
    out['context'] = float(cx) if isinstance(cx, (int, float)) and 10 <= cx <= 60 else None
    return out


def _check(out, payload):
    """(grounded evidence?, hook words the moment doesn't support)"""
    g = grounded(out['evidence'], payload)
    return g, (unsupported(out['hook'], out['evidence'] + ' ' + out['visible'], payload) if g else [])


# per-room hook experiments (clips/CLIPPING-GOAL.md, one measured editing change at a time). #1, 2026-10-19, Clavicular:
# premise-first hooks, the way his biggest clip pages write them this week (@vexrr4 292K in 3 h, @clavclips4 854K)
HOOK_STYLE = {'clavicular': {'max_words': 16, 'rules': '''
THIS ROOM'S HOOK STYLE (it replaces the hook frames above: no "the moment...", no "POV:", no "when he..."):
Premise first. Say what happened: who did what to whom, or how someone reacted. Name Clavicular and the other person
whenever the transcript or chat says their name (Kevin, Master, Marlon...); otherwise "his friend", "a camper", "chat".
Put one or two key words in CAPS (NERVOUS, PANICKED, VOTED, CHARACTER), never the whole hook.
WHO SAID IT: his co-host Hexumlite ("Hex") sounds like Clavicular and is on camera constantly, and a line of Hex's once went
out as Clavicular's. Credit Clavicular only when the frames show he's the one talking or the transcript makes it unambiguous;
otherwise name Hex if that's clear, or frame it as the group ("the guys", "Camp Looksmax", "a camper").
16 words or fewer, one emoji at the end. Say "Camp Looksmax" only when it fits naturally.
Never label anyone with a diagnosis, disability, condition or race ("ND", "autistic"...), even though other pages do. If the
person says it about themselves on stream ("I get really bad anxiety"), the hook may mention it plainly as context, never
as the joke (the user, 2026-10-19). No slurs. Examples: "Clavicular called Kevin out for putting on a CHARACTER and went full James Doakes 😭",
"Clavicular said he's had 10/10 experiences with EVERY gender, then PANICKED 😭",
"Chat VOTED Clavicular's date off after the most boring answer ever 💀".'''}}


def _claude(payload, frames=(), timeout=40, editor='sonnet'):
    try:
        import lore                                   # the streamer's flagged storylines and regular cast (research/lore)
        lb = lore.prompt_block(payload.get('slug') or payload.get('streamer'))
    except Exception:  # noqa: BLE001
        lb = ''
    hs = HOOK_STYLE.get(payload.get('slug') or '') or {}
    mw = hs.get('max_words', 11)
    prompt = RULES + lb + hs.get('rules', '') + '\n\nMOMENT:\n' + json.dumps(payload, ensure_ascii=False)
    ask = (lambda p: _ask_codex(p, frames)) if editor == 'codex' else (lambda p: _ask(p, frames, timeout, CLAUDE_MODELS.get(editor, 'sonnet')))
    out = _parse(ask(prompt), editor, mw)
    if editor != 'codex' and claude_limited():
        out['by'] = 'codex (Claude at its usage limit)'   # the call went to Codex (see _ask)
    g, extra = _check(out, payload)
    if out['keep'] and (not g or len(extra) > 1):
        # one more pass with the problem spelled out: a rewrite in the moment's own words beats a pasted quote
        problem = ("your evidence isn't in the transcript or chat word for word" if not g
                   else f"your hook says {', '.join(extra)}, which nobody says and the frames don't show")
        try:
            again = _parse(ask(prompt + f"\n\nYOUR FIRST ANSWER:\n{json.dumps({k: out[k] for k in ('hook', 'evidence', 'visible')}, ensure_ascii=False)}\n"
                               f"Problem: {problem}. Answer again with the same JSON. Keep the score. Write the hook from words that were said "
                               f"or things the frames show; describe, don't add facts."), editor, mw)
            g2, extra2 = _check(again, payload)
            if again['keep'] and g2 and len(extra2) <= 1:
                again['hook_first'] = out['hook']; again['by'] += '+retry'
                return again
        except Exception:  # noqa: BLE001
            pass
    if out['keep'] and (not g or len(extra) > 1):
        # the editor's hook still can't be traced to the moment: use the moment's own words rather than ship an invented detail
        out['hook_original'] = out['hook']
        # best fallback: the editor's own evidence line, already verified to be in the transcript; else the best sentence
        ev = tidy(out['evidence']).strip(' ,.')
        ev_ok = g and not ev.lower().startswith('frame:') and 3 <= len(ev.split()) <= 10 and not SLUR.search(ev) and english(ev)
        out['why'] += (' (hook replaced: evidence not found in transcript)' if not g else f" (hook replaced: says {', '.join(extra)}, which the moment doesn't)")
        sg = payload.get('signals') or {}
        safe = 'Chat lost it at this 😭' if (sg.get('laughs') or 0) >= 10 or (sg.get('chat_x') or 0) >= 2 else 'Wait for it 😳'
        out['hook'] = mask('“' + ev[:1].upper() + ev[1:] + '”') if ev_ok else (quote_hook(payload) or safe)
        out['caption'] = mask((out['hook'] if ev_ok else (quote_hook(payload, 14) or out['hook'])) + ' ' + ' '.join(t for t in out['caption'].split() if t.startswith('#')))
        out['by'] += '+quote'
    return out


_norm = lambda t: re.sub(r'[^a-z0-9 ]+', ' ', t.lower()).split()


def grounded(evidence, payload):
    """True when the evidence is frame-based, or ~75% of its words (by stem) appear in the transcript or chat.
    Word coverage, not strict order: ASR spellings drift ("OG" / "O.G.") while the words are still there."""
    ev = evidence.strip()
    if ev.lower().startswith('frame:'):
        return len(ev) > 8
    words = [w for w in _norm(ev) if len(w) > 1]
    if len(words) < 2:
        return False
    hay = {w[:5] for w in _norm(' '.join(ln['text'] for ln in payload['lines']) + ' ' + ' '.join(t for t, _ in payload['chat_top']))}
    return sum(1 for w in words if w[:5] in hay) >= 0.75 * len(words)


STOP = set('the a an and or but so to of in on at for with from by is are was were be been being it its this that these those he she they them '
           'his her their him you your we our i me my just about after before then than into out up down over off like what who why how when '
           'where all any some more most very really gets got get starts start says said say does did do has have had not no '
           'now still even already here there again ever every one two three wanna its it\'s he\'s she\'s that\'s '
           'can can\'t won\'t don\'t didn\'t isn\'t wasn\'t when while because if only also too '
           'chat stream streamer live clip clips viewers viewer camera cam'.split())   # meta words: true of every live moment
# framing: what the streamer is doing, not a claim about the world ("Konvy calls out how someone reacts to clips")
STOP |= set('call calls called calling react reacts reacted reacting watch watches watched watching roast roasts roasted roasting rant rants '
            'ranting explain explains explaining read reads reading ask asks asked asking tell tells telling talk talks talking reply replies '
            'respond responds responded answer answers argue argues arguing defend defends defending lose loses loses goes go going '
            'someone somebody anyone everyone people person guy guys girl girls man woman friend friends bro dude story take opinion moment '
            'reaction realizes realize realized notices notice tries try trying finds find pov really thought idea whole '
            'moment way already twice again'.split())


def supported(hook, evidence, payload, free=1):
    """The hook's content words must come from the moment (transcript, chat, evidence), allowing `free` paraphrase words.
    Catches a real quote used as evidence while the hook still adds something nobody said ("...going to the club")."""
    if evidence.strip().lower().startswith('frame:'):
        hay = set(_norm(evidence))
    else:
        hay = set()
    hay |= set(_norm(' '.join(ln['text'] for ln in payload['lines']) + ' ' + evidence + ' ' + ' '.join(t for t, _ in payload['chat_top'])))
    stems = {w[:5] for w in hay}
    return len(unsupported(hook, evidence, payload)) <= free


IRREGULAR = {'lost': 'lose', 'said': 'say', 'says': 'say', 'saw': 'see', 'seen': 'see', 'went': 'go', 'gone': 'go', 'got': 'get', 'made': 'make',
             'told': 'tell', 'found': 'find', 'took': 'take', 'taken': 'take', 'came': 'come', 'gave': 'give', 'given': 'give', 'thought': 'think',
             'bought': 'buy', 'left': 'leave', 'felt': 'feel', 'kept': 'keep', 'won': 'win', 'ran': 'run', 'ate': 'eat', 'eaten': 'eat',
             'began': 'begin', 'brought': 'bring', 'caught': 'catch', 'fell': 'fall', 'heard': 'hear', 'knew': 'know', 'paid': 'pay', 'sold': 'sell',
             'met': 'meet', 'wrote': 'write', 'broke': 'break', 'chose': 'choose', 'drove': 'drive', 'spent': 'spend', 'stole': 'steal', 'threw': 'throw'}
STYLE = set('suddenly literally actually basically totally completely instantly immediately finally lowkey straight casually randomly '
            'apparently somehow seriously honestly genuinely clearly definitely absolutely simply truly lowkey highkey whole entire'.split())


def stem(w):
    """Light stemming for matching hook words to the moment: irregular verbs, then common suffixes, then 5 letters."""
    w = IRREGULAR.get(w, w)
    for suf in ('ingly', 'edly', 'ing', 'ied', 'ies', 'ed', 'es', 's', 'ly', 'er'):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + ('y' if suf in ('ied', 'ies') else '')
            break
    if len(w) > 3 and w.endswith('e'):
        w = w[:-1]                                   # lose/los, make/mak: both sides land on the same stem
    return w[:5]


def unsupported(hook, evidence, payload):
    """The hook's content words that appear nowhere in the moment (transcript, chat, evidence), by 5-letter stem."""
    # the stream's title and the streamer's name are facts about the moment too ("Day 1 in the Philippines")
    hay = set(_norm(' '.join(ln['text'] for ln in payload['lines']) + ' ' + evidence + ' ' + ' '.join(t for t, _ in payload['chat_top'])
                    + ' ' + str(payload.get('title') or '') + ' ' + str(payload.get('streamer') or '')))
    stems = {stem(w) for w in hay} | {w[:5] for w in hay}
    content = [w for w in _norm(hook) if (len(w) > 2 or re.search(r'\d', w)) and w not in STOP and w not in STYLE]
    missing = [w for w in content if stem(w) not in stems and w[:5] not in stems and w not in hay]
    # only factual claims have to come from the moment: names, numbers, and words that assert an event. Interpretive
    # framing ("how fast her mood changes", "he really thought that would work") is the clip page's own voice.
    caps = {re.sub(r'[^a-z0-9]', '', w.lower()) for k, w in enumerate(hook.split()) if k and w[:1].isupper()}
    return [w for w in missing if w in caps or re.search(r'\d', w) or stem(w) in CLAIM_STEMS]


CLAIM = ('die died dead death kill killed arrest arrested police cops cop jail prison court sued lawsuit pregnant baby married marry wedding '
         'divorce divorced cheat cheated cheating breakup dating date dated girlfriend boyfriend wife husband engaged fight fought punch punched '
         'slap slapped stole stolen steal banned ban fired quit million thousand dollar dollars money paid pay bought buy sold sell won win lost lose '
         'record first ever never always hospital crash crashed accident kiss kissed drunk high drugs gun shot shooting robbed rob scam scammed '
         'leaked leak hacked hack sick cancer ex mom dad brother sister son daughter')
CLAIM_STEMS = {stem(w) for w in CLAIM.split()}


# on-screen text and captions: TikTok down-ranks profanity in text, and slurs never ship. Mask, keeping the first letter.
PROFANE = [r'f+u+c+k\w*', r'sh+i+t\w*', r'b+i+t+c+h\w*', r'd+i+c+k\w*', r'p+u+s+s+y\w*', r'\bpuss\b', r'c+u+n+t\w*', r'\bass(hole)?\b',
           r'n+[i1]+gg\w*', r'\bf+a+g+\w*', r'r+e+t+a+r+d\w*', r'\bwhore\w*', r'\bslut\w*']
SLUR = re.compile(r'n+[i1]+gg|\bf+a+g+|r+e+t+a+r+d|tranny|chink|\bspic\b|kike', re.I)


def mask(t):
    for pat in PROFANE:
        t = re.sub(pat, lambda m: m.group(0)[0] + '*' * (len(m.group(0)) - 2) + m.group(0)[-1] if len(m.group(0)) > 2 else m.group(0)[0] + '*', t, flags=re.I)
    return t


def profane(t):
    return any(re.search(p, t, re.I) for p in PROFANE)


FILLER = re.compile(r"^(?:(?:um+|uh+|uhm|erm|well|like|so|yeah|yes|okay|ok|oh|and|but|i mean|you know)[,.!?]*\s+)+", re.I)
INNER_FILLER = re.compile(r"(?:,\s*(?:like|yeah|i mean|you know)(?=,)|,?\s*\b(?:um+|uh+|uhm|erm)\b(?=[,.!?\s]|$))+,?", re.I)


def tidy(q):
    """A spoken line made quotable: no leading or inner fillers ("My response is, um, yeah, I mean, you ain't funny"
    -> "My response is, you ain't funny"), no stuttered repeats ("I I I")."""
    q = FILLER.sub('', q.strip())
    q = INNER_FILLER.sub(lambda m: ',' if m.group(0).startswith(',') else '', q)
    q = re.sub(r'\b(\w+)(?:[,\s]+\1\b)+', r'\1', q, flags=re.I)
    return re.sub(r'\s+,', ',', re.sub(r'\s{2,}', ' ', q)).strip(' ,')


def english(t):
    """Rooms clip for an English-speaking audience: a quote needs a fair share of everyday English words."""
    ws = _norm(t)
    return bool(ws) and sum(1 for w in ws if w in STOP or w in COMMON) >= 0.35 * len(ws)


COMMON = set('i you me my we us our like know think want need go going gonna got get see look lets let come good bad nice crazy bro dude man '
             'yeah yes no okay ok what why how who this that it is was are be do did done really right now here there love hate thing stuff '
             'people guy guys girl one time day night make made say said tell told feel well real money car food eat'.split())


def quote_hook(payload, n=9):
    """The moment's most quotable sentence as a direct quote: whole sentence, 3-9 words, fillers trimmed, exclaimed preferred."""
    text = ' '.join(ln['text'] for ln in payload['lines'])
    sents = [tidy(x) for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()]
    sents = [x for x in sents if 3 <= len(x.split()) <= n and not SLUR.search(x) and english(x)]
    if not sents:
        return ''
    best = max(sents, key=lambda x: (x.endswith(('!', '?')) * 3) + min(len(x.split()), 7) + bool(re.search(r'\b(no way|what|bro|why|never|always|crazy)\b', x, re.I)) * 2
               - profane(x) * 6)
    best = mask(best.strip(' ,.'))
    return ('“' + best[:1].upper() + best[1:] + '”') if best else ''


def _rules(p):
    s = p['signals']
    tag = '#' + re.sub(r'[^a-z0-9]', '', p['slug'].lower())
    # a line chat repeated is the best hook: it's what the audience already calls the moment
    rep = [t for t, c in p['chat_top'] if c >= 5 and 3 <= len(t.split()) <= 7 and '@' not in t and 'http' not in t]
    if rep:
        hook = rep[0].capitalize()
    else:
        best = max(p['lines'], key=lambda ln: len(re.findall(r'[!?]', ln['text'])) + len(ln['text']) / 80, default=None)
        hook = ' '.join(best['text'].split()[:7]).strip(' ,.') if best else 'Chat lost it at this'
    # without a reader, only keep what the room is sure about: a strong spike with words to caption
    keep = (s['score'] >= 8 or s['clip_it'] >= 3) and sum(len(ln['text'].split()) for ln in p['lines']) >= 25
    return {'keep': keep, 'why': 'rule-based editor', 'hook': hook, 'caption': f"{tag} #kick #irl", 'trim': None, 'by': 'rules'}


def judge(payload, mode='sonnet', frames=()):
    """mode: sonnet (default, live demo), haiku, codex, or off (rules only).
    If the editor's reply is unusable (a hook too long, no JSON...), it gets one more try with the rules restated; if it still
    fails, the moment is dropped: a rules-made hook (a raw transcript line) never ships when an editor was asked for."""
    if mode != 'off' and (payload['lines'] or frames):
        err = None
        for attempt in range(2):
            try:
                p = payload if attempt == 0 else {**payload, 'note': f'Your last reply was unusable ({str(err)[:80]}). Reply with the JSON only; hook at most 8 words.'}
                return _claude(p, frames, editor=mode)
            except Exception as e:  # noqa: BLE001
                err = e
        v = _rules(payload)
        v.update(keep=False, why=f'editor unavailable, not cut ({str(err)[:60]})', by='none')
        return v
    return _rules(payload)
