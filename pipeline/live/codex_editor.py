"""Codex as the room's editor, kept warm.

`codex exec` spends ~3.5 s starting up for every call. Instead this keeps one `codex app-server` process (JSON-RPC
over stdio) alive for the whole session, opens a fresh ephemeral thread per moment (so moments don't bias each
other), sends the prompt plus the frames as local images, and constrains the reply with a JSON schema.

    ed = CodexEditor()                 # once per room
    v = ed.ask(prompt, ['f0.jpg', ...], SCHEMA, effort='low')
"""
import json, os, subprocess, tempfile, threading, time


class CodexEditor:
    def __init__(self, model=None, timeout=60):
        self.model, self.timeout = model, timeout
        self.p = subprocess.Popen(['codex', 'app-server'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  text=True, bufsize=1, cwd=tempfile.gettempdir())
        self.n = 0
        self.lock = threading.Lock()
        self.req('initialize', {'clientInfo': {'name': 'cutroom', 'title': 'cutroom live room', 'version': '0.1'}})
        self.send({'method': 'initialized'})

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + '\n'); self.p.stdin.flush()

    def read(self, deadline):
        while time.time() < deadline:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('codex app-server exited')
            try:
                return json.loads(line)
            except ValueError:
                continue
        raise TimeoutError('codex app-server timed out')

    def req(self, method, params, deadline=None, on_note=None):
        self.n += 1
        rid = self.n
        self.send({'id': rid, 'method': method, 'params': params})
        deadline = deadline or time.time() + self.timeout
        while True:
            m = self.read(deadline)
            if m.get('id') == rid and ('result' in m or 'error' in m):
                if 'error' in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m['result']
            if 'method' in m and 'id' in m:      # a server request (approval etc.): we never approve anything
                self.send({'id': m['id'], 'result': {'decision': 'decline'}})
            elif on_note and 'method' in m:
                on_note(m)

    def ask(self, prompt, images=(), schema=None, effort='low'):
        """One editor call on a fresh ephemeral read-only thread. Returns the final agent message text."""
        with self.lock:
            deadline = time.time() + self.timeout
            th = self.req('thread/start', {'ephemeral': True, 'sandbox': 'read-only', 'approvalPolicy': 'never', 'cwd': tempfile.gettempdir(),
                                           **({'model': self.model} if self.model else {})}, deadline)
            tid = (th.get('thread') or {}).get('id') or th.get('threadId')
            inp = [{'type': 'text', 'text': prompt}] + [{'type': 'localImage', 'path': os.path.abspath(p)} for p in images]
            params = {'threadId': tid, 'input': inp, 'effort': effort}
            if schema:
                params['outputSchema'] = schema
            out = {'text': None, 'done': False}

            def note(m):
                meth, prm = m.get('method'), m.get('params') or {}
                if meth == 'item/completed' and (prm.get('item') or {}).get('type') == 'agentMessage':
                    out['text'] = prm['item'].get('text')
                elif meth == 'turn/completed':
                    out['done'] = True
                elif meth == 'error':
                    out['error'] = prm

            self.req('turn/start', params, deadline, note)
            while not out['done']:
                m = self.read(deadline)
                if 'method' in m and 'id' in m:
                    self.send({'id': m['id'], 'result': {'decision': 'decline'}})
                elif 'method' in m:
                    note(m)
            if out.get('text') is None:
                raise RuntimeError(f"codex gave no answer: {out.get('error')}")
            return out['text']

    def close(self):
        try:
            self.p.terminate()
        except Exception:  # noqa: BLE001
            pass


_shared = None
_shared_lock = threading.Lock()


def shared():
    """One warm Codex per process (survives importlib.reload of judge.py)."""
    global _shared
    with _shared_lock:
        if _shared is None or _shared.p.poll() is not None:
            _shared = CodexEditor()
        return _shared
