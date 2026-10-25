#!/usr/bin/env python3
"""cutroom ASR: whisper-large-v3-turbo on the GPU, shared by every live room on this machine.

Runs inside nvcr.io/nvidia/pytorch (see run.sh): the host venv's torch can't JIT its Triton kernels (no Python
headers on the host), the container has them. One model, one request at a time, GPU memory capped.

POST /transcribe   body: s16le mono 16 kHz PCM   ->   {"text", "language", "words": [{"w", "s", "e"}], "seconds"}
GET  /health       ->   {"ok": true, "model": ...}

English by default (ASR_LANGUAGE=english): rooms clip English-speaking streams. Don't call model.detect_language()
before the pipeline: it leaves the decoder without punctuation or casing for the following call.
"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.generation_whisper import _dynamic_time_warping, _median_filter

MODEL = os.environ.get('ASR_MODEL', 'openai/whisper-large-v3-turbo')
PORT = int(os.environ.get('ASR_PORT', '8835'))
LANG = os.environ.get('ASR_LANGUAGE', 'english') or None     # rooms clip English-speaking streams; '' lets whisper detect
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('ASR_GPU_FRACTION', '0.06')))   # ~7.7 GB of the shared 128 GB

t0 = time.time()
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='sdpa').to('cuda')
proc = AutoProcessor.from_pretrained(MODEL)
asr = pipeline('automatic-speech-recognition', model=model, tokenizer=proc.tokenizer, feature_extractor=proc.feature_extractor,
               torch_dtype=torch.float16, device='cuda')
lock = threading.Lock()
SR, TPS = 16000, 50                     # samples per second; encoder positions per second
HEADS = model.generation_config.alignment_heads
TOK = proc.tokenizer
EOT = TOK.convert_tokens_to_ids('<|endoftext|>')
print(f'{MODEL} ready in {time.time() - t0:.1f} s on :{PORT}', flush=True)


def align(x):
    """Words with timings for <= 30 s of audio: decode, then one teacher-forced pass for the cross-attention and DTW
    (openai-whisper's find_alignment). The pipeline's return_timestamps='word' copies every step's attention to the CPU
    one tensor at a time and took 3-8 s per 6 s chunk, so live rooms fell ~10 minutes behind; this takes ~0.3 s."""
    feat = proc.feature_extractor(x, sampling_rate=SR, return_tensors='pt').input_features.to('cuda', torch.float16)
    text = [t for t in model.generate(feat, return_timestamps=False, task='transcribe', **({'language': LANG} if LANG else {}))[0].tolist() if t < EOT]
    if not text:
        return '', []
    lang = {'english': 'en'}.get(LANG, LANG) if LANG else 'en'
    prefix = TOK.convert_tokens_to_ids(['<|startoftranscript|>', f'<|{lang}|>', '<|transcribe|>', '<|notimestamps|>'])
    # the decoder has max_target_positions (448) positions: a maximal generation (a hallucinated run) plus the prompt and
    # the end token is one too many, which is a CUDA device-side assert that broke every later call (2026-10-21)
    text = text[:model.config.max_target_positions - len(prefix) - 1]
    model.set_attn_implementation('eager')                       # sdpa hands back no attention weights
    try:
        out = model(input_features=feat, decoder_input_ids=torch.tensor([prefix + text + [EOT]], device='cuda'), output_attentions=True)
    finally:
        model.set_attn_implementation('sdpa')
    w = torch.stack([out.cross_attentions[l][0, h] for l, h in HEADS]).float()[:, :, :max(1, round(len(x) / SR * TPS))]
    std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
    m = _median_filter((w - mean) / (std + 1e-6), 7).mean(0)
    rows, cols = _dynamic_time_warping(-m.double().cpu().numpy())    # every row, so the prompt rows soak up leading silence
    starts = cols[np.pad(np.diff(rows), (1, 0), constant_values=1).astype(bool)] / TPS
    times = starts[len(prefix):]                                 # a token starts where its own row starts (checked against the pipeline)
    groups = []                                                  # a token with a leading space starts a word; punctuation sticks
    for k, t in enumerate(text):
        if not groups or TOK.convert_ids_to_tokens(t).startswith('Ġ'):
            groups.append([k])
        else:
            groups[-1].append(k)
    words = []
    for g in groups:
        w_ = TOK.decode([text[k] for k in g]).strip()
        if w_:
            s0 = float(times[g[0]])
            e0 = float(times[g[-1] + 1]) if g[-1] + 1 < len(times) else len(x) / SR
            words.append({'w': w_, 's': s0, 'e': max(e0, s0 + 0.02)})
    # openai's hack: a sentence-final or first word that runs on is really followed (or preceded) by a pause
    d = [x_['e'] - x_['s'] for x_ in words]
    cap = 2 * min(0.7, float(np.median([v for v in d if v > 0.02] or [0.35])))
    for i, x_ in enumerate(words):
        if x_['e'] - x_['s'] > cap and (x_['w'][-1] in '.?!' or i == len(words) - 1):
            x_['e'] = x_['s'] + cap
        if i == 0 and x_['e'] - x_['s'] > cap:
            x_['s'] = x_['e'] - cap
    return TOK.decode(text).strip(), [{**x_, 's': round(x_['s'], 2), 'e': round(x_['e'], 2)} for x_ in words]


def pieces(x, most=28 * SR):
    """Audio over 30 s, cut at the quietest 0.2 s between 20 and 28 s into each piece."""
    out, a = [], 0
    while len(x) - a > 30 * SR:
        seg = x[a + 20 * SR: a + most]
        e = np.convolve((seg[:len(seg) // 800 * 800].reshape(-1, 800) ** 2).mean(1), np.ones(4), 'same')
        cut = a + 20 * SR + int(np.argmin(e)) * 800 + 400
        out.append((a, x[a:cut])); a = cut
    return out + [(a, x[a:])]


def transcribe(pcm):
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    t = time.time()
    if len(x) >= 800 and np.sqrt((x[:len(x) // 800 * 800].reshape(-1, 800) ** 2).mean(1)).max() < 0.002:
        return {'text': '', 'language': LANG and 'en', 'words': [], 'seconds': 0.0, 'aligner': 'silence'}   # else whisper says "Thank you."
    try:
        with lock, torch.inference_mode():
            texts, words = [], []
            for a, p in pieces(x):
                tx, ws = align(p)
                texts.append(tx); words += [{**w, 's': round(w['s'] + a / SR, 2), 'e': round(w['e'] + a / SR, 2)} for w in ws]
        return {'text': ' '.join(t_ for t_ in texts if t_), 'language': LANG and 'en', 'words': words, 'seconds': round(time.time() - t, 2), 'aligner': 'dtw'}
    except Exception as e:  # noqa: BLE001
        if 'CUDA' in repr(e) or 'Accelerator' in type(e).__name__:
            # a CUDA error poisons the context: every later call fails. Exit, and docker (--restart) brings a clean one
            print(f'CUDA error ({e!r}); exiting so the container restarts clean', flush=True)
            os._exit(1)
        print(f'fast aligner failed ({e!r}); using the pipeline', flush=True)
    return transcribe_pipeline(x, t)


def transcribe_pipeline(x, t):
    gk = {'task': 'transcribe', **({'language': LANG} if LANG else {})}
    kw = {'chunk_length_s': 30} if len(x) > 16000 * 30 else {}   # chunking short audio trips the word aligner (IndexError)
    with lock, torch.inference_mode():
        try:
            r = asr({'raw': x, 'sampling_rate': 16000}, return_timestamps='word', return_language=True, batch_size=1, generate_kwargs=gk, **kw)
        except IndexError:
            # word alignment failed on this audio: take sentence timings and spread each sentence's words across it
            r = asr({'raw': x, 'sampling_rate': 16000}, return_timestamps=True, return_language=True, batch_size=1, generate_kwargs=gk, **kw)
            split = []
            for c in r.get('chunks') or []:
                s0, e0 = c['timestamp']; ws = c['text'].split()
                if s0 is None or not ws:
                    continue
                e0 = e0 if e0 is not None else s0 + 0.35 * len(ws)
                for k, w in enumerate(ws):
                    split.append({'text': w, 'timestamp': (s0 + (e0 - s0) * k / len(ws), s0 + (e0 - s0) * (k + 1) / len(ws)), 'language': c.get('language')})
            r['chunks'] = split
    lang = next((c.get('language') for c in r.get('chunks') or [] if c.get('language')), None)
    words = []
    for c in r.get('chunks') or []:
        s, e = c['timestamp']
        w = c['text'].strip()
        if not w or s is None:
            continue
        words.append({'w': w, 's': round(float(s), 2), 'e': round(float(e if e is not None else s + 0.3), 2)})
    return {'text': r['text'].strip(), 'language': lang, 'words': words, 'seconds': round(time.time() - t, 2)}


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        self._send(200, {'ok': True, 'model': MODEL}) if self.path == '/health' else self._send(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/transcribe':
            return self._send(404, {'error': 'not found'})
        n = int(self.headers.get('Content-Length') or 0)
        if not 0 < n <= 16000 * 2 * 120:
            return self._send(400, {'error': 'send 1-120 s of s16le mono 16 kHz audio'})
        try:
            self._send(200, transcribe(self.rfile.read(n)))
        except Exception as e:  # noqa: BLE001
            self._send(500, {'error': str(e)[:300]})

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
