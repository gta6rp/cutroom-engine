# Cutroom Engine

The live-stream watching and per-streamer agent engine behind Cutroom.
This repository contains the Python pipeline, not the Cutroom website or token-launch service.

Each room follows one streamer. During a live stream it watches video, listens to speech, reads chat, finds moments,
and cuts vertical clips. Between streams its agent can search past viewer-clipped moments and produce edits.
Rooms maintain separate activity histories while sharing scheduling, rendering limits and optional GPU transcription.

```text
Kick / Twitch → rolling HLS video + chat → transcription + signals
                                             ↓
                               semantic scan → editor decision
                                             ↓
                               framing + captions → local clips

Per-streamer fleet → live session or offline archive search → agent state
```

## Core source

| File | Responsibility |
| --- | --- |
| [`pipeline/live/live.py`](pipeline/live/live.py) | Stream capture, chat, transcription, moment detection, clip rendering and progress events. |
| [`pipeline/live/fleet.py`](pipeline/live/fleet.py) | Per-streamer state, live/offline scheduling and archive searches. |
| [`pipeline/live/room_agents.py`](pipeline/live/room_agents.py) | Keeps agents for configured and recently watched rooms. |
| [`pipeline/live/rooms_watch.py`](pipeline/live/rooms_watch.py) | Optional scheduler that prioritizes configured live sources. |
| [`pipeline/live/semantic.py`](pipeline/live/semantic.py) | Finds story beats in transcripts, including quieter moments. |
| [`pipeline/live/judge.py`](pipeline/live/judge.py) | Scores moments and writes evidence-grounded hooks and captions. |
| [`pipeline/asr/server.py`](pipeline/asr/server.py) | Optional shared GPU Whisper transcription service. |
| [`pipeline/edit.py`](pipeline/edit.py) | Vertical framing, captions and audio processing. |
| [`pipeline/live/gateway.py`](pipeline/live/gateway.py) | Read-only local API for agent state and live progress. |

## Local setup

Linux, Python 3.11+ and FFmpeg/FFprobe with libass and libx264 are expected.
Install FFmpeg on your system first. Then, from this repository:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/bootstrap.py
```

The bootstrap creates empty output folders and links FFmpeg/FFprobe into `.venv/bin`, where the original pipeline
expects them. It does not start a service, download a model, watch a stream or publish anything.

Start one room, replacing `your_streamer` with a source you are authorized to clip:

```bash
.venv/bin/python pipeline/live/live.py your_streamer --minutes 30 --judge off --semantic-every 0 --studio-edits 0 --asr ''
```

The CPU speech model downloads on first use. `--judge off` uses the rule-based moment editor; later quality and
safety gates can still call the configured editor, so it does not make the entire pipeline LLM-free.
`--semantic-every 0` disables the separate semantic scans. For LLM editing, install and
authenticate your own Claude CLI and use `--judge sonnet`, or your own Codex CLI and `--judge codex`.
These optional editor calls use your account and its usage limits. No account credentials are supplied.

The room listens on `127.0.0.1:8820`:

```bash
curl http://127.0.0.1:8820/state
curl -N http://127.0.0.1:8820/events
```

Finished clips and session state land in `pipeline/live/runs/<streamer>-<timestamp>/`.
Stop the process with SIGTERM so it can save its final state. The API binds to loopback; no public tunnel is included.

## Multiple streamer agents

```bash
.venv/bin/python pipeline/live/fleet.py --rooms your_streamer,twitch:another_streamer --max-live 2 --judge sonnet
```

The fleet checks live status and searches past clips when rooms are offline. Use `--dig off` to disable archive
searches. Agent state is written to `pipeline/live/agents/<slug>.json`. One room means one streamer workspace and
history, not a dedicated model running continuously for every streamer.

For the split scheduler setup, put handles such as `your_streamer` and `twitch:another_streamer` in
`config/watch.json`, run `room_agents.py` for offline work, and run `rooms_watch.py --launched-only` for live sessions.
The legacy flag name refers to the watch list; there is no coin-launch dependency in this repository.
Use one scheduling arrangement at a time to avoid duplicate work.

The optional gateway exposes all agents through one local address:

```bash
.venv/bin/python pipeline/live/gateway.py
curl http://127.0.0.1:8829/agents
curl http://127.0.0.1:8829/agent/your_streamer
curl http://127.0.0.1:8829/your_streamer/state
curl -N http://127.0.0.1:8829/your_streamer/events
```

## Configuration and optional GPU service

`config/kickrates.json` starts with an empty `rates` list. Explicit `fleet.py --rooms` handles do not need entries.
Optional entries have `kick`, `name`, `usd` (per 100K views) and `program` fields; supply your own verified values.
`config/lore.json` and `config/library-index.json` are empty optional context libraries.
`pipeline/live/profiles.json` holds optional per-streamer crop profiles. None of the production configuration is included.

CPU transcription works without the shared GPU service. For GPU transcription, install the PyTorch build appropriate
for your hardware, then `requirements-gpu.txt`, and run `pipeline/asr/server.py` in that environment. Set
`ASR_GPU_FRACTION` for your hardware; the original default was tuned for a 128 GB DGX Spark. The server binds to
`127.0.0.1:8835`, and rooms use `CUTROOM_ASR` to locate it. Model downloads and inference happen only when you run
the corresponding component. The watchers do not start Docker containers automatically in this extraction.
`resemblyzer` is optional for speaker voice matching.

The basic clip renderer includes the TikTok Sans fonts and their accompanying SIL Open Font License.
Advanced studio edits expect your own cleared music library at `portfolio/music/library.json`; music, recordings,
voiceprints, model weights and social accounts are not distributed here.

## Scope of this publication

This is an extraction of the working Cutroom source, with standalone configuration paths, a smaller read-only gateway
and local clip output. Website exports, social publishing clients, production settings, runtime histories and footage
are excluded. Some internal names still reference rooms, coins or the original deployment. `SOURCE_MAP.json` records
the hashes of the original and extracted Python source. It does not include private filesystem paths.

The public extraction has offline smoke checks in `tests/`. They check imports, configuration, agent state, gateway
responses and the local-output boundary without starting a live capture, downloading models or calling paid editors.
They do not establish fresh live performance on every platform or hardware configuration.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Use footage you have permission to process and follow each source's clipping terms. Capturing a public stream does
not itself grant redistribution rights. Third-party font licensing is preserved in `pipeline/fonts/TikTokSans-OFL.txt`;
no project-wide software license has been selected for this initial publication.
