#!/usr/bin/env python3
"""Create empty runtime folders and connect system FFmpeg tools to the project venv."""
from pathlib import Path
import shutil
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    bindir = root / '.venv/bin'
    if not (bindir / 'python').exists():
        raise SystemExit('First run: python3 -m venv .venv')
    for name in ('ffmpeg', 'ffprobe'):
        target = bindir / name
        if target.exists():
            continue
        binary = shutil.which(name)
        if not binary:
            raise SystemExit(f'Install {name} on PATH, then rerun this script.')
        target.symlink_to(Path(binary).resolve())
    for rel in ('runtime', 'pipeline/live/runs', 'pipeline/live/agents',
                'pipeline/work', 'portfolio/clips', 'posting/queue'):
        (root / rel).mkdir(parents=True, exist_ok=True)
    print('Local runtime directories and FFmpeg links are ready. No services started.')


if __name__ == '__main__':
    main()
