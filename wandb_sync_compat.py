#!/usr/bin/env python3
"""Run `wandb sync` with the protobuf runtime forced to the Python implementation."""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    env = os.environ.copy()
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    proc = subprocess.run(["wandb", "sync", *sys.argv[1:]], env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
