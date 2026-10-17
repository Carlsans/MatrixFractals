#!/usr/bin/env bash
# Launches the MatrixFractals GUI. First run creates a virtualenv and installs requirements-gui.txt.
set -e
cd "$(dirname "$0")"
VENV=.venv
if [ ! -x "$VENV/bin/python" ]; then
    rm -rf "$VENV"
    # some filesystems (FUSE, exFAT) cannot hold a venv: fall back to the home directory
    if ! python3 -m venv "$VENV" 2>/dev/null; then
        VENV="$HOME/.venvs/MatrixFractals"
        [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    fi
fi
if [ ! -f "$VENV/.deps-installed" ] || [ requirements-gui.txt -nt "$VENV/.deps-installed" ]; then
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r requirements-gui.txt && touch "$VENV/.deps-installed"
fi
exec "$VENV/bin/python" -m gui.app "$@"
