#!/usr/bin/env bash
# Lume AI — Launch Frontend and Backend Together
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

exec python3 run.py "$@"

