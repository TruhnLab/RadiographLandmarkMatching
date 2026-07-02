#!/usr/bin/env bash
set -e

export PYTHONPATH="/app:/app/ThirdParty"
export MPLBACKEND=Agg
export MKL_SERVICE_FORCE_INTEL=1

# Number of uvicorn workers is kept at 1: each worker would load the full
# model into GPU memory, and inference is serialized anyway. Scale by running
# multiple containers (one per GPU) instead.
exec uvicorn serve:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
