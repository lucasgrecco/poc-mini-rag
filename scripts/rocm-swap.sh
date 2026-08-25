#!/bin/sh
# Swaps the CUDA torch from uv.lock for the ROCm build. Runs inside the cli
# container; idempotent, so callers can run it after every uv sync.
#
# The desired venv state is not representable in uv.lock, so every reconciliation
# with the lock is a revert. docker-compose.rocm.yml sets UV_NO_SYNC=1 to stop the
# implicit ones; this script repairs the explicit ones (uv sync).
set -e

ROCM_INDEX="${ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.2}"

# --no-sync is load-bearing: a plain `uv run` reconciles the venv with uv.lock and
# reinstalls the CUDA torch, so the check would cause the revert it tests for.
# Keep stderr (it is the diagnostic when torch will not import) but decide on the
# last line only, so a uv warning ahead of the output cannot break the match.
out=$(uv run --no-sync python -c \
  'import torch; print("rocm" if torch.version.hip else "cuda")' 2>&1) || true
state=$(printf '%s\n' "$out" | tail -n 1)

case "$state" in
  rocm) echo "torch is already a ROCm build - nothing to do"; exit 0 ;;
  cuda) ;;
  *)
    echo "torch does not import - run 'make sync' first:" >&2
    echo "$out" >&2
    exit 1
    ;;
esac

echo "swapping CUDA torch for the ROCm build from $ROCM_INDEX"

# --index-url replaces PyPI outright, deliberately: uv gives --extra-index-url
# priority over --index-url, so adding PyPI as an extra makes the default
# first-index strategy match torch there and silently install the CUDA build.
# Everything torch needs beyond pytorch-triton-rocm is already in the venv from
# uv sync, and uv leaves satisfied deps alone, so replacing PyPI costs nothing here.
#
# The ROCm triton owns the same `triton/` import path as the CUDA triton, so it has
# to be rewritten alongside torch - otherwise a revert leaves ROCm triton files under
# CUDA triton metadata, which uv sync considers satisfied and never repairs. Its
# package name follows the ROCm generation (triton-rocm on 7.x, pytorch-triton-rocm
# on 6.x), so name both: --reinstall-package no-ops on a package that is not resolved.
uv pip install \
  --index-url "$ROCM_INDEX" \
  --reinstall-package torch \
  --reinstall-package triton-rocm \
  --reinstall-package pytorch-triton-rocm \
  torch
