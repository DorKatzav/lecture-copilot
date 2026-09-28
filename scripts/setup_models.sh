#!/usr/bin/env bash
# Pull the local models Lecture Copilot uses. Idempotent: re-running only verifies.
# Starts `ollama serve` in the background if nothing answers on :11434 (log: runs/ollama.log).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS=(gemma3:12b bge-m3)  # = config LIVE_MODEL/DIGEST_MODEL + EMBED_MODEL (D-M0-10)
HOST="http://127.0.0.1:11434"

command -v ollama >/dev/null || { echo "ollama not found — brew install ollama" >&2; exit 1; }

if ! curl -fsS "$HOST/api/version" >/dev/null 2>&1; then
  mkdir -p "$ROOT/runs"
  echo "starting ollama serve (log: runs/ollama.log)"
  nohup ollama serve >"$ROOT/runs/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do
    curl -fsS "$HOST/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fsS "$HOST/api/version" >/dev/null || { echo "ollama serve did not come up" >&2; exit 1; }
fi

for m in "${MODELS[@]}"; do
  echo "== $m"
  ollama pull "$m"
done

ollama list
