#!/usr/bin/env bash
# Start the Ollama server the way this project needs it (D-M1-4). Logs append to runs/ollama.log.
#
# Ollama 0.34 runs models inside llama-server, which by default keeps the KV state of every distinct prompt in
# RAM, up to 8 GiB (--cache-ram). Chunk prompts never repeat, so that cache only fills memory and then swap
# (measured: +940 MB per chunk, pressure "warning"). LLAMA_ARG_CACHE_RAM=0 turns it off; the runner inherits it.
#   scripts/ollama_serve.sh            # start (refuses if a server is already up)
#   scripts/ollama_serve.sh --restart  # stop the running `ollama serve` first
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "${1:-}" = "--restart" ]; then
  pkill -f "^ollama serve" || true
  sleep 3
fi
if curl -sf http://127.0.0.1:11434/api/version >/dev/null; then
  echo "ollama is already running — use --restart to apply LLAMA_ARG_CACHE_RAM=0" >&2
  exit 1
fi
mkdir -p "$ROOT/runs"
LLAMA_ARG_CACHE_RAM=0 nohup ollama serve >> "$ROOT/runs/ollama.log" 2>&1 &
for _ in $(seq 1 20); do
  curl -sf http://127.0.0.1:11434/api/version >/dev/null && { echo "ollama serve up (prompt cache off)"; exit 0; }
  sleep 0.5
done
echo "ollama serve did not come up — see runs/ollama.log" >&2
exit 1
