#!/usr/bin/env bash
# Cut the 10-minute benchmark fixture from a provided lecture recording (read-only; never modified).
#   scripts/make_fixture.sh "data/lectures/<7-6 lecture>.mp4" [start=00:10:00]
# → eval/fixture_10min.m4a (mono 16 kHz AAC, gitignored)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:?usage: make_fixture.sh <lecture file> [start hh:mm:ss]}"
START="${2:-00:10:00}"
OUT="$ROOT/eval/fixture_10min.m4a"

[ -f "$SRC" ] || { echo "not found: $SRC" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found — brew install ffmpeg" >&2; exit 1; }

mkdir -p "$ROOT/eval"
ffmpeg -y -v error -ss "$START" -t 600 -i "$SRC" -vn -ac 1 -ar 16000 -c:a aac -b:a 64k "$OUT"
echo "$OUT: $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT") s"
