#!/usr/bin/env bash

set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 <scene.vox> <camera location>" >&2
    exit 1
fi

scene_path="$1"
camera_location="$2"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
trace_directory="$repo_root/traces"

cd "$repo_root"

if [[ ! -x ./build/vxray ]]; then
    echo "vxray is not built at ./build/vxray" >&2
    exit 1
fi

command -v metalperftrace >/dev/null || {
    echo "metalperftrace is not available on PATH" >&2
    exit 1
}
command -v jq >/dev/null || {
    echo "jq is not available on PATH" >&2
    exit 1
}

mkdir -p "$trace_directory"

revision="$(git rev-parse --short HEAD)"
location_tag="$(printf '%s' "$camera_location" | tr -cs '[:alnum:].-' '_')"
location_tag="${location_tag#_}"
location_tag="${location_tag%_}"
location_tag="${location_tag:-unknown}"
trace_prefix="${revision}_loc${location_tag}"
vxray_pid=""

cleanup() {
    if [[ -n "$vxray_pid" ]] && kill -0 "$vxray_pid" 2>/dev/null; then
        kill -TERM "$vxray_pid" 2>/dev/null || true
        wait "$vxray_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

./build/vxray "$scene_path" "$camera_location" &
vxray_pid="$!"

echo "Warming up vxray for 10 seconds, then running it for a further 10 seconds..."
sleep 20

cleanup
vxray_pid=""

trace_path="$(metalperftrace collect --last 10s --prefix "$trace_prefix" --json "$trace_directory" | jq -er '.traces[0]')"

echo "Collected trace: $trace_path"
metalperftrace overview --json "$trace_path" | jq '
    .[0].Layers[0]["Total Session Stats"]["Presented Frame Stats"] |
    {
        fps: .FPS,
        on_gpu_average_ms: .["On-GPU Walltime Stats"]["Average (ms)"],
        frame_count: .["Frame Count"]
    }
'
