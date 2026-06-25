#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run TotalSegmentator GI segmentation through Docker Desktop.

This wrapper accepts the same core flags used by TotalSegmentator:
  -i, --input    Input CT NIfTI
  -o, --output   Output mask directory

Example:
  scripts/run_totalseg_docker_gi.sh \
    -i data/raw/btcv_abdomen/case0001/img0001.nii.gz \
    -o data/derived/gi_masks/btcv_case0001_totalseg \
    --roi_subset stomach duodenum small_bowel colon --fast

Environment overrides:
  DOCKER_BIN              Docker CLI path
  DOCKER_HOST             Docker socket, defaults to ~/.docker/run/docker.sock
  TOTALSEG_DOCKER_IMAGE   Defaults to wasserth/totalsegmentator:2.14.0
  TOTALSEG_PLATFORM       Defaults to linux/amd64 for Docker Desktop on Apple Silicon
  TOTALSEG_MIN_FREE_GB    Defaults to 35
  TOTALSEG_ALLOW_LOW_DISK Set to 1 to bypass the free-space guard
  TOTALSEG_PLAN_ONLY      Set to 1 to print the command without running it
USAGE
}

input_path=""
output_dir=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -i|--input)
      input_path="${2:-}"
      shift 2
      ;;
    -o|--output)
      output_dir="${2:-}"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$input_path" || -z "$output_dir" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$input_path" ]]; then
  echo "Input CT not found: $input_path" >&2
  exit 2
fi

if [[ -z "${DOCKER_BIN:-}" ]]; then
  if [[ -x "/Applications/Docker.app/Contents/Resources/bin/docker" ]]; then
    DOCKER_BIN="/Applications/Docker.app/Contents/Resources/bin/docker"
  else
    DOCKER_BIN="$(command -v docker || true)"
  fi
fi

if [[ -z "${DOCKER_BIN:-}" || ! -x "$DOCKER_BIN" ]]; then
  echo "Docker CLI not found. Start Docker Desktop or set DOCKER_BIN." >&2
  exit 2
fi

docker_host="${DOCKER_HOST:-unix://${HOME}/.docker/run/docker.sock}"
if [[ "$docker_host" == unix://* ]]; then
  socket_path="${docker_host#unix://}"
  if [[ ! -S "$socket_path" ]]; then
    echo "Docker socket not found: $socket_path" >&2
    exit 2
  fi
  if ! curl --silent --show-error --max-time 5 --unix-socket "$socket_path" http://localhost/_ping >/dev/null; then
    echo "Docker daemon is not responding on $socket_path." >&2
    exit 2
  fi
fi

mkdir -p "$output_dir"
input_dir="$(cd "$(dirname "$input_path")" && pwd -P)"
input_name="$(basename "$input_path")"
output_abs="$(cd "$output_dir" && pwd -P)"

min_free_gb="${TOTALSEG_MIN_FREE_GB:-35}"
free_kb="$(df -k "$output_abs" | awk 'NR == 2 {print $4}')"
free_gb="$(awk -v kb="$free_kb" 'BEGIN {printf "%.1f", kb / 1024 / 1024}')"

if [[ "${TOTALSEG_ALLOW_LOW_DISK:-0}" != "1" ]]; then
  if awk -v free="$free_gb" -v min="$min_free_gb" 'BEGIN {exit !(free < min)}'; then
    cat >&2 <<EOF
Refusing to start TotalSegmentator Docker run: only ${free_gb} GiB free, guard requires ${min_free_gb} GiB.

The official TotalSegmentator Docker image is large and also needs unpack/model-cache space.
Free more disk, lower TOTALSEG_MIN_FREE_GB after review, or set TOTALSEG_ALLOW_LOW_DISK=1.
EOF
    if [[ "${TOTALSEG_PLAN_ONLY:-0}" != "1" ]]; then
      exit 3
    fi
  fi
fi

image="${TOTALSEG_DOCKER_IMAGE:-wasserth/totalsegmentator:2.14.0}"
platform="${TOTALSEG_PLATFORM:-linux/amd64}"

cmd=(
  "$DOCKER_BIN"
  --host "$docker_host"
  run
  --rm
  --platform "$platform"
  -v "${input_dir}:/input:ro"
  -v "${output_abs}:/output"
  "$image"
  TotalSegmentator
  -i "/input/${input_name}"
  -o /output
  "${extra_args[@]}"
)

printf 'Planned TotalSegmentator Docker command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'
printf 'Free disk at output path: %s GiB\n' "$free_gb"

if [[ "${TOTALSEG_PLAN_ONLY:-0}" == "1" ]]; then
  exit 0
fi

exec "${cmd[@]}"
