#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/prepare_ezvc_dataset.sh (--manifest FILE | --audio-dir DIR) [options] [-- extra prepare args]

Prepare expressive audio clips for EZ-VC fine-tuning.

Options:
  --dataset-name NAME       Dataset name. Default: Expressive_EZVC
  --out-dir DIR             Prepared output directory. Default: data/<dataset-name>_custom
  --manifest FILE           CSV/TSV/pipe-delimited or JSONL manifest.
  --audio-dir DIR           Directory to recursively scan when no manifest is provided.
  --audio-root DIR          Base directory for relative manifest audio paths.
  --audio-column NAME       Manifest audio column/key. Default: audio_path
  --label-column NAME       Optional expressive style label column/key.
  --require-labels          Require labels for every manifest row.
  --delimiter VALUE         Manifest delimiter override, e.g. '|' or ','.
  --device DEVICE           auto, cpu, cuda, mps, or xpu. Default: auto
  --xeus-layer N            XEUS hidden-state layer. Default: 14
  --repeat-cap N            Consecutive repeated units to keep. Default: 2
  --min-duration SECONDS    Minimum clip duration. Default: 0.3
  --max-duration SECONDS    Maximum clip duration. Default: 30.0
  --limit N                 Optional sample limit for smoke tests.
  --source-prosody-sidecar  Write F0/voicing/energy sidecars.
  --venv PATH               Virtualenv directory. Default: .venv
  -h, --help                Show this help.

Examples:
  scripts/prepare_ezvc_dataset.sh --audio-dir /path/to/clips
  scripts/prepare_ezvc_dataset.sh --manifest manifest.csv --audio-root /path/to/dataset --label-column style
  scripts/prepare_ezvc_dataset.sh --audio-dir /path/to/clips --limit 8 --source-prosody-sidecar
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

dataset_name="Expressive_EZVC"
out_dir=""
manifest=""
audio_dir=""
audio_root=""
audio_column="audio_path"
label_column=""
require_labels="false"
delimiter=""
device="auto"
xeus_layer="14"
repeat_cap="2"
min_duration="0.3"
max_duration="30.0"
limit=""
source_prosody_mode="none"
venv_dir="${repo_root}/.venv"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      dataset_name="${2:?Missing value for --dataset-name}"
      shift 2
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --manifest)
      manifest="$2"
      shift 2
      ;;
    --audio-dir)
      audio_dir="$2"
      shift 2
      ;;
    --audio-root)
      audio_root="$2"
      shift 2
      ;;
    --audio-column)
      audio_column="${2:?Missing value for --audio-column}"
      shift 2
      ;;
    --label-column)
      label_column="$2"
      shift 2
      ;;
    --require-labels)
      require_labels="true"
      shift
      ;;
    --delimiter)
      delimiter="${2:?Missing value for --delimiter}"
      shift 2
      ;;
    --device)
      device="${2:?Missing value for --device}"
      shift 2
      ;;
    --xeus-layer)
      xeus_layer="${2:?Missing value for --xeus-layer}"
      shift 2
      ;;
    --repeat-cap)
      repeat_cap="${2:?Missing value for --repeat-cap}"
      shift 2
      ;;
    --min-duration)
      min_duration="${2:?Missing value for --min-duration}"
      shift 2
      ;;
    --max-duration)
      max_duration="${2:?Missing value for --max-duration}"
      shift 2
      ;;
    --limit)
      limit="${2:?Missing value for --limit}"
      shift 2
      ;;
    --source-prosody-sidecar)
      source_prosody_mode="sidecar"
      shift
      ;;
    --venv)
      venv_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${manifest}" && -n "${audio_dir}" ]]; then
  echo "Use either --manifest or --audio-dir, not both." >&2
  exit 2
fi

if [[ -z "${manifest}" && -z "${audio_dir}" ]]; then
  usage >&2
  echo >&2
  echo "Missing required input: provide --manifest FILE or --audio-dir DIR." >&2
  exit 2
fi

if [[ "${venv_dir}" != /* ]]; then
  venv_dir="${repo_root}/${venv_dir}"
fi

python_bin="${venv_dir}/bin/python"
if [[ ! -x "${python_bin}" ]]; then
  echo "Missing executable: ${python_bin}" >&2
  echo "Set up the environment with: ${python_bin} -m pip install -e ${repo_root}" >&2
  exit 1
fi

export HOME="${EZVC_HOME:-${repo_root}/.home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${repo_root}/.matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${repo_root}/.numba_cache}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

mkdir -p "${HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cmd=(
  "${python_bin}"
  "${repo_root}/src/f5_tts/train/datasets/prepare_ezvc_expressive.py"
  --dataset-name "${dataset_name}"
  --audio-column "${audio_column}"
  --device "${device}"
  --xeus-layer "${xeus_layer}"
  --repeat-cap "${repeat_cap}"
  --min-duration "${min_duration}"
  --max-duration "${max_duration}"
  --source-prosody-mode "${source_prosody_mode}"
)

if [[ -n "${out_dir}" ]]; then
  cmd+=(--out-dir "${out_dir}")
fi
if [[ -n "${manifest}" ]]; then
  cmd+=(--manifest "${manifest}")
fi
if [[ -n "${audio_dir}" ]]; then
  cmd+=(--audio-dir "${audio_dir}")
fi
if [[ -n "${audio_root}" ]]; then
  cmd+=(--audio-root "${audio_root}")
fi
if [[ -n "${label_column}" ]]; then
  cmd+=(--label-column "${label_column}")
fi
if [[ "${require_labels}" == "true" ]]; then
  cmd+=(--require-labels)
fi
if [[ -n "${delimiter}" ]]; then
  cmd+=(--delimiter "${delimiter}")
fi
if [[ -n "${limit}" ]]; then
  cmd+=(--limit "${limit}")
fi
if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

echo "Preparing EZ-VC dataset: ${dataset_name}"
exec "${cmd[@]}"
