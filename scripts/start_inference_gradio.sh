#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/start_inference_gradio.sh [options] [-- extra gradio args]

Start the EZ-VC inference Gradio app.

Options:
  --host HOST       Bind host. Default: 127.0.0.1
  --port PORT       Bind port. Default: 7861
  --venv PATH       Virtualenv directory. Default: .venv
  --share           Enable Gradio sharing.
  --inbrowser       Ask Gradio to open a browser.
  -h, --help        Show this help.

Examples:
  scripts/start_inference_gradio.sh
  scripts/start_inference_gradio.sh --port 7863 --share
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

host="127.0.0.1"
port="7861"
venv_dir="${repo_root}/.venv"
share="false"
inbrowser="false"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host="${2:?Missing value for --host}"
      shift 2
      ;;
    --port)
      port="${2:?Missing value for --port}"
      shift 2
      ;;
    --venv)
      venv_dir="$2"
      shift 2
      ;;
    --share)
      share="true"
      shift
      ;;
    --inbrowser)
      inbrowser="true"
      shift
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

if [[ "${venv_dir}" != /* ]]; then
  venv_dir="${repo_root}/${venv_dir}"
fi

app_bin="${venv_dir}/bin/ez-vc_infer-gradio"
if [[ ! -x "${app_bin}" ]]; then
  echo "Missing executable: ${app_bin}" >&2
  echo "Set up the environment with: ${venv_dir}/bin/python -m pip install -e ${repo_root}" >&2
  exit 1
fi

export HOME="${EZVC_HOME:-${repo_root}/.home}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${repo_root}/.matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${repo_root}/.numba_cache}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

mkdir -p "${HOME}" "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

cmd=("${app_bin}" --host "${host}" --port "${port}")
if [[ "${share}" == "true" ]]; then
  cmd+=(--share)
fi
if [[ "${inbrowser}" == "true" ]]; then
  cmd+=(--inbrowser)
fi
if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

echo "Starting EZ-VC inference Gradio app at http://${host}:${port}"
exec "${cmd[@]}"
