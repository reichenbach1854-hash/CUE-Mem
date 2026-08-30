#!/usr/bin/env bash
#SBATCH -J bench_base
#SBATCH -o ./logs/bench_base_%j.out
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH --mem 80G
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --exclude=gpu04,gpu05

# This is the only retained shell wrapper.  The shared helper functions are
# kept inline so the wrapper remains self-contained after the other .sh files
# are removed from the public repository.
set -euo pipefail

_RQ1_RQ2_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_RQ1_RQ2_PACKAGE_ROOT="$(cd -- "${_RQ1_RQ2_SCRIPT_DIR}/.." && pwd)"
_RQ1_RQ2_REPOSITORY_ROOT="$(cd -- "${_RQ1_RQ2_PACKAGE_ROOT}/../.." && pwd)"

rq1_rq2_prepare_runtime() {
    export CUE_MEM_PROJECT_ROOT="${CUE_MEM_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${_RQ1_RQ2_REPOSITORY_ROOT}}}"

    if [[ -n "${CUE_MEM_BENCHMARK_CONDA_ENV:-}" ]]; then
        if ! command -v conda >/dev/null 2>&1; then
            echo "CUE_MEM_BENCHMARK_CONDA_ENV is set but conda is unavailable" >&2
            return 1
        fi
        # shellcheck disable=SC1090
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CUE_MEM_BENCHMARK_CONDA_ENV}"
    fi
}

rq1_rq2_run() {
    local module="$1"
    shift
    local python_bin="${CUE_MEM_PYTHON:-python}"
    local script_path="${_RQ1_RQ2_PACKAGE_ROOT}/benchmark/run/${module}.py"
    if [[ ! -f "${script_path}" ]]; then
        echo "Benchmark script not found: ${script_path}" >&2
        return 1
    fi
    (cd "${CUE_MEM_PROJECT_ROOT}" && "${python_bin}" "${script_path}" "$@")
}

rq1_rq2_vllm_command() {
    printf '%s\n' "${CUE_MEM_VLLM_COMMAND:-vllm}"
}

rq1_rq2_start_vllm() {
    local max_attempts="${1:-${CUE_MEM_VLLM_WAIT_ATTEMPTS:-360}}"
    local model_path="${CUE_MEM_VLLM_MODEL_PATH:-}"
    local health_url="${CUE_MEM_VLLM_HEALTH_URL:-}"
    local port="${CUE_MEM_VLLM_PORT:-${VLLM_PORT:-}}"

    if [[ -z "${model_path}" ]]; then
        echo "Set CUE_MEM_VLLM_MODEL_PATH before starting the model server" >&2
        return 1
    fi
    if [[ -z "${health_url}" ]]; then
        echo "Set CUE_MEM_VLLM_HEALTH_URL before starting the model server" >&2
        return 1
    fi
    if [[ -z "${port}" ]]; then
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            port="$((8000 + (SLURM_JOB_ID % 450) * 2))"
        else
            port="8000"
        fi
    fi

    export VLLM_PORT="${port}"
    local log_root="${CUE_MEM_LOG_ROOT:-${CUE_MEM_BENCHMARK_ROOT:-${CUE_MEM_PROJECT_ROOT}/scripts/RQ1_RQ2/benchmark}/logs}"
    mkdir -p "${log_root}"
    local log_file="${log_root}/vllm_server_${SLURM_JOB_ID:-manual}.log"
    "$(rq1_rq2_vllm_command)" "${model_path}" \
        --served-model-name "${CUE_MEM_VLLM_SERVED_MODEL:-qwen36-35b-a3b-fp8}" \
        --host "${CUE_MEM_VLLM_HOST:-0.0.0.0}" \
        --port "${VLLM_PORT}" \
        --trust-remote-code \
        --max-model-len "${CUE_MEM_VLLM_MAX_MODEL_LEN:-65536}" \
        --reasoning-parser "${CUE_MEM_VLLM_REASONING_PARSER:-qwen3}" \
        --language-model-only \
        --enable-prefix-caching \
        >"${log_file}" 2>&1 &
    VLLM_PID=$!
    export VLLM_PID

    local attempt
    for ((attempt = 1; attempt <= max_attempts; attempt++)); do
        if curl -sf --connect-timeout 3 --max-time 8 "${health_url}" >/dev/null 2>&1; then
            echo "Model server is ready on port ${VLLM_PORT}"
            return 0
        fi
        if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
            echo "Model server exited; inspect ${log_file}" >&2
            return 1
        fi
        sleep "${CUE_MEM_VLLM_WAIT_INTERVAL:-5}"
    done
    echo "Model server did not become ready; inspect ${log_file}" >&2
    return 1
}

rq1_rq2_cleanup_vllm() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
        wait "${VLLM_PID}" >/dev/null 2>&1 || true
    fi
}

rq1_rq2_prepare_runtime
export CUE_MEM_LLM_MODEL="${CUE_MEM_LLM_MODEL:-qwen36-35b-a3b-fp8}"
trap rq1_rq2_cleanup_vllm EXIT
rq1_rq2_start_vllm 240
rq1_rq2_run run_bench \
    --llm_name qwen3.6-35b-a3b \
    --memory_name FUMemory LTMemory GAMemory MGMemory RFMemory \
    --caption_category base \
    --all_datasets \
    --save_results \
    --max_workers 32 \
    --with_reasoning
