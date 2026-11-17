#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ASR_GPU="${ASR_GPU:-2}"
MINICPM_GPU="${MINICPM_GPU:-0}"
ASR_HOST="${ASR_HOST:-127.0.0.1}"
ASR_PORT="${ASR_PORT:-18180}"
CONSOLE_HOST="${CONSOLE_HOST:-0.0.0.0}"
CONSOLE_PORT="${CONSOLE_PORT:-8005}"
SSL_CERTFILE="${SSL_CERTFILE:-/root/paddlejob/workspace/env_run/avatar_full/audio_dual_avatar/minicpm_demo/certs/cert.pem}"
SSL_KEYFILE="${SSL_KEYFILE:-/root/paddlejob/workspace/env_run/avatar_full/audio_dual_avatar/minicpm_demo/certs/key.pem}"
ASR_MODEL_PATH="${ASR_MODEL_PATH:-${ROOT_DIR}/models/Qwen3-ASR-1.7B}"
MINICPM_MODEL_PATH="${MINICPM_MODEL_PATH:-${ROOT_DIR}/model}"
KNOWLEDGE_PATH="${KNOWLEDGE_PATH:-${ROOT_DIR}/knowledge/squad_zh_25.json}"
REF_AUDIO_PATH="${REF_AUDIO_PATH:-${ROOT_DIR}/assets/ref_audio.wav}"
QWEN_PYTHON="${QWEN_PYTHON:-${ROOT_DIR}/.venv-qwen/bin/python}"
MINICPM_PYTHON="${MINICPM_PYTHON:-${ROOT_DIR}/.venv/bin/python}"

asr_pid=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${asr_pid}" ]] && kill -0 "${asr_pid}" 2>/dev/null; then
        kill "${asr_pid}" 2>/dev/null || true
        wait "${asr_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

require_file() {
    if [[ ! -e "$1" ]]; then
        echo "missing required path: $1" >&2
        exit 1
    fi
}

require_file "${QWEN_PYTHON}"
require_file "${MINICPM_PYTHON}"
require_file "${ASR_MODEL_PATH}"
require_file "${MINICPM_MODEL_PATH}"
require_file "${KNOWLEDGE_PATH}"
require_file "${REF_AUDIO_PATH}"
require_file "${SSL_CERTFILE}"
require_file "${SSL_KEYFILE}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<USAGE
Start the Qwen3-ASR service and the MiniCPM streaming RAG console.

Environment overrides:
  ASR_GPU=2 MINICPM_GPU=0
  ASR_HOST=127.0.0.1 ASR_PORT=18180
  CONSOLE_HOST=0.0.0.0 CONSOLE_PORT=8005
  SSL_CERTFILE=... SSL_KEYFILE=...
  ASR_MODEL_PATH=... MINICPM_MODEL_PATH=...
  KNOWLEDGE_PATH=... REF_AUDIO_PATH=...

Open https://${CONSOLE_HOST}:${CONSOLE_PORT}/console after startup.
USAGE
    trap - EXIT INT TERM
    exit 0
fi

echo "starting Qwen3-ASR on GPU ${ASR_GPU} at http://${ASR_HOST}:${ASR_PORT}"
CUDA_VISIBLE_DEVICES="${ASR_GPU}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_HOST_IP=127.0.0.1 VLLM_PORT=18580 \
NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
"${QWEN_PYTHON}" -m qwen_asr.cli.demo_streaming \
  --asr-model-path "${ASR_MODEL_PATH}" \
  --host "${ASR_HOST}" --port "${ASR_PORT}" \
  --chunk-size-sec 1.0 --unfixed-chunk-num 2 --unfixed-token-num 5 &
asr_pid=$!

echo "waiting for Qwen3-ASR service..."
for _ in $(seq 1 180); do
    if ! kill -0 "${asr_pid}" 2>/dev/null; then
        wait "${asr_pid}" || true
        echo "Qwen3-ASR exited before becoming ready" >&2
        exit 1
    fi
    if (exec 3<>"/dev/tcp/${ASR_HOST}/${ASR_PORT}") 2>/dev/null; then
        exec 3>&-
        exec 3<&-
        break
    fi
    sleep 1
done

if ! (exec 3<>"/dev/tcp/${ASR_HOST}/${ASR_PORT}") 2>/dev/null; then
    echo "timed out waiting for Qwen3-ASR at ${ASR_HOST}:${ASR_PORT}" >&2
    exit 1
fi
exec 3>&-
exec 3<&-

echo "starting MiniCPM streaming RAG console at https://${CONSOLE_HOST}:${CONSOLE_PORT}/console"
CUDA_VISIBLE_DEVICES="${MINICPM_GPU}" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"${MINICPM_PYTHON}" -u "${ROOT_DIR}/console_server.py" \
  --knowledge "${KNOWLEDGE_PATH}" \
  --full-text \
  --model-path "${MINICPM_MODEL_PATH}" \
  --ref-audio "${REF_AUDIO_PATH}" \
  --qwen-asr-url "http://${ASR_HOST}:${ASR_PORT}" \
  --host "${CONSOLE_HOST}" --port "${CONSOLE_PORT}" \
  --ssl-certfile "${SSL_CERTFILE}" --ssl-keyfile "${SSL_KEYFILE}"
