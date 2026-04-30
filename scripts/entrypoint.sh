#!/usr/bin/env bash
# Entrypoint for vllm-awq4-qwen — starts the correct vLLM service
# based on SERVICE_MODE env var. Defaults to LLM if unset.
set -euo pipefail

exec_vllm() {
  echo ">>> vLLM $1 service starting: $2"
  exec vllm serve "$2" "${@:3}"
}

case "${SERVICE_MODE:-llm}" in
  llm)
    exec_vllm "LLM" "${VLLM_MODEL_ID}" \
      --host "${HOST:-0.0.0.0}" \
      --port "${PORT:-8000}" \
      --served-model-name "${VLLM_SERVED_MODEL_NAME}" \
      --attention-backend "${VLLM_ATTENTION_BACKEND}" \
      --mm-encoder-attn-backend "${VLLM_MM_ENCODER_ATTN_BACKEND}" \
      --reasoning-parser "${VLLM_REASONING_PARSER}" \
      --tool-call-parser "${VLLM_TOOL_CALL_PARSER}" \
      --enable-auto-tool-choice \
      --enforce-eager \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTIL}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --speculative-config "${VLLM_SPECULATIVE_CONFIG}"
    ;;
  asr)
    exec_vllm "ASR" "${VLLM_ASR_MODEL_ID}" \
      --host "${HOST:-0.0.0.0}" \
      --port "${VLLM_ASR_HOST_PORT}" \
      --supported-tasks "${VLLM_ASR_SUPPORTED_TASKS}" \
      --enforce-eager \
      --max-model-len "${VLLM_ASR_MAX_MODEL_LEN}" \
      --mm-encoder-attn-backend "${VLLM_MM_ENCODER_ATTN_BACKEND}" \
      --gpu-memory-utilization "${VLLM_ASR_GPU_MEMORY_UTIL}" \
      --max-num-seqs "${VLLM_ASR_MAX_NUM_SEQS}"
    ;;
  *)
    echo "ERROR: SERVICE_MODE must be 'llm' or 'asr', got '${SERVICE_MODE}'"
    exit 1
    ;;
esac
