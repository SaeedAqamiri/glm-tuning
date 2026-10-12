#!/usr/bin/env bash
set -euo pipefail
experiment=/home/saeed/glm-tuning-20261011
export GGML_AMX_MOE=1 GGML_CPU_FTZ=1 GGML_TREE_BARRIER=1
export GGML_AMX_NUMA_ROWS=1 GGML_NUMA_PIN=1 GGML_AMX_MOE_SMALL=4
export GGML_AMX_NUMA_DENSE=1 GGML_ROWS_CHUNKS=1
if [[ "${FUSE_GATE:-0}" == 1 ]]; then export GGML_FUSE_GATE=1; else unset GGML_FUSE_GATE; fi
unset GGML_AMX_DENSE_SMALL GGML_AMX_MOE_WIDE GGML_Q8_SPLIT
exec numactl --interleave=all "$experiment/llama/build-batched/bin/llama-server" \
  -m /srv/models/glm53-gguf/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --host "${BIND_HOST:-127.0.0.1}" --port "${PORT:-8016}" \
  -t "${DECODE_THREADS:-48}" -tb "${PREFILL_THREADS:-56}" \
  -c 32768 -np 4 --numa distribute --load-mode mmap --lazy-mode off \
  --no-cache-prompt --spec-type "${SPEC_TYPE:-none}" \
  --spec-draft-n-max "${DRAFT_MAX:-7}" --spec-draft-p-min "${DRAFT_PMIN:-0.7}" "$@"
