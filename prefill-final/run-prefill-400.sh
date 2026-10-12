#!/usr/bin/env bash
set -euo pipefail
experiment=/home/saeed/glm-tuning-20261011
export GGML_AMX_MOE=1 GGML_CPU_FTZ=1 GGML_TREE_BARRIER=1
export GGML_AMX_NUMA_ROWS=1 GGML_NUMA_PIN=1 GGML_AMX_MOE_SMALL=1
export GGML_AMX_NUMA_DENSE=1 GGML_ROWS_CHUNKS=1 GGML_AMX_SHARED_DECODE=1 GGML_AMX_SMALL_ROWS=1
export LLAMA_MTP_ADAPTIVE=0
export GGML_CONCAT_CHUNKS=1 GGML_VNNI_ROWS2=1 GGML_COPY_ROWS=1 GGML_CPU_SMALL_CHAIN=1 GGML_CPU_SMALL_CHAIN_MAX=8192 GGML_MTP_SHARED_POOL=1 GGML_FUSE_QKV=1 GGML_FUSE_MOE_PAIR=1 GGML_K_INTERLEAVE=1 GGML_Q8_INTERLEAVE=1 GGML_SMALL_BATCH_MAX=8
if [[ "${FUSE_GATE:-0}" == 1 ]]; then export GGML_FUSE_GATE=1; else unset GGML_FUSE_GATE; fi
unset GGML_AMX_DENSE_SMALL GGML_AMX_MOE_WIDE GGML_Q8_SPLIT
cache_flags=()
if [[ "${NO_CACHE_PROMPT:-0}" == 1 ]]; then cache_flags+=(--no-cache-prompt); fi
exec numactl --interleave=all "$experiment/prefill-400/build/bin/llama-server" \
  -m /srv/models/glm53-gguf/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --host "${BIND_HOST:-127.0.0.1}" --port "${PORT:-8017}" \
  -t "${DECODE_THREADS:-64}" -tb "${PREFILL_THREADS:-112}" \
  -c "${CTX:-32768}" -np 4 --numa distribute --load-mode mmap --lazy-mode off \
  "${cache_flags[@]}" --spec-type "${SPEC_TYPE:-draft-mtp}" \
  --spec-draft-n-max "${DRAFT_MAX:-1}" --spec-draft-p-min "${DRAFT_PMIN:-0}" "$@"
