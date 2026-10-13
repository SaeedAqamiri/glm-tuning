#!/usr/bin/env bash
set -euo pipefail
base=/home/saeed/glm-tuning-20261011
export GGML_Q8_PREFILL_SIGNED=1 GGML_PREFILL_SCALED_K=1
export GGML_FA_PREPACK=1
export GGML_MOE_WIDE_M=64
export GGML_AMX_MOE=1 GGML_CPU_FTZ=1 GGML_TREE_BARRIER=1
export GGML_AMX_NUMA_ROWS=1 GGML_NUMA_PIN=1 GGML_AMX_MOE_SMALL=1
export GGML_AMX_NUMA_DENSE=1 GGML_ROWS_CHUNKS=1
export GGML_AMX_SHARED_DECODE=1 GGML_AMX_SMALL_ROWS=1 LLAMA_MTP_ADAPTIVE=0
export GGML_CONCAT_CHUNKS=1 GGML_VNNI_ROWS2=1 GGML_COPY_ROWS=1
export GGML_CPU_SMALL_CHAIN=1 GGML_CPU_SMALL_CHAIN_MAX=8192
export GGML_MTP_SHARED_POOL=1 GGML_FUSE_QKV=1 GGML_FUSE_MOE_PAIR=1
export GGML_K_INTERLEAVE=1 GGML_Q8_INTERLEAVE=1 GGML_SMALL_BATCH_MAX=8
exec numactl --interleave=all "$base/prefill-400-v2/build/bin/llama-server" \
  -m /srv/models/glm53-gguf/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --host 127.0.0.1 --port 8019 -t 64 -tb 112 -c 32768 -np 4 \
  --numa distribute --load-mode mmap --lazy-mode off --no-cache-prompt \
  --spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 0 \
  -b 4096 -ub 4096
