#!/usr/bin/env bash
set -euo pipefail
export GGML_Q8_PREFILL_SIGNED=1 GGML_PREFILL_SCALED_K=1
exec /home/saeed/glm-tuning-20261011/run-prefill-400.sh -b 4096 -ub 4096 "$@"
