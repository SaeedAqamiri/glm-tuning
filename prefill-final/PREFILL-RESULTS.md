# Prefill optimization results

Base: validated-real-mtp-18p4 (1a1d1ab). Source: prefill-400 detached worktree. Changes are saved as prefill-final.patch; no new commit or upstream push.

| Input | Baseline prefill | Final prefill | Baseline decode | Final decode |
|---|---:|---:|---:|---:|
| English | 154.538 | 206.655 | 18.090 | 17.676 |
| Persian | 147.252 | 206.052 | 17.816 | 17.508 |

Rates are tokens/second, single active request, no prompt cache, fixed 1860/4397-token prompts, temperature0/seed42. Full messages and draft acceptance counts equal the reference checkpoint. These two inputs are not an exhaustive quality benchmark.

## Retained changes

- Q8 prefill: convert packed unsigned weights to signed AMX scratch, avoiding repeated per-row compensation. M1/M2 decode paths unchanged.
- Q4_K/Q5_K prefill: apply integer scales to low/high byte planes, accumulate a complete 256-element block with AMX, then retain original floating-point accumulation and min correction. Integer results remain exact.
- Batch/ubatch4096; decode64/prefill112 threads.
- 384 QKV/raw-extreme tests, 72 MoE tests and 12 large MoE tests passed byte-for-byte on final build.

## Rejected or unused experiments

GDN splitting (288 exact tests), MoE128-row blocks and cyclic scheduling showed no compelling extra gain. Their source backups and logs remain outside the final source. Prefill56 and224 threads did not beat112. Microbench gains alone are not treated as model throughput.

## Run and rollback

Candidate: localhost8017, restart with run-prefill-validated.sh. Stable8016 still runs original d079289 build; it was not replaced. To disable kernel changes use run-prefill-400.sh without GGML_Q8_PREFILL_SIGNED and GGML_PREFILL_SCALED_K. The model files and system settings are unchanged.

Build: cmake -S prefill-400 -B prefill-400/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AMX=ON -DGGML_OPENMP=OFF -DCMAKE_CXX_FLAGS=-DGGML_AMX_Q8_UNSIGNED=1; cmake --build prefill-400/build --target llama-server -j24.

Target400 has not been reached. Profiling at baseline identified dense/expert matmul as the main costs, followed by flash attention. The retained changes improve prefill while preserving tested complete outputs.
