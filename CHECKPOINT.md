# Validated CPU tuning checkpoint

Commit: `44779d4efae1cf13498b09cdadd185b58052b7c1`

Private server repository: `/home/saeed/glm-tuning-20261011/llama`

Branch: `cpu-tuning`

This full source snapshot includes the validated AMX MoE and batched dispatch fixes, expert and dense NUMA placement, worker affinity and two-group barrier, plus the execution script, numerical test sources, patch against the original source, and benchmark results under `cpu-tuning/`.

Validated real-prompt rates without MTP: English 70.55 prefill / 8.43 decode; Persian 68.66 prefill / 8.59 decode, in tokens per second. Both complete output messages exactly matched the preceding no-MTP implementation. The 10 tok/s general-text decode target has not been met.

No remote Git destination was configured and nothing was pushed. Further profiling experiments are on the `cpu-profile` branch.

## Follow-up experiment, pending real-prompt validation

On branch `cpu-profile`, optional decode node wall-time profiling was added to `ggml-cpu.c`. Aggregated profiling suggested substantial CONCAT and GET_ROWS wall time; this includes synchronization and requires further interpretation.

`ops.cpp` now has an experimental F32/I32 CONCAT fast path for contiguous rows: flattened row partitioning across workers and memcpy of row segments. All 32 independent correctness cases passed (dimensions 0–3, threads 1/3/8/48, contiguous and permuted inputs). No performance result is available yet.

Candidate server PID 4066095, port 8017, log `/home/saeed/glm-tuning-20261011/concat-server.log`. Validated server PID 3932650, port 8016, was left unchanged. Candidate loading was still in progress at the last successful SSH inspection. Then SSH connections timed out and ping had 100% packet loss. Current remote process status is unknown.

Resume by checking connectivity and server health, then running `/home/saeed/glm-tuning-20261011/bench_concat.py`. This uses the same two real prompts with no MTP and writes `real-concat.jsonl` plus `real-concat-results/`. Compare timings and complete outputs against `real-dense-numa.jsonl`. Do not promote the candidate before validation.

## Network restored; CONCAT validated

Commit `c9b8ee9`, tag `validated-real-9p4`, on private branch `cpu-profile`. Only the CONCAT change and its correctness test/results were committed; optional profiling remains uncommitted. Both real prompts produced identical full messages to the previous checkpoint. English: 93.833 prefill / 9.258 decode, 1860 prompt / 352 output tokens. Persian: 89.027 prefill / 9.387 decode, 4397 prompt / 374 output tokens. No MTP and no prompt cache. Candidate server 8017 is now a validated build; server 8016 retains the earlier build.

## GET_ROWS checkpoint: real decode above 13 tok/s

Commit `d079289`, tag `validated-real-13p3`. Independent reference tests: 128 GET_ROWS cases, including million-value recurrent rows. Gate-fusion comparisons: 192 exact matches. Profiling instrumentation is excluded from the checkpoint.

GET_ROWS enabled, no MTP: English 91.179 prefill / 13.354 decode; Persian 86.469 prefill / 13.102 decode, with identical complete messages to the previous checkpoint. Optional gate fusion measured 13.269 / 13.060 decode and did not demonstrate further improvement, so it stays disabled by default (`FUSE_GATE=1` enables it for reproduction).

Startup script now enables GGML_ROWS_CHUNKS. Main server 8016 is being replaced with this validated setting (PID 389016); experiment server 8017 (PID 276636) is scheduled to stop after health verification.

Main server 8016 health verified after deployment. PID 389016 has GGML_ROWS_CHUNKS=1, with gate fusion and profiling absent. A Persian arithmetic smoke request returned the correct answer. Extra experiment server PID 276636 was stopped.
