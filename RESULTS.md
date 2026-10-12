# GLM-5.3-Flash CPU tuning — 2026-10-11

Server: `saeed@192.168.20.126`, HPE DL380 Gen11, two Intel Xeon Platinum 8480+ processors (112 physical cores total), 1 TiB RAM, no compute GPU.

## Latest follow-up: real prompts without MTP

The default running configuration is now **without MTP**, with local output-row halves for both experts and eligible packed dense/batched matrices. These comparisons used the same real source-code prompts, temperature 0, seed 42, low reasoning effort, one active request, and zero cached prompt tokens. All completed with `finish_reason=stop`.

| Configuration | Input tokens | Prefill tok/s | Decode tok/s | Output tokens |
|---|---:|---:|---:|---:|
| Previous MTP7, p_min=0.7, English | 1860 | 64.67 | 5.32 | 387 |
| MTP disabled, before dense NUMA fix, English | 1860 | 69.46 | 7.26 | 352 |
| MTP disabled, with dense NUMA fix, English | 1860 | 70.55 | **8.43** | 352 |
| Previous MTP7, p_min=0.7, Persian | 4397 | 64.20 | 5.61 | 429 |
| MTP disabled, before dense NUMA fix, Persian | 4397 | 68.18 | 7.45 | 374 |
| MTP disabled, with dense NUMA fix, Persian | 4397 | 68.66 | **8.59** | 374 |

The dense NUMA fix improved real decode by about 15–16% compared with disabling MTP alone, and about 53–58% compared with the previous MTP configuration. Both full messages exactly matched the no-MTP results before this fix. MTP and non-MTP outputs differed in length, so those comparisons are throughput observations rather than identical-output latency tests. The first no-MTP case included a brief perf sampling window.

A decode perf sample on the actual English request showed 59.24% of CPU-cycle samples in the barrier and 19.71% in the Q8_0 vector matrix kernel. Waiting samples include load imbalance and waiting for other work; they cannot all be treated as removable synchronization overhead. The dense fix assigns packed weight row halves and their compute jobs to the same socket, reducing remote accesses in the Q8/dense path.

The fix passed 160 batched numerical comparisons, including N=64 fallback and N=128 socket splitting, broadcasted weights and permuted activation rows. A wider MoE kernel passed numerical tests but was slower on the real English request (7.92 decode), despite a slightly higher raw benchmark (9.24 versus 9.11). It was rejected. Splitting Q8 integer accumulation into independent chains also passed numerical tests but was slower on the real request and was rejected. The final source restores the validated dense NUMA implementation.

**10 tok/s decode remains unachieved on these real prompts.** The best measured real rates are 8.43–8.59 without MTP. The historical repeated-text MTP benchmark below must not be used as general-text performance. Set `SPEC_TYPE=draft-mtp` to opt into MTP; the default is `none`.

The 50 tok/s prefill and 10 tok/s decode targets were exceeded on the defined repeated-text benchmark. This is not a guarantee for ordinary text: speculative decoding depends on draft acceptance, and short requests have significant fixed overhead. Varied-prompt measurements are reported separately below.

All changes are in the private copy `/home/saeed/glm-tuning-20261011/llama`. The original engine and model files were preserved. Model: `/srv/models/glm53-gguf/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf` (six shards, about 186 GiB). Results apply to this quantized GGUF, not the original BF16 weights.

## Measured performance

One active request, temperature 0, seed 42, 128 generated tokens, exact prompt token counts, `cache_prompt=false`, `cache_n=0`. Throughput comes from server timing fields, with loading excluded. The prompt repeats an English database paragraph; this strongly favors MTP acceptance.

| Engine/configuration | Input tokens | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Original no-OpenMP engine, no repack, t32/tb56 | 512 | 57.60 | 6.20 |
| Original no-OpenMP engine, no repack, t32/tb56 | 2048 | 56.84 | 6.10 |
| AMX MoE, one socket, t24/tb56, ub2048 | 512 | 65.50 | 7.75 |
| AMX MoE, one socket, t24/tb56, ub2048 | 2048 | 54.74 | 7.54 |
| Batched AMX, local expert rows, MTP7, p_min=0 | 512 | 63.45 | 11.16 |
| Batched AMX, local expert rows, MTP7, p_min=0 | 2048 | 67.14 | 11.43 |
| Same optimized engine, MTP7, p_min=0.7 | 512 | 64.82 | 11.29 |
| Same optimized engine, MTP7, p_min=0.7 | 2048 | 67.20 | 11.14 |

The last two measurements were the previous MTP configuration. Draft acceptance was 111/112 and 111/111 respectively. A separate unconditioned 64-token decode sweep without MTP peaked at 8.37 tok/s with 48 threads before the dense NUMA change. This is a different workload and cannot be substituted for the HTTP results.

Four short, varied prompts with MTP7/p_min=0 produced correct completed answers: multiplication (391), Python Euclid GCD, Persian Rayleigh scattering explanation, and capacity planning (7 workers). Their decode rates were 5.27, 8.58, 6.12, and 6.42 tok/s. The first produces very few tokens, making its rate especially sensitive to overhead. Thus, 10 tok/s has not been demonstrated for general text.

Two varied, realistic code-review prompts were then tested with the running MTP7/p_min=0.7 configuration. Both completed with `finish_reason=stop`, and both had zero cached prompt tokens:

| Real prompt | Input/output tokens | Prefill tok/s | Decode tok/s | Accepted/generated drafts |
|---|---:|---:|---:|---:|
| Review actual engine synchronization code in English | 1860 / 387 | 64.67 | 5.32 | 203 / 304 |
| Review a longer source excerpt in Persian | 4397 / 429 | 64.20 | 5.61 | 236 / 304 |

The resulting code reviews were coherent. They were not evaluated as an exhaustive correctness test of every assertion in the generated reviews. These measurements show that the prefill target holds on the tested realistic inputs, while the general-text decode target remains unmet. The repeated-text benchmark must not be presented as general application performance.

## Changes and correctness checks

AMX hardware works. A fresh INT8 tile test requested Linux XTILEDATA permission, initialized a valid tile configuration, zeroed its accumulator, and verified all 256 outputs against 384. The older test incorrectly suggested broken AMX hardware; no BIOS change or reboot was needed.

The patch adds AMX-buffer support for MoE multiplication, fixes allocation for expert/batched matrices, and fixes byte-versus-element offsets and activation strides in batched matrix multiplication. Unsupported packed-weight operations are rejected instead of falling through to raw-weight kernels. FP16 AMX dispatch remains disabled because its rounding did not meet the chosen strict comparison tolerance.

The experimental NUMA path binds each expert's first and second output-row halves to nodes 0 and 1. Threads process the half local to their socket. The running configuration pins each worker to a physical core and uses a two-group barrier with acquire/release synchronization. FTZ/DAZ is enabled for worker threads. Dense small-batch VNNI was tested and slower, so it is disabled. MoE batches of at most four rows use VNNI; larger batches use AMX. MTP uses the model's included NextN head and target verification.

Checks completed:

- 24 MoE comparisons against the host reference for Q4_K, Q5_K, Q6_K, and Q8_0, including token counts 1, 7, and 65 and activation channel counts 1 and 2.
- The same 24 comparisons at K=4096/N=2048 passed, including the small-batch VNNI path; relative errors were below 1e-5.
- 80 batched comparisons passed, including broadcasting, permuted activation strides, and batch row counts 1, 2, 4, 7, and 33.
- Both optimized MTP7/p_min=0 repeated-prompt outputs exactly matched the original baseline's full 128-token output text.
- All four varied-prompt messages were exactly equal between MTP3 and MTP7.

These are focused numerical and functional checks, not a full model-quality evaluation. An early unsafe dispatch experiment generated corrupt text and was rejected; its speed is excluded from the table.

## Run and reproduce

The server listens at `127.0.0.1:8016` on the remote host. It has four slots, 32768 total context (8192 per slot), decode threads 48, prefill threads 56, and microbatch 512. These benchmarks evaluate one active request; concurrent throughput and long-context performance remain unmeasured.

Start on the server:

```bash
bash /home/saeed/glm-tuning-20261011/run-server.sh
```

The script permits overrides through `DECODE_THREADS`, `PREFILL_THREADS`, `DRAFT_MAX`, `DRAFT_PMIN`, `BIND_HOST`, and `PORT`. Stop the existing instance before using its port. No boot service was installed.

Access from another machine:

```bash
ssh -N -L 8016:127.0.0.1:8016 saeed@192.168.20.126
```

The OpenAI-compatible endpoint is then `http://127.0.0.1:8016/v1/chat/completions`. For brief reasoning use `chat_template_kwargs: {"reasoning_effort":"low"}`; this model's template does not use `enable_thinking=false` to disable reasoning.

Build:

```bash
cmake -S /home/saeed/glm-tuning-20261011/llama \
  -B /home/saeed/glm-tuning-20261011/llama/build-batched \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=OFF \
  -DLLAMA_CURL=OFF
cmake --build /home/saeed/glm-tuning-20261011/llama/build-batched \
  -j 24 --target llama-server llama-bench
```

`cpu-tuning.patch` is relative to the original `/home/saeed/llama-smalinin` source. Benchmark scripts and numerical test sources accompany this report. Raw logs/results remain under `/home/saeed/glm-tuning-20261011/`, including `amx-mtp7-pmin-bench.log`, `amx-mtp7-pmin-results`, `quality-mtp7`, `test-small-large.log`, and `test-dense-small.log`.

The running process uses about 375 GiB RSS, including shared mapped source weights and repacked weights. After loading, the machine reported about 795 GiB available RAM and no swap use.

### Parallel CONCAT, validated after network restoration

All 32 CONCAT correctness tests pass. Same real prompts and identical full responses to dense-NUMA baseline, no MTP: English 93.833 prefill / 9.258 decode; Persian 89.027 prefill / 9.387 decode. Commit c9b8ee9, tag validated-real-9p4. Decode targets 10 and 20 tok/s remain unachieved.

### Parallel GET_ROWS, validated on real prompts

GET_ROWS copies of recurrent state contain rows with 1,048,576 values. Split profiling measured approximately 32.407 ms per decode token in GET_ROWS, including 31.816 ms in thread 0 computation rather than graph barrier waiting.

The experimental GGML_ROWS_CHUNKS path distributes contiguous 1024-element column blocks across graph workers. All 128 independent bit-exact reference tests passed, including million-element rows, F32/I32, 1/3/8/48 threads, and permuted tensors.

Real prompts with no MTP: English 1860 prompt / 352 output tokens, 91.179 prefill / 13.354 decode; Persian 4397 / 374, 86.469 prefill / 13.102 decode. Both complete messages are identical to the validated CONCAT baseline. This exceeds the 10 tok/s decode target on both tested real prompts.

A second experiment fuses SCALE→SIGMOID→SCALE with zero biases, preserving the same arithmetic primitives and intermediate rounding. Its 192 tests are bit-exact versus separate kernels. Real performance validation is pending.

Gate fusion results: English 89.453 prefill / 13.269 decode; Persian 85.099 prefill / 13.060 decode. All complete messages match the previous checkpoint. No additional gain demonstrated, therefore fusion remains opt-in and disabled by default. Commit d079289 records both experiments, defaults to parallel GET_ROWS, and excludes profiling instrumentation.
