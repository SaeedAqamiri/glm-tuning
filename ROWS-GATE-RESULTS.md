# Parallel GET_ROWS and gated activation fusion

Decode node profiling separated thread-0 computation from graph-barrier waiting. GET_ROWS used 32.407 ms/token in total, of which 31.816 ms was thread-0 computation. Recurrent-state rows have 1,048,576 values, making single-worker copies a substantial bottleneck.

GGML_ROWS_CHUNKS distributes 1024-element column blocks among graph workers. It supports F32/I32 rows with contiguous columns and preserves the existing path for small rows. All 128 independent bit-exact reference tests passed, including million-value rows, 1/3/8/48 workers, and permuted outer dimensions.

GGML_FUSE_GATE optionally fuses SCALE→SIGMOID→SCALE when biases are zero, tensors are contiguous F32, and intermediates can be elided. It uses the existing arithmetic primitives and preserves intermediate rounding. All 192 bit-exact comparisons versus separate kernels passed, including bias and required-intermediate fallbacks.

| Configuration | Prompt | Prefill tok/s | Decode tok/s | Complete message |
|---|---|---:|---:|---|
| Previous CONCAT checkpoint | English | 93.833 | 9.258 | Reference |
| Previous CONCAT checkpoint | Persian | 89.027 | 9.387 | Reference |
| GET_ROWS only | English | 91.179 | 13.354 | Identical |
| GET_ROWS only | Persian | 86.469 | 13.102 | Identical |
| GET_ROWS + gate fusion | English | 89.453 | 13.269 | Identical |
| GET_ROWS + gate fusion | Persian | 85.099 | 13.060 | Identical |

English: 1860 prompt and 352 output tokens. Persian: 4397 prompt and 374 output tokens. No MTP, no prompt caching, temperature 0, seed 42. Decode 48 and prefill 56 workers. No concurrent benchmark requests. Profiling was disabled in performance measurements; profiler instrumentation is excluded from this source checkpoint.

Gate fusion did not demonstrate an additional performance improvement in these runs and stays disabled by default; use FUSE_GATE=1 with run-server.sh to reproduce the optional experiment. GET_ROWS is enabled by default. Speed targets are surpassed at 50 tok/s prefill and 10 tok/s decode on both tested real prompts; 20 tok/s decode remains unachieved.
