# Experiments toward 100 prefill / 20 decode

Baseline QKV, 48 decode / 96 prefill threads: English 103.230842 / 13.161685 tok/s; Persian 101.403606 / 13.012374. Both exact baseline messages. Checkpoint f4e9512 (validated-prefill-100).

Q4 expanded weights, same threads and prompts: English 98.403275 / 12.693671; Persian 96.550221 / 12.490098. Both complete messages identical. Rejected: additional memory traffic outweighs unpack removal. Golden tests: 80 cases, 7,700,480 bytes exactly equal; MoE 24 cases passed. Fixed packed index arithmetic to int64_t after full-model tensors exceeded 2 GiB. Expanded layout stays compile-time opt-in.

Q4/Q5 independent integer dot-product chains, same 48/96 threads: English 106.348502 / 13.484393; Persian 103.205230 / 13.366921 tok/s. Both complete messages exactly equal reference. Golden 80 cases bit-exact; large MoE 24 cases pass.

Single-row AMX tiles: golden 80 cases bit-exact, QKV 240 cases bit-exact, MoE 24 pass. Direct dense K4096/N12288/48-thread microbenchmark: Q8 4250.51 -> 4172.86 us; Q4 4228.43 -> 4615.29 us; Q5 4035.00 -> 4169.12 us. This direct test included thread-pool creation, so its absolute times and comparisons require a repeat with a persistent pool before drawing performance conclusions. Fused MoE gate/up pair: 96 small tests and 24 large tests bit-exact; real English 105.199389 / 13.645907 and Persian 102.651902 / 13.436134, both complete messages exactly equal. Small valid decode improvement. Shared activation, small AMX rows, four Q8 integer chains and bounded MTP are pending experiments.

20 tok/s single-request decode has NOT been achieved.

Persistent-pool microbench K4096/N12288/48 threads (us): baseline Q8 57.20, Q4 57.92, Q5 69.84; interleaved Q8 54.20, Q4 56.09, Q5 65.24. Single-row AMX Q8 86.36, Q4 108.15, Q5 108.90 in earlier persistent-pool trial: rejected for normal decode. New kernels preserve all golden bytes (80 cases), QKV 240 cases and expert-pair 96 cases.

MTP cap=1, fixed/adaptive-off, p_min=0, small AMX rows, shared decode, 48 decode/96 prefill threads but verification uses batch pool96: English 98.427728 / 6.838437 tok/s, 158/208 drafts accepted (75.96%); Persian 97.087025 / 6.750048, 148/191 accepted (77.49%). Messages differ from no-MTP baseline, output lengths366/339. Rejected configuration. New experimental small-batch routing uses decode pool48 for <=8 tokens, prefill112, interleaved kernels; HTTP validation and model speed pending.

Validated MTP persistent pools + parallel strided state CPY + small chains, t48/tb112, global draft_max1/p_min0/adaptiveoff: English105.269485 /15.092519 (366 output,158/208 drafts accepted); Persian102.404155 /14.919439 (339 output,148/191 accepted). Both messages exactly equal prior MTP runs. Model weights unchanged. COPY128 independent byte-reference cases, small chains60 exact cases; prior kernel golden80/QKV240/MoE96+24 pass. Shared pool activation positively logged. CPU-only, sequential graph execution; draft context uses target-owned pools and is destroyed before target. Goal decode20 still unachieved. New two-row VNNI kernels and threshold8192 are next, pending.

## Exact two-row VNNI checkpoint
Q8 dense/QKV and Q4/Q5/Q8 MoE share weight loads across two activation rows. Floating accumulation order preserved. SMALL_CHAIN limit configurable, default4096, validated8192. Golden80 byteexact, QKV240 exact,192 smallchain cases everyintermediate exact, MoE40 cases small andlarge exact,48 traces sameexpert test andactualthreads3/48 passes. BuildnativeAMX successful. FullMTP max1 pmin0 adaptiveoff sharedpool64/prefill112: English105.5843746 prefill/16.3669666 decode,366 tokens,158/208 drafts; Persian103.7455836/16.2347542,339 tokens,148/191 drafts. Both fullmessages exact previous sharedpool48 MTP. Target20 not achieved. This comparison includes thread48→64 and chainlimit4096→8192 changes; kernel-only Q8 M2micro94.18us off vs59.67/62.25us on. DenseQ4/Q5micro doesnot exerciseMoErows2 path.
