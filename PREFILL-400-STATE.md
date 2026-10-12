# Prefill tuning: final saved state

Server192.168.20.126, directory/home/saeed/glm-tuning-20261011. Latest reproducible artifacts are in local prefill-final/ and remote PREFILL-RESULTS.md / prefill-final-summary.json.

Basecommit1a1d1ab, validated-real-mtp-18p4. Isolated detached worktree prefill-400; only ggml/src/ggml-cpu/amx/mmq.cpp differs. No new commit or push. Source patch and base Git bundle are saved locally.

Final real English:206.655 prefill/17.676 decode. Persian:206.052/17.508. Fresh baseline:154.538/18.090 and147.252/17.816. Prefill improves33.7%/39.9%; decode is2.3%/1.7% lower in these runs. Exact full outputs and draft counts retained. 468 correctness cases passed. Target400 NOT achieved.

Candidate8017 PID1114185, healthy at final check. Start with run-prefill-validated.sh; batch/ubatch4096,64decode/112prefillthreads,MTPcap1. Stable8016 PID389016 unchanged (older13.3decode checkpoint).

Retained: GGML_Q8_PREFILL_SIGNED and GGML_PREFILL_SCALED_K, both restricted to matrix blocks with >2 rows. Q8signed AMX scratch removes repeated compensation. Q4/Q5scaled low/high byte planes retain exact integer and floating accumulation. Compiler barriers make scratch reads explicit to GCC tile intrinsics.

Rejected/removed from final source: GDN splitting, MoE128 blocks, cyclic scheduling. Source/log backups remain remotely. Batch1024/2048 and56/224threads were tested; results in prefill-sweep.log. Next work should start with current profile of final kernels; original profile showed dense/expert matmul dominant.
