# Synthetic local-data development evaluation

Transport: **axl**. Seed 17; four local CPU workers; eight rounds of 50 Adam steps.
All training nodes are separate NodeRuntime instances in one Python process, with thread-offloaded CPU work. AXL mode adds four real AXL subprocesses on that same host.

This is synthetic development evidence. It does not replace the M2 GPU/WAN matrix, NCCL reference, or identity ablation. Every profile uses the same frozen settings and explicit balanced common held-out set. The reference independently implements the outer optimizer but shares the generic inner trainer and workload; its pair exchange is synchronous in-process.

Frozen experiment SHA-256: `d3c68077ae9f7d82439844cea4da0450a247997bc88127f945b5681270b9b4d6`.

Predeclared gates: common held-out mean accuracy ≥ 0.80; worst-node accuracy ≥ 0.60 for each codec; compressed mean at most 0.10 below identity; identity/reference parity at every retained outer step within atol=rtol=1e-6. No M2 compression-ratio or residual threshold is reused.

| Profile | Identity mean/min | Compressed mean/min | Reference mean/min | Local-only mean/min | Parity | Decision |
|---|---:|---:|---:|---:|---|---|
| iid | 0.994/0.994 | 0.994/0.994 | 0.994/0.994 | 0.994/0.992 | True | pass |
| label-skew-moderate | 0.996/0.996 | 0.996/0.996 | 0.996/0.996 | 0.994/0.992 | True | pass |
| label-skew-severe | 0.995/0.994 | 0.994/0.992 | 0.995/0.994 | 0.367/0.252 | True | pass |
| quantity-skew | 0.994/0.994 | 0.994/0.994 | 0.994/0.994 | 0.993/0.988 | True | pass |
| feature-skew | 0.994/0.994 | 0.994/0.994 | 0.994/0.994 | 0.995/0.994 | True | pass |

| Profile | Identity round wire bytes | Compressed round wire bytes | Identity/compressed | Compressed max residual L2 |
|---|---:|---:|---:|---:|
| iid | 500960 | 533280 | 0.939 | 0.810 |
| label-skew-moderate | 512736 | 545056 | 0.941 | 0.767 |
| label-skew-severe | 511264 | 543584 | 0.941 | 1.054 |
| quantity-skew | 508320 | 540640 | 0.940 | 1.124 |
| feature-skew | 507584 | 539904 | 0.940 | 0.815 |

Per-node common/local loss, accuracy and per-class support/scores are retained in [results.json](results.json), along with every runtime round's metrics, residual norms, retries, sender wire accounting and cross-node weight dispersion. The full logs and durable run stores remain under `runs/`; bounded post-COMMITTED slow-weight snapshots remain under each node's `analysis/`. Failed criteria are retained and require a new, separately declared experiment for ablations; they are not adjusted after seeing results.

Wire accounting counts full serialized Dromeus envelopes accepted by the transport, including retries and protocol overhead, with telemetry and formation/control separated. It excludes AXL/TCP framing. These tiny models can cost more bytes when compressed because bitmap/scales and metadata dominate; no speed or 5× compression claim follows. Runtime timing includes same-host CPU contention and durability; reference/local-only timing is sequential and is not a network speed comparison.

Every method receives 400 optimizer steps per node with batch size 16, retaining partial batches and repeatedly sampling local datasets as needed. Actual sample presentations are recorded. The objective gives each node equal influence despite unequal data counts. Local-only learning has no communication; its local fit may coexist with poor common-held-out accuracy.

The initial checkpoint, data-plan hashes, actual sorted public keys, configuration, source-file hashes, complete source snapshot and host runtime are retained. The container image digest is explicitly null because these are host runs. In-memory results prove the production protocol's local integration; real local AXL proves the AXL path on one host. Neither establishes privacy, Byzantine tolerance, WAN resilience, or general non-IID convergence.
