---
title: >-
  [Paper Note] Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds
description: >-
  [ICML 2026][Model Compression][Memory-efficient inference] This paper presents the design of the "DNN Pipeline Scheduling Subsystem" within the CROWDio framework. Without modifying the model itself (no pruning…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Memory-efficient inference"
  - "Model partitioning"
  - "Edge ML"
  - "ONNX"
  - "Android"
  - "Pipelined scheduling"
date: 2026-05-08
content_hash: 5b48cc60b5af4a03
---

# Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds

**Conference**: ICML 2026  
**arXiv**: [2605.20723](https://arxiv.org/abs/2605.20723)  
**Code**: Undisclosed (DNN scheduling subsystem of the CROWDio framework)  
**Area**: Model Compression / Edge Deployment / Mobile Crowdsourced Inference  
**Keywords**: Memory-efficient inference, Model partitioning, Edge ML, ONNX, Android, Pipelined scheduling  

## TL;DR
This paper presents the design of the "DNN Pipeline Scheduling Subsystem" within the CROWDio framework. Without modifying the model itself (no pruning, quantization, or distillation), a complete ONNX model is partitioned into multiple segments by layer and distributed across multiple Android devices with 3.3–7.4 GB RAM for pipelined inference. Five mechanisms—**JIT lazy loading, single partition residency constraint, 4-tier affinity scheduling, zlib-compressed tensor transmission, and streaming 1:1 dependencies**—are employed to compress the peak RSS of each device to $43\pm 2$ MB and achieve a batch latency 34% faster than traditional barrier synchronization.

## Background & Motivation

**Background**: For DNN inference on resource-constrained devices, the mainstream approach is **model reduction**—quantization, pruning, knowledge distillation, and low-rank decomposition—to minimize the "per-parameter cost." This assumes the model is still fully hosted by a single device. A complementary approach is **deployment-aware partitioning**: instead of reducing the model, its layers are split across multiple memory-constrained devices, allowing the team's total memory to "pool" into a complete model, where no single device exceeds its RAM limit.

**Limitations of Prior Work**:
- Commercial Android phones often have only 3.3–7.4 GB RAM, but a "medium" Transformer like DistilBERT requires 2–4 GB for the ONNX session alone, leaving almost no space for the OS and background processes.
- Quantization, pruning, and distillation all require model modification, leading to long deployment chains and sensitivity to performance. Partitioning systems like SPINN and Neurosurgeon assume a "one mobile + one reliable cloud" split, failing to handle **volunteer crowdsourcing** scenarios with more than two segments, frequent device churn, and heterogeneous RAM.
- Existing "intra-device layer offloading" (e.g., Melon) only solves memory reuse within one device; it cannot distribute the load across multiple devices. GPipe and PipeDream are designed for GPU clusters and rely on stable networks and homogeneous hardware.

**Key Challenge**: In **heterogeneous, unstable, and extremely memory-constrained** Android crowds, performing full Transformer inference requires satisfying three conditions simultaneously: (1) no single device's peak RSS can exceed its RAM budget; (2) network transmission of pipeline segments must not saturate Wi-Fi channels; (3) the scheduling system must redistribute tasks without interrupting in-flight jobs when devices disconnect. If any of these are violated, the entire inference process fails.

**Goal**: Without modifying model weights, provide an engineering system capable of running DistilBERT (≈67M parameters, SST-2) on a cluster of Android phones with 3.3 GB RAM, and quantify its memory, latency, power consumption, and compression ratio.

**Key Insight**: Treat "memory pressure" as a first-class citizen for scheduling. Since each device cannot fit the entire model, the system **limits the number of resident partitions to one** and amortizes the "on-demand loading" latency cost using JIT and affinity scheduling.

**Core Idea**: **Single partition residency + JIT lazy loading** serves as the hard constraint for memory safety. The other four components (affinity scheduling, compressed transmission, streaming dependencies, and fault recovery) are designed to reduce latency and energy consumption to acceptable levels under this constraint.

## Method

### Overall Architecture

The CROWDio three-layer architecture runs over a persistent WebSocket channel:

- **Developer SDK**: Developers submit ordered ONNX fragments. The SDK automatically generates a linear DAG, performs topological validation, adds SHA-256 integrity checks, and packages them using base64.
- **Foreman (Scheduler)**: Instantiates $N$ inputs $\times S$ stages into $N\times S$ task records, maintains the dependency graph, schedules tasks based on 4-tier affinity, and handles failure recovery.
- **Android Worker**: Executes ONNX inference under the single partition residency constraint and reports CPU/RAM/battery/RTT/temperature telemetry every 30 seconds.

The reference workload is DistilBERT-SST2, split into three segments at layer 3: `cell_a` (Embedding + Layers 0–2, the smallest fragment), `cell_b` (Layers 3–5, the largest fragment), and `cell_c` (Pre-classifier + Classifier).

### Key Designs

1. **JIT Lazy Loading + Single Partition Residency (Hard Constraints for Memory Safety)**:
    - **Function**: Strictly limits the peak RSS of any Android device to the memory footprint of a single fragment, regardless of pipeline depth or fragment size.
    - **Mechanism**: Since `cell_a` is the smallest, it is **eagerly broadcast** to all workers to ensure all devices can immediately accept Stage 0 tasks. In contrast, `cell_b` and `cell_c` are only sent as **loading commands when upstream tasks are completed** (JIT). Combined with the "one active ONNX session per worker" constraint, this ensures peak RSS = single fragment size. End-to-end correctness is verified by comparing chained ONNX Runtime outputs vs. global PyTorch outputs, with max-abs error $\approx 0$.
    - **Design Motivation**: A complete DistilBERT ONNX session requires 2–4 GB, which would starve the OS on a 3.3 GB RAM phone. Forced residency and JIT are prerequisites for **deployment feasibility** on such devices.

2. **Streaming 1:1 Dependency Model (Eliminating Batch Sync Barriers)**:
    - **Function**: Decouples synchronization across stages from "batch barriers" to "per-input" units, allowing each input to flow independently and eliminating the "straggler effect."
    - **Mechanism**: After Foreman instantiates $N\times S$ tasks, it assigns a 1:1 dependency for each Stage-$k$ task on the Stage-$(k-1)$ task of the **same input**. Barrier mode would wait for all Stage-$(k-1)$ tasks to complete before releasing Stage-$k$ (1:$N$ dependency), which is inevitably slowed down by the slowest node in heterogeneous hardware. For $N=5, S=3$, streaming generates 15 independent tasks without inter-stage barriers—effectively bringing PipeDream's micro-batching to heterogeneous mobile scenarios.
    - **Design Motivation**: Volunteer phones vary significantly (SoC, thermal throttling, Wi-Fi signal). Batch barriers penalize performance heavily in such environments. 1:1 dependencies allow fast devices to pull downstream tasks immediately, making the overall latency determined by the bottleneck input rather than the bottleneck batch.

3. **4-Tier Affinity Scheduler (Amortizing Cold Start Costs under Memory Constraints)**:
    - **Function**: Maximizes the reuse of "already resident fragments" and minimizes offloading-reloading cold-start latency while maintaining the single residency constraint.
    - **Mechanism**: The scheduler wraps any base algorithm (FIFO/EDAS/ARAS/MABAC) with two gates: a **Model Gate** that only assigns tasks to workers whose "required fragment is already in session memory," and **Affinity Ranking** that sorts candidates into 4 tiers (Tier 1: Resident $\rightarrow$ Tier 2: Disk cached $\rightarrow$ Tier 3: Idle/No residency $\rightarrow$ Tier 4: Explicit offload and reload). Tier 4 is a fallback that triggers an explicit `UNLOAD_MODEL` $\rightarrow$ `LOAD_MODEL` sequence, incurring high latency but preserving memory invariants. Within the same tier, workers are ranked by Shannon entropy-weighted telemetry.
    - **Design Motivation**: Once memory is stabilized by JIT, "model loading" becomes the next bottleneck—cold starts take $48\pm 4$ s, while hot starts take only $6\pm 1$ s. Affinity scheduling makes Tier-1 hits the norm, reducing the amortized cold start cost to nearly negligible levels.

**Supplementary Mechanisms**: **Compressed Tensor Transmission** uses self-describing JSON for symmetric Python $\leftrightarrow$ Kotlin serialization, achieving a $62\pm 4\%$ compression ratio for $[1, 768]$ FP32 tensors. When payloads exceed $\tau_{\text{ws}}=1$ MB, they are written to a shared filesystem to avoid WebSocket blocking.

### Loss & Training
This is a **system paper and does not involve training**: pre-trained DistilBERT-SST2 weights are used directly. All tunable parameters are at the system level (partition points, $\tau_{\text{ws}}$, affinity weights).

## Key Experimental Results

### Main Results
5 heterogeneous Android phones (RAM 3.3–6 GB, Android 13–14, ONNX Runtime 1.17) ran the DistilBERT-SST2 pipeline ($N=5, S=3$) over local Wi-Fi.

| Metric | Streaming | Barrier | Remarks |
|---|---|---|---|
| Peak RSS per Device | $43\pm 2$ MB | $43\pm 2$ MB | Single residency constraint effective |
| End-to-end Batch Latency | $18.4\pm 1.1$ s | $27.9\pm 2.3$ s | Streaming **34% faster** |
| Cold Start Loading | $48\pm 4$ s | $48\pm 4$ s | Includes fragment download + ONNX init |
| Hot Start (Tier-2 hit) | $6\pm 1$ s | $6\pm 1$ s | Payoff of affinity scheduling |
| Battery Consumption | $50\pm 3$ mAh/run| $53\pm 4$ mAh/run| Streaming slightly more efficient |
| Tensor Compression Ratio| $62\pm 4\%$ | – | $[1, 768]$ FP32 tensor: 3072 $\rightarrow$ ≈1168 bytes |

### Ablation Study

| Dimension | Configuration | Conclusion |
|---|---|---|
| Streaming vs. Barrier | $N=5, S=3$ DistilBERT | Streaming batch latency -34% |
| Single Residency | Enabled vs. Disabled | Enabled: RSS locked at 43 MB; Disabled: Un-deployable (2–4 GB) |
| JIT Lazy Loading | Enabled vs. Eager Broadcast | JIT prevents large `cell_b` from idling in memory |
| Affinity Tier-1 vs. Tier-4| Hit Rate | Tier-1/2 hits reduce loading latency from 48s to 6s |
| Compression vs. Raw | $[1,768]$ FP32 tensor | 62% compression; avoid Wi-Fi blocking via filesystem |

### Key Findings
- **Single partition residency is the threshold for feasibility**: Without it, 3.3 GB RAM devices cannot host DistilBERT. With it, the RSS remains stable at 43 MB.
- **Streaming 1:1 dependencies provide a "free" 34% speedup**: It eliminates straggler effects in heterogeneous clusters without model changes or extra memory.
- **Affinity scheduling reduces amortized cold start from 48s to 6s**: This confirms that "which device hosts which model" is the core problem of edge deployment.

## Highlights & Insights
- **"Deployment-Aware Partitioning" as the Fourth Pillar of ML Compression**: The paper positions partitioning alongside pruning, quantization, and distillation, emphasizing that they are **orthogonal and stackable**.
- **Engineering Aesthetics of "Hard Constraints + Soft Optimizations"**: Single residency and JIT are non-negotiable hard constraints; affinity scheduling and streaming are soft optimizations. This ensures the system remains functional in worst-case scenarios and fast in best-case scenarios.
- **Entropy-Weighted Two-Stage Dispatch**: Using Shannon entropy to dynamically weight telemetry allows metrics with high variance to dominate the ranking automatically, making the scheduler robust for heterogeneous clusters.

## Limitations & Future Work
- **Single residency is conservative for high-RAM devices**: Phones with >6 GB RAM could host two partitions simultaneously. "Memory-adaptive multi-partition residency" is a natural extension.
- **48s cold start remains painful**: While Tier-2 reduces this, the initial download and ONNX initialization costs are unavoidable without aggressive prefetching.
- **Linear Topologies Only**: Branching/merging structures (e.g., MoE, ResNet skip connections) require explicit manual DAG definition.
- **Evaluation limited to DistilBERT**: It remains to be seen if larger models like LLaMA-class can be partitioned small enough to fit in 3.3 GB RAM devices without the single fragment itself exceeding the limit.

## Related Work & Insights
- **vs. Melon (MobiSys 2022)**: Melon performs hierarchical offloading within a single device. CROWDio's JIT extends this to heterogeneous clusters.
- **vs. SPINN / Neurosurgeon**: These focus on "Mobile + Cloud" splits. CROWDio targets cloudless volunteer crowds requiring dynamic scheduling for $S \ge 3$.
- **vs. GPipe / PipeDream**: While same-origin in pipelining, CROWDio's streaming 1:1 dependency is the mobile-equivalent implementation of micro-batching for heterogeneous environments.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ICML 2026\] EpiCache: Episodic KV Cache Management for Long-Term Conversation on Resource-Constrained Environments](epicache_episodic_kv_cache_management_for_long-term_conversation_on_resource-con.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)
- [\[ICML 2026\] Towards Resource-Efficient LLMs: End-to-End Energy Accounting of Distillation Pipelines](towards_resource-efficient_llms_end-to-end_energy_accounting_of_distillation_pip.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](../../NeurIPS2025/model_compression/keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)

</div>

<!-- RELATED:END -->
