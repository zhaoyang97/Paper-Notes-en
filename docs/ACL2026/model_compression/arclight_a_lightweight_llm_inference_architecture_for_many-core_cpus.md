---
title: >-
  [Paper Note] ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs
description: >-
  [ACL 2026][Model Compression][Many-Core CPU] ArcLight is a lightweight LLM inference framework written from scratch (~10 C++ files) specifically designed for many-core CPUs with multiple NUMA nodes. By integrating NUMA-l…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Many-Core CPU"
  - "NUMA-aware"
  - "Tensor Parallelism"
  - "llama.cpp"
  - "LLM Inference Framework"
date: 2026-05-08
content_hash: c7619c14e8c536ad
---

# ArcLight: A Lightweight LLM Inference Architecture for Many-Core CPUs

**Conference**: ACL 2026  
**arXiv**: [2603.07770](https://arxiv.org/abs/2603.07770)  
**Code**: https://github.com/OpenBMB/ArcLight  
**Area**: Model Compression / Inference Optimization / CPU Deployment  
**Keywords**: Many-Core CPU, NUMA-aware, Tensor Parallelism, llama.cpp, LLM Inference Framework

## TL;DR
ArcLight is a lightweight LLM inference framework written from scratch (~10 C++ files) specifically designed for many-core CPUs with multiple NUMA nodes. By integrating NUMA-local memory pools, multi-view thread pools, cross-NUMA tensor parallelism, and asynchronous subgraph synchronization, it breaks the "remote memory wall." On a 192-core ARM Kunpeng platform, it improves Qwen3-4B Q4_0 decoding throughput by up to 46% compared to llama.cpp.

## Background & Motivation

**Background**: Two mainstream routes exist for LLM inference frameworks: the GPU side (vLLM / TensorRT / SGLang) and the CPU side (represented by llama.cpp). The CPU route is critical for edge deployment, web servers, and high-end network devices, as it allows direct reuse of existing hardware.

**Limitations of Prior Work**: Web servers and network devices commonly use many-core CPUs (48–192 cores divided into 2–4 NUMA nodes). Cross-NUMA memory access latency is approximately 4x higher than local access. On a 192-core machine, local memory bandwidth is ~102 GB/s, while cross-node bandwidth drops to 22–26 GB/s. llama.cpp only exposes `--numa distribute` to evenly split threads across nodes but **does not explicitly bind tensors to NUMA nodes**. The OS allocates physical pages via a first-touch strategy, leading to massive cross-node accesses. Consequently, while computing power scales from 6 to 48 cores, it hits a memory wall when scaling to 192 cores.

**Key Challenge**: Traditional CPU frameworks are designed with "UMA + single thread pool + sequential computation graph," which cannot express the NUMA-aware computation pattern where "multiple NUMA nodes run different subgraphs in parallel, each accessing only local tensors." Modifying llama.cpp would require a major architectural overhaul, from low-level memory allocation to high-level model definitions.

**Goal**: Design a minimal, modular, and NUMA-aware CPU inference framework from scratch to specifically solve the cross-node memory wall.

**Key Insight**: It is observed that in Transformers, $W_q, W_k, W_v, W_{gate}, W_{up}$ are row-splittable and $W_o, W_{down}$ are column-splittable, matching the Megatron-style Tensor Parallelism (TP). By ensuring each "split shard" resides in the local memory of a specific NUMA node, subgraphs can be computed entirely within the node, leaving only Gather/Scatter steps for cross-node communication.

**Core Idea**: Port Megatron-LM tensor parallelism, commonly used across multiple GPUs, down to "CPU multi-NUMA nodes." Combined with NUMA-local memory pools, multi-view thread groups, and asynchronous subgraph synchronization, cross-node memory access is completely eliminated from the main GEMM loop.

## Method

### Overall Architecture
The architecture consists of two layers. **Frontend**: Handles weight loading, model definition, and autoregressive decoding loops; hardware-agnostic. **Backend Inference Engine**: Comprises 5 core modules—Tensor Library (C++ class wrapping header and data), Memory Manager (independent memory pools per NUMA node + activation double-buffering), Thread Manager (multi-view thread groups + global/local barriers), Forward Graph Builder (append-only static graph construction), and Graph Computation Scheduler (sequential operator scheduling within containers). The entire engine consists of ~10 C++ files, reusing llama.cpp's kernels like GEMM and FlashAttention for operators.

### Key Designs

1.  **NUMA Local Memory Pool + Double-buffering Activation**:
    *   **Function**: Strictly binds the physical memory of each tensor to the local memory of a specific NUMA node and reduces peak activation usage via double-buffering.
    *   **Mechanism**: Unlike llama.cpp's single UMA buffer, a separate memory pool is created for each node when NUMA is enabled. Tensors are explicitly bound to the corresponding node's local pool based on TP splitting results. Activation buffering uses double-buffering between alternating layers, where two blocks are shared and rotated.
    *   **Design Motivation**: Under first-touch strategies, the OS allocates pages to the node where the first access occurs. llama.cpp lets the OS decide page placement under distributed threads, resulting in most GEMMs reading weights cross-node. Explicit node binding + pooling makes "which weight is on which node" a framework-level decision; activation double-buffering halves resident memory, suiting edge devices.

2.  **Multi-view Thread Pool + Global/Local Barrier**:
    *   **Function**: Allows a single set of worker threads to dynamically switch between "single-graph mode" (all threads collaborate on one GEMM) and "multi-subgraph mode" (threads split into $n$ groups, each running a TP subgraph independently).
    *   **Mechanism**: Introduces a "thread group" abstraction in the pool. Pools can be reorganized via explicit APIs during initialization or graph execution. Two synchronization mechanisms are introduced: legacy intra-group barriers (within a group) and new global barriers (across all logical groups). The Scatter operator splits the pool into $n$ groups and creates view tensors for each subgraph; the Gather operator collects and sums subgraph outputs before restoring the pool to a single group.
    *   **Design Motivation**: A single thread pool can only run one operator at a time, failing to express parallel subgraphs after TP. The group abstraction allows the same physical pool to switch between "collaborative" and "independent" execution modes based on whether the operator is in a TP region, avoiding the overhead of maintaining multiple pools.

3.  **Cross-NUMA Tensor Parallelism + Asynchronous Subgraph Synchronization**:
    *   **Function**: Splits GEMM sequences within each Transformer block into TP shards equal to the number of NUMA nodes. Gather operations are only needed at the Attention and MLP exits, with node-local memory access held throughout the rest.
    *   **Mechanism**: In MLP, $Y=\sigma(AX), Z=BY$ is split into $[Y_1,Y_2]=[\sigma(A_1 X),\sigma(A_2 X)]$ and $[Z_1,Z_2]=[B_1 Y_1, B_2 Y_2]$, with $Z=Z_1+Z_2$ (reduced only during Gather). All TP tensors ($A_i, B_i, W_q^i, W_k^i, W_v^i$, etc.) are split and reside in their respective node-local pools. Synchronization uses "Sync B": subgraphs are asynchronous by default, with global barriers applied only at Scatter entries and Gather exits, rather than after every GEMM.
    *   **Design Motivation**: Standard TP using "Sync A" (cross-group barrier after every GEMM) causes threads to wait for the slowest group, leading to significant idle time. Sync B allows each NUMA group to progress at its own pace, synchronizing only at merge points, which adds ~5 token/s in practice. This is a critical adaptation when translating GPU TP logic to CPU multi-NUMA environments.

### Loss & Training
This is a pure inference framework with no training involved. The operator layer relies on ARM NEON (SIMD) and i8mm (int8 matrix multiplication), but ArcLight itself is hardware-agnostic and can be ported to x86 by rewriting the ops.

## Key Experimental Results

### Main Results (Qwen3-4B Q4_0, prompt=15 / decode=256, HUAWEI Kunpeng-920 ARM, 4 NUMA × 48 cores + 6×DDR4/node)

**Single NUMA Node (Intra-node Scalability)**

| Threads | llama.cpp (tok/s) | ArcLight (tok/s) | Gain |
|---|---|---|---|
| 6 | ~12 | ~13 | +8% |
| 24 | ~28 | ~31 | +11% |
| 48 | ~32 | ~36 | +12% |

ArcLight is slightly faster, attributed to forced local node allocation.

**Multi-NUMA Node (Core Scenario)**

| Configuration | llama.cpp | ArcLight (Sync A) | ArcLight (Sync B) | vs. llama.cpp |
|---|---|---|---|---|
| 2 NUMA × 48 cores = 96 | ~38 tok/s | ~50 tok/s | ~55 tok/s | **+45%** |
| 4 NUMA × 48 cores = 192 | ~42 tok/s | ~57 tok/s | ~62 tok/s | **+46%** |

After the cross-node memory wall is addressed via TP, llama.cpp stops scaling at 192 cores, while ArcLight continues to show linear growth. Sync B (asynchronous) adds ~5 tok/s over Sync A.

### Cross-NUMA Access Benchmark (4-node machine, 48 ARM cores per node + 6×DDR4)

| Core Location \ Memory Location | node 0 | node 1 | node 2 | node 3 |
|---|---|---|---|---|
| node 0 | **102** | 26 | 24 | 23 |
| node 1 | 26 | **103** | 23 | 22 |
| node 2 | 24 | 23 | **103** | 26 |
| node 3 | 23 | 22 | 26 | **101** |

Local access is ~4x faster than cross-node (GB/s), confirming the "memory wall" is the true bottleneck for many-core LLM inference.

### Key Findings
- **Cross-NUMA TP is essential**: Scaling a single NUMA node to 48 cores nears the memory bandwidth limit; further throughput increases require cross-node TP to avoid the 4x remote access penalty.
- **Sync B (Asynchronous Subgraphs) is consistently superior**: The ~5 tok/s improvement proves that strict global synchronization wastes ~10% of thread time in multi-subgraph scenarios.
- **ArcLight's advantage in prefill is smaller than in decode**: Prefill is compute-bound, yielding lower gains from memory optimization; decode is memory-bound, perfectly aligning with NUMA-local optimizations.
- **Decode advantage holds for long prompts (300 tokens)**: While long KV caches increase memory pressure, the NUMA-aware design still outperforms.
- **Lightweight code is a product advantage**: ~10 C++ files vs. the massive llama.cpp codebase lowers the barrier for researchers to modify or add new models.

## Highlights & Insights
- **Porting GPU multi-card TP to CPU multi-NUMA is the core insight**: While TP was previously thought to be relevant only for multi-GPU/multi-machine setups, this work proves it is a powerful tool to "break the wall" on multi-NUMA CPU nodes. This logic extends to multi-chiplet GPUs (e.g., H100/B100 with NVLink C2C).
- **Multi-view thread pool design is clever**: Using one pool for both "collaboration" and "grouping" avoids multi-pool switching overhead and provides a general foundation for dynamic operator parallelism.
- **Sync B is an undervalued engineering detail**: Textbook TP defaults to Sync A, but high CPU thread-switching costs make Sync B necessary, as it defers synchronization to the last possible moment.
- **Separation of framework and operator**: ArcLight does not write its own operator library, instead reusing llama.cpp's GEMM/FlashAttention. This focuses effort on memory/thread/graph innovations, a smart choice to avoid the "red ocean" of kernel optimization.
- **Code simplicity is a research contribution**: By making the framework "readable and hackable," researchers can quickly validate new models and scheduling strategies, providing value to the academic community beyond just performance numbers.

## Limitations & Future Work
- Tested only on the ARM Kunpeng platform; the x86 platform requires rewriting NEON sections (e.g., adapting to AVX-512 VNNI for i8mm equivalents).
- The authors admit Scatter/Gather operators are "preliminary," suggesting room for further optimization in fine-grained operator parallelism.
- Only evaluated Qwen3-4B (Q4_0 quantization); scalability for larger models (e.g., 70B+ Q4_0, which might be quite slow on 192-core CPUs) is unknown.
- Asynchronous execution may suffer from "fast subgraphs waiting for slow subgraphs" in long-tail cases (e.g., sudden node slowdowns), which is not discussed.
- No direct cost/performance comparison with GPU inference was provided, providing little guidance on when to choose CPU inference.

## Related Work & Insights
- **vs. llama.cpp**: Inherits its operator library but replaces the memory, thread, and graph modules and adds cross-NUMA TP; directly compared in experiments.
- **vs. vLLM / SGLang / TensorRT**: GPU frameworks do not face the NUMA memory wall; PagedAttention solves GPU KV cache fragmentation, which is orthogonal to this target scenario.
- **vs. ONNX Runtime CPU**: A general-purpose framework without specific optimizations for many-core NUMA.
- **vs. Megatron-LM TP**: The source of the core idea; this is a "lite" CPU version (no NCCL, only NUMA local memory).
- **vs. Quantization/Pruning (OmniQuant / OneBit / Wanda / CRVQ)**: Those reduce memory access volume; this work reduces memory access latency. The two are orthogonal and stackable.

## Rating
- Novelty: ⭐⭐⭐⭐ Porting TP to CPU multi-NUMA is a clear engineering breakthrough; multi-view pools and Sync B are specific, novel details.
- Experimental Thoroughness: ⭐⭐⭐ Covers decode/prefill, single/multi-NUMA, and Sync A/B on a 192-core machine; however, it only evaluates one model, one quantization level, and one hardware type.
- Writing Quality: ⭐⭐⭐⭐ 7 core figures and tables clearly communicate the memory wall, TP splitting, double-buffering, and thread views.
- Value: ⭐⭐⭐⭐ Open-source, minimal code, and directly applicable to web servers and network devices for edge inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](../../ICLR2026/model_compression/paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)

</div>

<!-- RELATED:END -->
