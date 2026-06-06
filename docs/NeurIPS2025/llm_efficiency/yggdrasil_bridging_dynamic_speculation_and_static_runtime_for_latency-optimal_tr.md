---
title: >-
  [Paper Note] Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding
description: >-
  [NeurIPS 2025][LLM Efficiency][speculative decoding] This paper proposes Yggdrasil, a latency-optimal speculative decoding system that achieves compiler-friendly dynamic drafting via the Equal-Growth Tree (EGT) structure…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "speculative decoding"
  - "tree-structured draft"
  - "compiler optimization"
  - "latency optimization"
  - "LLM inference acceleration"
date: 2026-05-08
content_hash: bdffc089f2b90eac
---

# Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2512.23858](https://arxiv.org/abs/2512.23858)  
**Code**: None  
**Area**: LLM Efficiency / Speculative Decoding
**Keywords**: speculative decoding, tree-structured draft, compiler optimization, latency optimization, LLM inference acceleration

## TL;DR

This paper proposes Yggdrasil, a latency-optimal speculative decoding system that achieves compiler-friendly dynamic drafting via the Equal-Growth Tree (EGT) structure, replaces the conventional AAL metric with a latency-aware optimization objective, and reduces CPU-GPU coordination overhead through a stage-based scheduling runtime, achieving up to 3.98× end-to-end speedup on A100/A40 GPUs.

## Background & Motivation

**Background**: Speculative decoding accelerates LLM inference by having a small draft model generate candidate token sequences that a larger verifier model validates in parallel, serving as a lossless acceleration method that preserves the output distribution. Tree-based speculative decoding further expands the search space by verifying multiple candidate paths in a single forward pass.

**Limitations of Prior Work**: Existing systems exhibit a fundamental mismatch between dynamic drafting and static runtimes. (1) Dynamic drafting algorithms (e.g., DISCO) adaptively adjust tree structures and operator shapes at each step to maximize token acceptance rates, but this conflicts with the static computation graph assumptions required by deep learning compilers (e.g., TorchInductor), precluding the 2–3× speedups from graph fusion and kernel tuning. (2) Existing methods typically optimize Average Acceptance Length (AAL), yet a higher AAL does not guarantee faster end-to-end throughput—as the number of verified tokens increases, verification latency grows nonlinearly, and actual per-token speedup may plateau or even decline.

**Key Challenge**: A fundamental trade-off exists between the high AAL enabled by dynamic tree structures and the static shapes required by compilers; no prior framework simultaneously achieves both.

**Goal**: (1) Design a tree structure that adapts dynamically to context while maintaining static operator shapes; (2) Use actual latency rather than AAL as the optimization objective; (3) Eliminate GPU idle bubbles caused by CPU-side logic in speculative decoding.

**Key Insight**: The paper frames speculative decoding as an algorithm-system co-design problem and optimizes it simultaneously at three levels.

**Core Idea**: Employ the Equal-Growth Tree (EGT) to reconcile dynamic adaptability with compiler compatibility, optimizing latency across the full stack from algorithm to runtime.

## Method

### Overall Architecture

Yggdrasil performs co-optimization at three levels: (1) **Algorithm level**: EGT tree structure with a latency-aware objective; (2) **Compilation level**: leveraging EGT's static shape properties to enable TorchInductor compilation; (3) **Runtime level**: stage-based scheduling to eliminate CPU-GPU coordination overhead. The system accepts any unmodified LLM drafter-verifier pair as input and produces a latency-optimal speculative decoding pipeline.

### Key Designs

1. **Latency-Aware Optimization Objective**

    - **Function**: Replaces the conventional AAL metric as the tree structure selection criterion, directly maximizing wall-clock speedup.
    - **Mechanism**: The speedup is formulated as $\frac{AAL(W_d, D_d, W_v) \cdot T_{verifier}(1)}{\sum_{D_d} T_{drafter}(W_d) + T_{verifier}(W_v)}$, where $W_d$, $D_d$, and $W_v$ denote draft width, draft depth, and verification width, respectively. Hardware profiling is used to obtain latency curves for $T_{drafter}$ and $T_{verifier}$, enabling latency-optimal tree structure selection at runtime.
    - **Design Motivation**: AAL ignores the nonlinear growth of verification latency with respect to token count. Experiments show that AAL continues to increase in high-token-count regimes even as the actual speedup saturates or decreases. The latency-aware objective yields an additional 8% performance gain.

2. **Equal-Growth Tree (EGT)**

    - **Function**: Achieves context-adaptive tree structures while maintaining static operator shapes compatible with compilers.
    - **Mechanism**: EGT decomposes tree construction into three greedy sub-decisions: (a) **Depth prediction**: a lightweight MLP predictor infers the optimal draft depth $D_d$ from the target model's last-token embedding; graphs for all depths are pre-compiled to eliminate conditional branching. (b) **Width selection**: given the predicted depth, the width $W_d$ maximizing speedup is selected; equal-width growth at each step guarantees fixed operator shapes. (c) **Verification pruning**: after drafting, dynamic programming extracts the subtree that maximizes speedup as the verification set. Since the number of new leaves added per step is fixed at $W_d$, operator shapes are determined at compile time.
    - **Design Motivation**: Fully dynamic methods such as DISCO achieve high AAL but are incompatible with compilation, while static methods such as Sequoia are compiler-friendly but apply a single tree structure to all inputs. EGT resolves this tension through the principle of "statically shaped, structurally dynamic."

3. **Stage-Based Scheduling Runtime**

    - **Function**: Reduces CPU-GPU bubbles in speculative decoding through ahead-of-time execution and overlapped scheduling.
    - **Mechanism**: (a) **Ahead-of-Time Tail Draft**: the full candidate sequence is drafted before acceptance results are available and reused directly upon acceptance. (b) **Ahead-of-Time Head Draft**: the head draft is initiated after the bonus draft of the previous iteration, overlapping with the acceptance phase. Profile-guided search then identifies the optimal execution plan among all feasible stage-overlap configurations.
    - **Design Motivation**: CPU-side logic (acceptance checking, token management) in speculative decoding causes GPU idle time. By breaking data dependencies and pre-executing downstream stages, CPU logic is removed from the critical path.

### System Implementation

Yggdrasil is built on PyTorch 2.0 and TorchInductor. The core abstraction is a `TokenTree` class that encapsulates EGT semantics. Graph optimization and kernel tuning for the drafter and verifier are performed at compile time, while KV cache management, attention masking, and leaf position tracking are handled at runtime through the `TokenTree` interface. No modifications to the original LLM architecture are required.

## Key Experimental Results

### Main Results (End-to-End Per-Token Latency)

| Hardware | Model | Yggdrasil Speedup | vs SpecInfer | vs Sequoia | vs vLLM-Spec |
|----------|-------|-------------------|-------------|------------|-------------|
| A100 | Llama-2-7B + 68M | **3.98×** | +3.37× | +1.54× | +1.32× |
| A100 | Llama-2-13B + 160M | 3.45× | +2.89× | +1.41× | +1.23× |
| A40 | Llama-2-7B + 68M | **2.76×** | +2.31× | +1.13× | +1.04× |

### Ablation Study (Optimization Decomposition, Llama-2-7B)

| Optimization Stage | Cumulative Speedup | Incremental Gain | Notes |
|--------------------|-------------------|------------------|-------|
| O1: EGT structure | 1.0× (baseline) | — | Algorithm-level baseline |
| O2: + Graph compilation | 2.775× | **2.775×** | Largest single contributor |
| O3: + Verification pruning | 2.970× | 1.07× | Adaptive verification count |
| O4: + Stage scheduling | 3.593× | 1.21× | Reduced GPU bubbles |
| O5: + Depth prediction | 3.952× | 1.10× | Context-adaptive depth |

### Key Findings

- **Compilation is the dominant contributor**: Graph compilation alone yields 2.775× speedup, demonstrating that static shape compatibility is critical for speculative decoding. Even a simple but compiler-friendly sequence structure surpasses complex but uncompilable approaches such as Sequoia.
- **Latency objective vs. AAL objective**: On the C4 dataset, the latency-aware objective provides an additional 8% speedup by more accurately capturing the nonlinear characteristics of verification latency.
- **Temperature robustness**: Yggdrasil consistently outperforms Sequoia across sampling temperatures, with an average additional speedup of 1.49×.
- **EGT parameter sensitivity**: The optimal configuration is $D_d=8, W_d=8, W_v=64$, indicating that tree width, depth, and verification count must be jointly optimized.

## Highlights & Insights

- **Algorithm-system co-design perspective**: Prior work addresses either the algorithm side (improving AAL) or the system side (compilation speedup), but Yggdrasil identifies the fundamental tension between these two directions and proposes a unified solution. This co-design paradigm is transferable to other settings that require both dynamism and compilation efficiency.
- **Elegance of EGT's constraints**: By fixing the number of new leaves added per step as a constant, EGT parameterizes the tree structure as three decoupled sub-problems, preserving compile-time static shapes while enabling runtime dynamic behavior—a clean engineering-algorithm trade-off.
- **Analogy to branch prediction**: The paper draws an analogy between speculative decoding and CPU branch prediction and cache prefetching; the ahead-of-time execution strategy is directly inspired by speculative execution in processor design.

## Limitations & Future Work

- **Single-request latency only**: Yggdrasil assumes a single request occupies the GPU exclusively and is not applicable to batch serving scenarios; joint optimization with batch schedulers remains an important open problem.
- **Offline profiling required**: The depth predictor and stage scheduling policy require offline training and search for each drafter-verifier pair and hardware configuration.
- **Model-invasive methods not supported**: Approaches that modify the target model architecture, such as Medusa and Eagle, are outside the scope of the current framework.
- Future work may explore combining EGT with self-speculative decoding methods such as LayerSkip.

## Related Work & Insights

- **vs. Sequoia**: Sequoia constructs a dataset-adaptive tree that balances AAL and overhead, but applies a single tree to all inputs and is incompilable. Yggdrasil's EGT dynamically selects the tree per request while remaining compiler-compatible.
- **vs. vLLM-Spec**: vLLM leverages compilation optimizations but supports only sequence structures (not trees), limiting AAL. Yggdrasil combines the high AAL of tree structures with the low latency of compilation optimization.
- **vs. DISCO**: DISCO employs fully dynamic trees for maximum AAL but incurs substantial runtime overhead that precludes compilation. EGT finds an optimal balance between dynamism and compiler compatibility.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel algorithm-system co-design perspective; both EGT and stage-based scheduling offer original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-hardware, multi-model evaluation with detailed optimization decomposition, though evaluation is limited to the Llama-2 family.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, figures and tables are well-designed, and the logical flow is complete.
- Value: ⭐⭐⭐⭐ A significant contribution to speculative decoding system optimization; the 3.98× speedup demonstrates strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DISC: Dynamic Decomposition Improves LLM Inference Scaling](disc_dynamic_decomposition_improves_llm_inference_scaling.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](3model_speculative_decoding.md)
- [\[ACL 2026\] TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs](../../ACL2026/llm_efficiency/tokentiming_a_dynamic_alignment_method_for_universal_speculative_decoding_model_.md)
- [\[NeurIPS 2025\] MoESD: Revealing the Potential of Speculative Decoding to Accelerate Sparse MoE](moesd_unveil_speculative_decodings_potential_for_accelerating_sparse_moe.md)
- [\[NeurIPS 2025\] Approximately Aligned Decoding](approximately_aligned_decoding.md)

</div>

<!-- RELATED:END -->
