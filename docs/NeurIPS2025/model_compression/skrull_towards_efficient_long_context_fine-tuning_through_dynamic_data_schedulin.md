---
title: >-
  [Paper Note] Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling
description: >-
  [NeurIPS 2025][Model Compression][long-context fine-tuning] To address the training inefficiency caused by mixing long and short sequences in Long-context Supervised Fine-Tuning (Long-SFT), this paper proposes Skrull, a dynamic data scheduler consisting of two components — Distribution-Aware Context Parallelism (DACP) and Global Data Scheduling (GDS) — achieving an average 3.76× (up to 7.54×) training speedup in realistic Long-SFT scenarios.
tags:
  - NeurIPS 2025
  - Model Compression
  - long-context fine-tuning
  - data scheduling
  - context parallelism
  - training efficiency
  - large language models
date: 2026-05-08
content_hash: 16b176eb5f727753
---

# Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling

**Conference**: NeurIPS 2025
**arXiv**: [2505.19609](https://arxiv.org/abs/2505.19609)
**Code**: Not available
**Area**: Model Compression
**Keywords**: long-context fine-tuning, data scheduling, context parallelism, training efficiency, large language models

## TL;DR

To address the training inefficiency caused by mixing long and short sequences in Long-context Supervised Fine-Tuning (Long-SFT), this paper proposes Skrull, a dynamic data scheduler consisting of two components — Distribution-Aware Context Parallelism (DACP) and Global Data Scheduling (GDS) — achieving an average 3.76× (up to 7.54×) training speedup in realistic Long-SFT scenarios.

## Background & Motivation

Long-context capability is critical for LLMs; leading models (Llama3, Qwen2.5, GPT-4) support context windows of 128K or even 1M tokens. Acquiring this capability typically requires Long-SFT on datasets mixing long and short sequences. For instance, 99.89% of Llama3's fine-tuning data consists of short sequences (average 1K tokens), with only 0.11% being long sequences (average 37K tokens).

This **heterogeneous sequence length distribution** creates a dilemma for existing training frameworks:

**Parallelism strategy dilemma**: Long sequences require context parallelism (CP) and other memory-reduction strategies to avoid OOM, but these strategies introduce unnecessary communication overhead and low GPU utilization for short sequences. Experiments show that higher CP degrees increasingly degrade kernel execution efficiency for short sequences.

**Load balancing dilemma**: The $O(n^2)$ computational complexity and $O(n)$ memory consumption of the attention module make it impossible to simultaneously satisfy compute balance and memory balance.

**Low GPU utilization**: Memory-reduction strategies configured to accommodate the longest sequences result in severely idle GPU memory when processing short sequences.

Existing frameworks (DeepSpeed, Megatron) apply a uniform parallelism configuration to all sequences, making it impossible to efficiently handle both long and short sequences simultaneously, leading to suboptimal end-to-end performance.

## Method

### Overall Architecture

Skrull approaches the problem from a data scheduling perspective and comprises two levels of scheduling: (1) DACP performs fine-grained scheduling at the micro-batch level, selectively sharding sequences; (2) GDS performs coarse-grained scheduling at the global batch level, generating optimal micro-batch partitioning schemes. The two components work in concert, achieving near-zero-overhead runtime scheduling via offline performance modeling and lightweight online heuristics.

### Key Designs

1. **Distribution-Aware Context Parallelism (DACP)**: The core idea is to dynamically classify sequences within a micro-batch into two categories — **distributed sequences** (long sequences that require CP sharding) and **local sequences** (short sequences processed entirely on a single device). Both types are processed within the same CP group without increasing the number of GPUs. Key advantages include: local sequences avoid unnecessary communication overhead, and DACP can overlap the communication of distributed sequences with the computation of local sequences, further hiding communication latency. The scheduling objective is formalized as:
$$\min_{D,P} \max_j \text{Time}_j, \quad \text{Time}_j = \max(T_{comm}(V), T_{comp}(\text{Local}_j)) + T_{comp}(\text{Dist})$$
where $D$ is the sequence classification array (distributed/local), $P$ is the device assignment matrix for local sequences, and constraints include the memory limit $\sum_k S_k \cdot P_{kj} + D_k \cdot S_k / N \leq C$.

2. **Global Data Scheduling (GDS)**: DACP alone is insufficient — heterogeneous lengths also cause load imbalance across micro-batches. GDS performs coarse-grained batch partitioning at the global batch level (preserving optimizer mathematical equivalence). By pairing long and short sequences into micro-batches, it both expands DACP's optimization space and improves memory utilization. The joint optimization is formulated as:
$$\min_{B,D,P} \max_i \sum_j \text{Time}_{ij}$$
where $B_{kij}$ denotes the assignment of the $k$-th sequence to the $j$-th micro-batch of the $i$-th DP rank.

3. **Lightweight Heuristic Scheduling Algorithm**: Since exact MILP solving is too time-consuming, the authors design a heuristic algorithm integrated into the DataLoader with near-zero overhead. DACP scheduling follows three principles: **avoid sharding** (prefer local processing), **prioritize compute balance**, and **rollback mechanism** (reverting decisions upon memory overflow). GDS scheduling employs a FLOPs-estimation-based bin-packing algorithm to balance workload across DP workers and interleaves long and short sequences to optimize micro-batch composition.

### Loss & Training

Skrull's core contribution lies at the system level rather than the algorithmic level. An offline profiling procedure builds a performance model: memory consumption $Memory(S) = \alpha S + \beta$ (linear in sequence length), and compute $FLOPs(S_k) = 20bh^2 S_k + 4bhh_{kv}S_k + 4bhS_k^2$ (linear for linear modules, quadratic for attention). Skrull does not alter any training content or the global batch sequence order; it only changes the accumulation order, introducing negligible numerical differences due to floating-point non-associativity, with fully equivalent convergence behavior.

## Key Experimental Results

### Main Results — Training Speedup

| Model | Dataset | Skrull vs. DeepSpeed | Skrull vs. Sorted Batching |
|------|--------|---------------------|--------------------------|
| Qwen2.5-0.5B | Wikipedia | ~7.54× | ~6.85× |
| Qwen2.5-0.5B | LMsysChat1M | ~6.17× | ~5.40× |
| Qwen2.5-0.5B | ChatQA2-Long-SFT | ~2.79× | ~1.86× |
| Qwen2.5-7B | Wikipedia | ~2.60× | ~2.30× |
| Qwen2.5-7B | LMsysChat1M | ~2.14× | ~1.90× |
| Qwen2.5-7B | ChatQA2-Long-SFT | ~1.35× | ~1.20× |
| **Average** | — | **3.76×** | **3.45×** |

### Ablation Study — Component Contributions and Parameter Sensitivity

| Scheduling Strategy | Speedup | Notes |
|---------|--------|------|
| Round-Robin w/ rollback | 1.17× | Simple round-robin fails to effectively balance load |
| Round-Robin w/o rollback | OOM | Missing rollback causes out-of-memory error |
| Skrull w/ rollback | 1.40× | Heuristic scheduling significantly outperforms round-robin |
| Skrull w/o rollback | OOM | Rollback mechanism is critical for memory safety |

### Key Findings

- The average speedup for Qwen-0.5B (5.50×) is significantly higher than for Qwen-7B (2.03×), as the smaller model accommodates a larger BucketSize, providing greater scheduling headroom.
- Long-tail distribution datasets (Wikipedia, LMsysChat1M) offer larger optimization potential than bimodal distributions (ChatQA2-Long-SFT).
- Increasing BatchSize yields continuous but gradually saturating speedups; increasing BucketSize improves performance but also raises OOM risk.
- Incrementally enabling DACP and GDS validates both their individual effectiveness and their synergistic gains.
- The rollback mechanism is a **necessary** safety guarantee; omitting it consistently leads to OOM.

## Highlights & Insights

- **Novel perspective**: Addresses Long-SFT efficiency from a data scheduling viewpoint (rather than model or algorithm optimization), providing an optimization dimension orthogonal to existing approaches.
- **Balance of theory and practice**: Rigorously formalizes the scheduling process as a joint optimization problem while proposing practical heuristic approximations.
- **Elegant rollback design**: Incorporating reversible operations into greedy scheduling to prevent OOM reflects sound engineering judgment in system design.
- Training semantics are preserved and convergence behavior is fully equivalent; the method can be orthogonally combined with other techniques such as PEFT.

## Limitations & Future Work

- BucketSize requires manual configuration and currently relies on offline profiling.
- Speedup is limited for the ChatQA2-Long-SFT + Qwen-7B setting (1.35×), as most sequences already exceed BucketSize.
- The heuristic algorithm, while practical, is not optimal and may leave room for further improvement.
- Currently implemented only on DeepSpeed; adaptation to other frameworks such as Megatron-LM has not yet been completed.
- Combination with PEFT methods such as LoRA to enlarge BucketSize is possible but not fully validated experimentally.

## Related Work & Insights

Skrull is related to LongAlign (sorted batching), Chunkflow (fixed-size chunk training), and HotSPA (dynamic parallelism configuration). Skrull adopts a fixed parallelism configuration with dynamic data scheduling, making it orthogonal to dynamic parallelism approaches. Its data scheduling philosophy is broadly applicable to any training scenario mixing long and short data, such as RLHF.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The data scheduling perspective is novel and the joint optimization modeling is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple models and datasets with thorough incremental ablations, though testing on a broader range of model scales is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated and mathematical modeling is rigorous, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ Addresses an important and pervasive system efficiency problem in Long-SFT; the average 3.76× speedup carries high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)
- [\[NeurIPS 2025\] Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers](compress_gather_and_recompute_reforming_long-context_processing_in_transformers.md)

</div>

<!-- RELATED:END -->
