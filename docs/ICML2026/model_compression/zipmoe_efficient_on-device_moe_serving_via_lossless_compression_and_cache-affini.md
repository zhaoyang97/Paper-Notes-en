---
title: >-
  [Paper Note] ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling
description: >-
  [ICML2026][Model Compression][MoE Inference] ZipMoE targets MoE large model inference on mobile and edge devices. It decomposes BF16 expert parameters into compressible exponent bits and high-entropy sign-mantissa bits.…
tags:
  - "ICML2026"
  - "Model Compression"
  - "MoE Inference"
  - "Lossless Compression"
  - "Edge Deployment"
  - "Cache Scheduling"
  - "Unified Memory Architecture"
date: 2026-05-08
content_hash: 3e5dd89c53dfd313
---

# ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling

**Conference**: ICML2026  
**arXiv**: [2601.21198](https://arxiv.org/abs/2601.21198)  
**Code**: https://github.com/npnothard/ZipMoE-ICML26  
**Area**: Model Compression / LLM Systems / Edge Inference  
**Keywords**: MoE Inference, Lossless Compression, Edge Deployment, Cache Scheduling, Unified Memory Architecture  

## TL;DR
ZipMoE targets MoE large model inference on mobile and edge devices. It decomposes BF16 expert parameters into compressible exponent bits and high-entropy sign-mantissa bits. Through lossless compression, hierarchical caching, and cache-affinity scheduling, the system transforms expert loading—previously bottlenecked by SSD I/O—into a process of decompression and recombination parallelized across multi-core CPUs. This reduces latency and improves throughput without altering model semantics.

## Background & Motivation
**Background**: MoE language models scale capacity via sparse computation where only a few experts are activated per token. Cloud deployments rely on CPU memory/SSD offloading, expert caching, and pipelining to move experts to GPUs. Edge devices aim to keep models local for privacy, low network dependency, and interactive responsiveness.

**Limitations of Prior Work**: Edge platform constraints differ significantly from servers. Devices like Jetson and Apple Silicon utilize Unified Memory Architectures (UMA) with limited DRAM, requiring expert retrieval from NVMe SSDs. Interactive applications often run at batch size 1, making it difficult to amortize I/O costs through large batches. Measurements show that I/O stalls in MoE decoding layers can rise from 38.5% on servers to 80.1% on edge devices, leaving computing resources idle.

**Key Challenge**: While model compression reduces memory footprint, common quantization and pruning techniques alter model behavior. For safety-sensitive or unsupervised edge deployments, metrics like perplexity or zero-shot accuracy are insufficient to guarantee reliability. Conversely, maintaining original parameters leads to severe I/O bottlenecks. The core conflict is achieving algorithmic consistency while preventing I/O from stalling edge MoE serving.

**Goal**: The problem is decomposed into three components: identifying a lossless compressible structure in MoE parameters, hiding decompression costs behind I/O on UMA/multi-core CPUs, and managing the caching and scheduling of experts (as full tensors, compressed blocks, or bit-fields) under tight memory budgets.

**Key Insight**: An observation of BF16 bit-field distributions reveals that while sign and mantissa bits are nearly random, exponent bits show a highly skewed distribution with a Shannon entropy of approximately 2.55-2.65 bits. Compressed models can be reduced to 68%-74% of their original size. This statistical redundancy in MoE parameters can be exploited losslessly.

**Core Idea**: Utilizing bit-field lossless compression and cache-scheduling co-design to shift expert access from "waiting for full tensor disk reads" to "reading partial data, parallel decompression, and rapid BF16 tensor recovery."

## Method
ZipMoE is an edge MoE serving system rather than a new model architecture. Its design focuses on decomposing, compressing, and serializing expert parameters offline, while constructing fine-grained task DAGs based on gated experts during online inference to interleave SSD I/O, CPU decompression, and GPU tensor recovery.

### Overall Architecture
The system consists of offline initialization and real-time inference.

During offline initialization, ZipMoE performs bit-field decomposition on each BF16 expert tensor. Exponent bits are partitioned into shards and compressed into E-chunks using lossless compressors (e.g., LZ4, ZSTD). Sign and mantissa bits are packed into byte-aligned SM-chunks. All chunks and metadata are stored as binary files on NVMe SSDs. As this is lossless, the recovered BF16 tensors are identical to the original parameters.

During real-time inference, the model gate identifies the required experts for the current sparse MoE layer. The cache manager determines capacity for different compression states, and the scheduler constructs a DAG for each expert tensor. This DAG involves reading SM-chunks, reading compressed E-chunks, CPU decompression of E-chunks, and a GPU kernel to recombine them into BF16 tensors. Execution involves an I/O thread, a pool of CPU worker threads, and a CUDA stream to minimize GPU wait time.

### Key Designs
1. **BF16 Bit-field Lossless Compression**:
    - **Function**: Reduces the bytes read from SSD without altering numerical values.
    - **Mechanism**: Processes sign, exponent, and mantissa separately. Exponents exhibit low entropy and are compressed, while high-entropy sign-mantissa bits are stored directly. Inference involves decompressing E-chunks and merging them with SM-chunks to restore BF16 values.
    - **Design Motivation**: Quantization saves memory but changes behavior, whereas full offloading is I/O bound. Bit-field compression exploits BF16 representation redundancy rather than model approximation, making it ideal for safety-critical edge scenarios.

2. **Compression State and Cache-affinity Scheduling**:
    - **Function**: Selects execution DAGs based on cached components and interleaves I/O with decompression.
    - **Mechanism**: Experts are categorized into states such as E-expert, SM-expert, compressed-expert, or full tensor. The scheduler classifies tasks into Type-I (requiring SM-chunk loading) and Type-II (SM already cached). Type-I SM reads occupy the I/O thread while Type-II decompression tasks fill CPU worker idle time. The makespan adheres to $ALG \le (3 - 1/L) \cdot OPT$, where $L$ is the number of decompression threads.
    - **Design Motivation**: Edge bottlenecks arise from the lack of simultaneous I/O, CPU, and GPU utilization. Explicitly modeling cache states as DAGs allows the system to determine which tasks can be hidden behind others.

3. **Compression-aware Hierarchical Cache Management**:
    - **Function**: Allocates fixed memory budgets between full tensors, compressed tensors, SM-chunks, and E-chunks.
    - **Mechanism**: ZipMoE maintains four cache pools. It uses historical activation frequencies to construct a rank-based activation distribution. A Poisson-binomial dynamic programming approach estimates hit combination probabilities to find the optimal pool partitions via grid search.
    - **Design Motivation**: MoE expert access is skewed, but popular experts change with prompts. Modeling by rank rather than fixed expert IDs allows the system to exploit long-tail distributions without being overfitted to specific workloads.

### Loss & Training
Ours does not involve training new models or additional loss functions. The optimization objective is system-level sparse layer makespan and end-to-end inference latency. Offline work is restricted to lossless parameter re-encoding. Models used in experiments are directly from Hugging Face with unmodified weights.

## Key Experimental Results

### Main Results
Experiments were conducted on DeepSeekV2-Lite, Qwen1.5-MoE, and SwitchTransformers-Large-128 using Jetson AGX Orin (64GB/32GB). Baselines include MoE-Infinity, DeepSpeed ZeRO-3 offloading, and Accelerate.

| Scenario | Metric | Results of ZipMoE relative to baseline | Baseline | Description |
|------|------|-----------------------------|----------|------|
| Decoder-only MoE Inference | TPOT | Reduced by 62.65%-97.97% | MoE-Infinity / DeepSpeed / Accelerate | Real-time response during token generation is significantly improved |
| Decoder-only MoE Inference | TTFT | Reduced by 53.25%-87.90% | Same as above | Wait time for the first token is significantly shortened |
| Encoder-decoder MoE | TPOT | Reduced by 4.99%-81.24% | Same as above | Gains exist despite higher activation skew |
| Batch inference | Throughput | Decoder-only: 1.79x-42.49x Gain; Encoder-decoder: 1.31x-5.82x Gain | Same as above | Larger batches increase parallel scheduling efficiency |
| End-to-end generation | Latency | Decoder-only: 3.03x-42.49x speedup; Encoder-decoder: 1.11x-5.64x speedup | Same as above | Advantages sustained across varied output lengths |

### Ablation Study
The study evaluates the contributions of basic eviction, heterogeneous cache pools, and cache planning (Table 1).

| Model | Configuration | Throughput (tokens/s) | E2E (s) | Description |
|------|------|------------------------|---------|------|
| DeepSeekV2-Lite 16B | Baseline | 1.60 | 585.96 | Offloading baseline |
| DeepSeekV2-Lite 16B | ZipMoE avg. basic | 4.43 | 204.05 | Major gains from basic caching |
| DeepSeekV2-Lite 16B | ZipMoE +C | 5.18 | 176.68 | Gains from heterogeneous pools |
| DeepSeekV2-Lite 16B | ZipMoE +C+P | 5.30 | 173.23 | Optimal Pareto point with planning |
| Qwen1.5-MoE 14B | Baseline | 1.99 | 515.12 | Offloading baseline |
| Qwen1.5-MoE 14B | ZipMoE avg. basic | 6.39 | 160.33 | Decompression & scheduling contribute most |
| Qwen1.5-MoE 14B | ZipMoE +C | 7.64 | 134.10 | Hierarchical caching boosts throughput |
| Qwen1.5-MoE 14B | ZipMoE +C+P | 7.79 | - | Planning provides incremental gains |

### Key Findings
- Benefits stem from a paradigm shift: using lossless compression to reduce disk reads and multi-core CPU parallelism to transition expert loading from I/O-bound to compute-parallel. The core system contributes ~76% of throughput gains, while cache management adds ~24%.
- OS page cache is a secondary benefit. Even with 32GB RAM artificially occupied, ZipMoE still achieves 56.64% lower TTFT and 53.32% lower TPOT compared to baselines.
- Encoder-decoder MoE gains are lower than decoder-only versions due to more skewed activations and lower I/O intensity, suggesting ZipMoE is most effective in I/O-dominant scenarios.

## Highlights & Insights
- The system cleverly avoids the "compression equals quantization" trap by finding lossless redundancy within the BF16 format.
- The abstraction of "compression state" is highly practical, allowing the scheduler to distinguish between full hits, partial hits, and misses to generate optimal DAGs.
- The design exploits an edge SoC counter-intuition: while the CPU is often idle during I/O stalls, its multi-core decompression capability is sufficient to hide the cost of bit-field recovery. This is applicable to other sparse models or retrieval-based parameter banks.

## Limitations & Future Work
- Evaluation is primarily on NVIDIA Jetson AGX Orin. While applicable to other shared-memory platforms, performance on mobile NPUs or Apple Neural Engine requires further validation.
- The method depends on the low-entropy exponent distribution of BF16. Gains might diminish if models use different numerical formats or if training processes change parameter distributions.
- Current focus is on expert parameter access. Prefill KV cache pressure, multi-app concurrency, and energy consumption remain for future study.
- Cache planning relies on historical activation statistics; rapid adaptation in non-stationary personal assistant scenarios remains an open question.

## Related Work & Insights
- **vs. Quantized/Pruned MoE Systems**: These use approximations to reduce size. ZipMoE maintains original BF16 semantics, ensuring behavioral consistency at the cost of a lower compression ceiling.
- **vs. MoE-Infinity / MoELightning / Klotski**: These focus on offloading, prefetching, and pipelining. ZipMoE specifically addresses edge UMA + SSD environments by decomposing experts into parallelizable chunks.
- **vs. DFloat11 / HuffLLM / nvCOMP**: While these show lossless compression works for LLMs, they are not optimized for conditional MoE activation or edge AArch64 CPUs. ZipMoE integrates compression, partial caching, and scheduling into a unified system.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines BF16 bit-field compression with cache-affinity scheduling in a comprehensive system.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models and hardware budgets; however, lacks extensive mobile SoC/phone and power consumption metrics.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and thorough theoretical grounding.
- Value: ⭐⭐⭐⭐⭐ High practical value for edge MoE serving, especially for security-sensitive applications requiring lossless parameters.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ACL 2026\] The Pitfalls of KV Cache Compression](../../ACL2026/model_compression/the_pitfalls_of_kv_cache_compression.md)
- [\[ICML 2026\] Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression](breaking_the_moe_llm_trilemma_dynamic_expert_clustering_with_structured_compress.md)
- [\[ICML 2026\] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](xkv_cross-layer_kv-cache_compression_via_aligned_singular_vector_extraction.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)

</div>

<!-- RELATED:END -->
