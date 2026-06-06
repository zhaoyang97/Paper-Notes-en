---
title: >-
  [Paper Note] Efficient Learned Data Compression via Dual-Stream Feature Decoupling
description: >-
  [ACL 2026][Model Compression][Learned Data Compression] This paper proposes the FADE framework, which separates microscopic syntax and macroscopic semantic features into parallel shallow streams for processing (replacing…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Learned Data Compression"
  - "Dual-Stream Feature Decoupling"
  - "Probability Modeling"
  - "Parallel Pipeline"
  - "Lossless Compression"
date: 2026-05-08
content_hash: b8d62ef27e09fe8a
---

# Efficient Learned Data Compression via Dual-Stream Feature Decoupling

**Conference**: ACL 2026  
**arXiv**: [2604.07239](https://arxiv.org/abs/2604.07239)  
**Code**: [https://github.com/huidong-ma/FADE](https://github.com/huidong-ma/FADE)  
**Area**: Model Compression / Data Compression  
**Keywords**: Learned Data Compression, Dual-Stream Feature Decoupling, Probability Modeling, Parallel Pipeline, Lossless Compression

## TL;DR
This paper proposes the FADE framework, which separates microscopic syntax and macroscopic semantic features into parallel shallow streams for processing (replacing deep serial stacks) via a Dual-stream Multi-scale Decoupler. Combined with a Hierarchical Gated Refiner and a Concurrent Stream Parallel Pipeline, it achieves SOTA performance in both compression ratio and throughput.

## Background & Motivation

**Background**: Learned Data Compression (LDC) leverages deep learning for probability prediction, significantly outperforming traditional methods (Gzip, zstd, etc.) in compression ratios. Mainstream methods utilize autoregressive frameworks—predicting the conditional probability distribution $P(x_t|x_{<t})$ at each step, followed by entropy coding.

**Limitations of Prior Work**: Two structural constraints exist: (1) Single-stream architectures struggle to simultaneously capture microscopic syntax (local N-gram patterns) and macroscopic semantics (long-range dependencies), forcing the use of deep MLP stacks to approximate complex distributions, which exacerbates autoregressive decoding latency; (2) Speed mismatch between GPU probability generation and CPU arithmetic coding in heterogeneous systems causes pipeline stalls, while serial autoregressive decoding is strictly limited by Amdahl’s law, preventing parallel acceleration.

**Key Challenge**: Precise probability modeling (high compression ratio) requires deep networks, but deep serial execution leads to high latency. Analysis of mutual information decay curves reveals that data sequences indeed exhibit two distinct dependency patterns: "microscopic syntax" (sharp initial decay) and "macroscopic semantics" (sustained non-zero tail). Single-stream MLPs use shared parameters to fit these heterogeneous features, leading to significant distribution dispersion.

**Goal**: Substantially reduce latency and increase throughput while maintaining or improving the compression ratio.

**Key Insight**: An information-theoretic analysis of the dual dependency patterns justifies the design of explicit feature decoupling—replacing deep serial layers with shallow parallel streams to address both model and system-level bottlenecks.

**Core Idea**: Use a CNN branch to capture microscopic local patterns and an MLP branch to capture macroscopic global dependencies. These are fused dynamically via a content-adaptive router, followed by instance-adaptive refinement using a Hierarchical Gated Refiner.

## Method

### Overall Architecture
FADE consists of three core innovations: (1) Dual-stream Multi-scale Decoupler (DMD) separates features into a local CNN stream and a global MLP stream for parallel processing; (2) Hierarchical Gated Refiner (HGR) achieves instance-adaptive probability modeling through coarse-to-fine refinement; (3) Concurrent Stream Parallel Pipeline (CSPP) integrates data and temporal parallelism for zero-wait processing.

### Key Designs

1. **Dual-stream Multi-scale Decoupler (DMD)**:

    - **Function**: Separates microscopic syntax and macroscopic semantic features into parallel streams with different inductive biases, replacing deep serial stacking.
    - **Mechanism**: The global stream uses a GeGLU-based Rolling Cache to capture long-range dependencies, maintained via a rolling cache $\bm{M}$ updated as $\bm{M}_t = \text{Roll}(\bm{M}_{t-1}, \text{GeGLU}(\bm{X}_t))$. The local stream applies 1D convolutions to enforce strong local inductive biases for N-gram patterns. A content-adaptive router generates dimension-wise mixing weights via a Sigmoid gate: $\bm{H}_{\text{mix}} = \bm{\alpha} \odot \bm{H}_{\text{global}} + (1-\bm{\alpha}) \odot \bm{H}_{\text{local}}$.
    - **Design Motivation**: Mutual information decay analysis and feature saliency heatmaps confirm that single-stream MLP saliency is dispersed, failing to capture sharp microscopic fluctuations. Two parallel shallow streams replace one deep serial stream, resolving both feature interference and latency issues.

2. **Hierarchical Gated Refiner (HGR)**:

    - **Function**: Performs coarse-to-fine instance-adaptive refinement on the fused DMD features to improve probability estimation accuracy.
    - **Mechanism**: A two-stage cascade: (a) Coarse-grained channel interaction: uses Batch Matrix Multiplication (BMM) with persistent memory $\bm{W}_U \in \mathbb{R}^{B \times d_h \times d_h}$, where each batch index corresponds to a fixed data stream to capture stream-specific patterns through backpropagation; then utilizes content-aware self-gating $\bm{H}_{\text{coarse}} = (\bm{H}_a \odot \sigma(\bm{H}_{\text{mix}} \bm{W}_c)) + \lambda_c \cdot \bm{H}_{\text{mix}}$ to suppress noise. (b) Fine-grained non-linear refinement: further refined through GeGLU and projections.
    - **Design Motivation**: Globally shared DMD parameters struggle with non-stationary feature distributions in online compression. Persistent memory allows "each stream to remember its own patterns," while gating selectively enhances useful features.

3. **Concurrent Stream Parallel Pipeline (CSPP)**:

    - **Function**: Overcomes serial autoregressive constraints to achieve full-pipeline parallelism for compression and decompression.
    - **Mechanism**: Two dimensions of parallelism: (a) Temporal parallelism: asynchronous ping-pong buffers decouple GPU and CPU producer-consumer threads, with zero-copy pointer exchanges eliminating memory contention; (b) Data parallelism: the input stream is split into $N$ independent sub-streams, each maintaining internal causality, where $N$ workers execute concurrently via a double-barrier protocol, reducing complexity from $O(B)$ to $O(B/N)$.
    - **Design Motivation**: Existing methods utilize temporal parallelism during compression but revert to serial execution during decompression due to autoregressive causality. CSPP bypasses global causal dependencies through sub-stream partitioning, matching decompression speed with compression speed.

### Loss & Training
The cross-entropy loss is used to optimize probability prediction accuracy. Persistent memory in the HGR is adapted to stream-specific patterns through online backpropagation during compression/decompression.

## Key Experimental Results

### Main Results

| Method | Avg Compression Ratio↑ | Throughput | Latency | GPU Memory |
|------|-----------|--------|------|--------|
| Traditional (Gzip/zstd) | Low | High | Low | — |
| PAC | Med-High | Med | Med | Med |
| SEP | High | Med-High | Med | Med-High |
| EDPC | High | High | Med-Low | Med-Low |
| FADE | **Highest** | **Highest** | **Lowest** | **Lowest** |

### Ablation Study

| Configuration | Compression Ratio | Throughput | Description |
|------|--------|--------|------|
| Full FADE | Optimal | Optimal | Complete model |
| w/o Local Stream | Decrease | Slight Increase | Loss of micro-syntax capture |
| w/o HGR | Decrease | Slight Increase | Loss of instance adaptivity |
| w/o CSPP | Same | Significant Drop | Importance of system parallelism |

### Key Findings
- FADE achieves SOTA in both compression ratio and throughput, breaking the conventional trade-off between the two.
- Dual-stream decoupling replaces deep serial layers with shallow parallel ones, significantly reducing latency while enhancing representational capacity.
- Persistent memory enables HGR to achieve stream-specific adaptation during online compression, making it more accurate than globally shared parameters.
- The CSPP data parallelism strategy makes decompression speed nearly equal to compression speed, solving the long-standing asymmetry problem.
- Excellent performance across heterogeneous data including text, audio, images, video, floating-point, and genomic sequences.

## Highlights & Insights
- **Complete Chain from Information Theory to Architecture**: The existence of dual dependency patterns is verified via mutual information decay and self-similarity matrices before designing the decoupled architecture. This "analysis-driven design" is more rigorous than intuition.
- **Shallow Parallelism over Deep Serialism**: Reducing latency without sacrificing expressiveness. The core insight is that "separation + specialization" is superior to "unification + stacking."
- **Innovative Use of Persistent Memory**: Each batch index in BMM corresponds to a learnable weight matrix that evolves via backpropagation during online compression, effectively "memorizing the unique patterns of each data stream."

## Limitations & Future Work
- Data parallelism requires splitting input into independent sub-streams, which ignores cross-stream dependencies.
- Persistent memory size scales linearly with batch size, potentially leading to significant memory overhead in large-scale parallelism.
- There is still a gap in compression ratio compared to LLM-based methods (e.g., LLMZip), though the efficiency advantage is massive.
- The weight assignment strategy of the adaptive router is relatively simple; more complex MoE-style routing could be explored.

## Related Work & Insights
- **vs PAC/OREO**: Lightweight MLP-based methods using masking and caching. FADE further improves efficiency and expressiveness through dual-stream decoupling.
- **vs SEP**: SEP introduces semantic enhancement modules and multi-stream pipelines. FADE's CSPP achieves more comprehensive parallelization.
- **vs EDPC**: EDPC proposes a dual-path framework and latent transformation engine. FADE's DMD specifically targets decoupling at the micro/macro pattern level.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-stream decoupling design has clear theoretical support and experimental validation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 7 datasets (text, audio, image, video, float, genome, heterogeneous).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, progressing logically from analysis to design to system implementation.
- Value: ⭐⭐⭐⭐ High engineering practicality, simultaneously addressing both model and system bottlenecks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation](../../CVPR2026/model_compression/dage_dual-stream_architecture_for_efficient_and_fine-grained_geometry_estimation.md)
- [\[ACL 2026\] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)
- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](../../ICML2026/model_compression/efficient_learned_image_compression_without_entropy_coding.md)
- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](../../AAAI2026/model_compression/infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ACL 2026\] Alignment Tuning for Large Language Models: A Data-Centric Lens on Alignment Data Pipelines](alignment_tuning_for_large_language_models_a_data-centric_lens_on_alignment_data.md)

</div>

<!-- RELATED:END -->
