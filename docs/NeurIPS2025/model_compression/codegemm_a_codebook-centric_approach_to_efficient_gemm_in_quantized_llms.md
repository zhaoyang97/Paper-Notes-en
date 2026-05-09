---
title: >-
  [Paper Note] CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs
description: >-
  [NeurIPS 2025][Model Compression][model quantization] This paper proposes CodeGEMM, a codebook-centric GEMM kernel that precomputes inner products between centroids and activations and caches them as a Psumbook, replacing the conventional dequantization pipeline to achieve end-to-end speedups of 1.83× (8B) to 8.93× (70B) on 2-bit quantized LLMs.
tags:
  - NeurIPS 2025
  - Model Compression
  - model quantization
  - codebook quantization
  - GEMM acceleration
  - LLM inference
  - low-bit quantization
  - CUDA kernel
  - lookup table
date: 2026-05-08
content_hash: 6030ad8b99d2ca82
---

# CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2512.17970](https://arxiv.org/abs/2512.17970)
**Code**: [GitHub](https://github.com/naver-aics/codegemm)
**Area**: Model Compression
**Keywords**: model quantization, codebook quantization, GEMM acceleration, LLM inference, low-bit quantization, CUDA kernel, lookup table

## TL;DR

This paper proposes CodeGEMM, a codebook-centric GEMM kernel that precomputes inner products between centroids and activations and caches them as a Psumbook, replacing the conventional dequantization pipeline to achieve end-to-end speedups of 1.83× (8B) to 8.93× (70B) on 2-bit quantized LLMs.

## Background & Motivation

- **Background**: Weight-only quantization is the mainstream approach for alleviating memory bottlenecks in LLM inference; codebook-based quantization methods (e.g., AQLM, GPTVQ, QuIP#) maintain competitive accuracy at extremely low bit-widths (e.g., 2-bit) and have become a leading research direction.
- **Limitations of Prior Work**: Inference kernels for codebook quantization rely on a dequantization pipeline—loading the full codebook into on-chip programmable cache (shared memory) and then performing element-wise centroid lookups to reconstruct weights—which leads to: (1) codebooks potentially exceeding shared memory capacity (e.g., AQLM 1×16 requires 1 MB, far exceeding the 164 KB on A100); and (2) redundant computation of the same centroid–input products across every GEMM call.
- **Key Challenge**: Codebook quantization achieves superior accuracy over uniform quantization, yet its inference efficiency can be worse—the AQLM 1×16 kernel latency even exceeds the FP16 baseline, completely negating the memory savings from compression.
- **Goal**: Design an efficient codebook-centric GEMM kernel that simultaneously reduces computational complexity and on-chip memory requirements, enabling codebook quantization to achieve both high accuracy and high throughput at extremely low bit-widths.
- **Key Insight**: The number of centroids in a codebook is bounded by $2^b$, while the number of weight matrix rows $M$ greatly exceeds $2^b$; consequently, many codes point to the same centroid, causing their inner products with the input to be recomputed repeatedly. Precomputing and caching these inner products eliminates this redundancy.
- **Core Idea**: Cache a Psumbook (a precomputed partial-sum lookup table) in shared memory in place of the codebook itself; at inference time, results are retrieved by code index and accumulated directly, bypassing the dequantization step entirely.

## Method

### Overall Architecture

CodeGEMM simplifies the conventional codebook GEMM pipeline of "load codebook → dequantize → matrix multiply" into "build Psumbook → index lookup → accumulate." Specifically: (1) the input tile is partitioned into segments of vector length $v$; (2) the inner product of each input segment with all $2^b$ centroids is computed and stored in the Psumbook (cached in shared memory); (3) the code matrix indices are used to directly retrieve the corresponding partial sums from the Psumbook and accumulate them into the output. The entire process requires neither loading the full codebook nor element-wise dequantization.

### Key Design 1: Psumbook Precomputation and Caching

- **Function**: For each input segment $\mathbf{x}^j$, precompute its inner product with every centroid as $p_i^j = \sum_{k=0}^{v-1} c_k^i \times x_k^j$, and store these scalar results in the Psumbook.
- **Mechanism**: The codebook stores $v$-dimensional vectors (centroids), whereas the Psumbook stores scalars (inner product results), reducing space complexity from $\mathcal{O}(m \cdot 2^b \cdot v)$ to $\mathcal{O}(m \cdot 2^b \cdot t_w/v)$, inversely proportional to the vector length $v$.
- **Design Motivation**: Conventional methods must load the entire codebook into shared memory; large codebooks (e.g., $2^{16}$ centroids) immediately exceed capacity limits. By storing only scalar results, the Psumbook substantially reduces on-chip memory requirements, enabling all codebook configurations to fit within shared memory.

### Key Design 2: Reduction in Computational Complexity

- **Function**: CodeGEMM's computational complexity is $\mathcal{O}(MNK \cdot m/v)$, reducing the standard GEMM cost of $\mathcal{O}(MNK)$ by a factor of $v/m$.
- **Mechanism**: The Psumbook construction cost $C_{build} = \mathcal{O}(m \cdot 2^b \cdot K \cdot N)$ is negligible when $M \gg 2^b$; the retrieval cost requires only one table lookup per code (rather than $v$ multiply-add operations), yielding $C_{read} = \mathcal{O}(m \cdot M \cdot K/v \cdot N)$.
- **Design Motivation**: Dequantization-based kernels optimize only data movement efficiency while leaving the arithmetic workload identical to FP16 GEMM. CodeGEMM simultaneously optimizes both data movement and arithmetic, constituting a genuine improvement in computational efficiency.

### Key Design 3: Unified Kernel for Flexible Hyperparameter Exploration

- **Function**: A single kernel implementation supports arbitrary combinations of hyperparameters: number of codebooks $m$, vector length $v$, bits per code $b$, and group size $g$.
- **Mechanism**: Different hyperparameter combinations can yield drastically different latency–accuracy trade-offs at the same average bit-width (e.g., $(v=4, m=1, b=8, g=128)$ and $(v=16, m=3, b=8, g=32)$ are both approximately 2-bit but exhibit markedly different performance profiles).
- **Design Motivation**: Existing codebook kernels are typically optimized for fixed configurations. A unified kernel allows systematic exploration of the latency–memory–accuracy trade-off space to identify optimal configurations for specific deployment scenarios.

### Key Design 4: Fine-Grained Group Normalization

- **Function**: Weights are normalized by group size $g$ prior to quantization; smaller $g$ yields finer-grained normalization ($g=v$ corresponds to per-vector, $g=-1$ to per-row).
- **Mechanism**: Fine-grained normalization reduces quantization error, trading a small amount of additional memory (for storing scale factors) for improved accuracy.
- **Design Motivation**: On 70B models, fine-grained normalization allows CodeGEMM to match the accuracy of AQLM 1×16 (which uses a $2^{16}$-entry codebook) while achieving 8.93× higher throughput.

## Loss & Training

- **Quantization Optimization**: The block-wise codebook optimization strategy from AQLM is adopted; centroids are determined via K-means clustering on a calibration dataset.
- **PV-Tuning**: An optional post-quantization calibration method that further refines the codebook for improved accuracy (with PV-Tuning, CodeGEMM-m1v4g128 improves average accuracy on Llama-3.1-8B from 53.93 to 63.96).

## Key Experimental Results

### Main Results 1: Kernel-Level Latency and End-to-End Throughput (Llama-3.1-8B, 2-bit)

| Method | Config | Kernel Latency (μs) | End-to-End Throughput (tok/s) | Avg. Accuracy |
|------|------|:-:|:-:|:-:|
| cuBLAS (FP16) | — | 332.45 | 103.8 | 71.26 |
| AQLM | 1×16 | 645.51 | 49.0 | 63.57 |
| AQLM | 2×8 | 250.12 | 124.5 | 47.82 |
| QuIP# | e8p | 162.63 | — | — |
| LUTGEMM | q2-g128 | 160.1 | — | — |
| **CodeGEMM** | m1v4g128 | **152.69** | **228.3** | 53.93 |
| **CodeGEMM+PV** | m1v4g128 | 152.69 | 228.3 | **63.96** |

CodeGEMM-m1v4g128 achieves the lowest kernel latency (152.69 μs) and an end-to-end throughput of 228.3 tok/s, which is 4.66× that of AQLM 1×16. With PV-Tuning, its accuracy (63.96) surpasses AQLM 1×16 (63.57), yielding an overall 1.83× end-to-end speedup.

### Main Results 2: Scalability on 70B Models

| Method | $\bar{q}$ | tok/s | MMLU | WG | HS | ARC-E | ARC-C | Avg. |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| AQLM 1×16 | 2.055 | 5.5 | 73.07 | 76.16 | 80.83 | 82.20 | 57.17 | 73.89 |
| AQLM 2×8 | 2.002 | 19.0 | 61.45 | 59.59 | 52.83 | 48.82 | 28.67 | 50.27 |
| **CodeGEMM** m1v4g32 | 2.500 | 49.1 | 71.21 | 76.64 | 79.43 | 82.41 | 56.06 | **73.15** |
| **CodeGEMM** m1v4g128 | 2.125 | **51.2** | 68.15 | 74.90 | 75.37 | 79.42 | 52.73 | 70.11 |

On the 70B model, CodeGEMM-m1v4g128 achieves 51.2 tok/s, which is **8.93×** the throughput of AQLM 1×16 (5.5 tok/s). CodeGEMM-m1v4g32 achieves an average accuracy of 73.15, closely matching AQLM 1×16 (73.89), while being nearly 9× faster.

### Main Results 3: Energy Efficiency and Hardware Utilization

| Method | TFLOPS | Power (W) | GFLOPS/W | Mem Util (%) |
|------|:-:|:-:|:-:|:-:|
| cuBLAS (FP16) | 1.58 | 318.55 | 4.95 | 96.94 |
| AQLM 1×16 | 0.75 | 126.54 | 5.93 | 6.00 |
| AQLM 2×8 | 2.59 | 254.20 | 10.18 | 19.96 |
| **CodeGEMM** m1v4g128 | **6.12** | 316.38 | **19.36** | **49.80** |

CodeGEMM's energy efficiency (19.36 GFLOPS/W) is 1.9× that of AQLM 2×8, and its memory subsystem utilization (49.80%) is substantially higher than AQLM (6%–20%), indicating more structured and efficient DRAM access patterns.

## Highlights & Insights

- **Core Innovation**: Replacing codebook caching with a Psumbook of precomputed inner products simultaneously reduces computational complexity (by $v/m$) and space complexity (by $v^2/t_w$), representing a fundamental paradigm shift in codebook quantization inference.
- **Excellent Large-Model Scalability**: The advantage grows with model size (8B: 1.83× → 70B: 8.93×), since larger models are more likely to have codebooks that exceed shared memory capacity.
- **Unified Kernel for Hyperparameter Exploration**: A single kernel covers diverse $(m, v, b, g)$ configurations, for the first time systematically characterizing the three-dimensional latency–memory–accuracy trade-off space in codebook quantization.
- **Strong Engineering Completeness**: System-level evaluations including energy efficiency and hardware utilization are provided, making this contribution both an algorithmic and an engineering advance.

## Limitations & Future Work

- Experiments are conducted exclusively on NVIDIA A100; generalizability to other GPU architectures (e.g., H100, consumer GPUs) remains unverified.
- Accuracy evaluation relies primarily on perplexity and zero/few-shot tasks; evaluations on practical scenarios such as long-form text generation are absent.
- Psumbook construction overhead scales linearly with $N$ ($C_{build} \propto N$), and the paper provides insufficient analysis of performance under large-batch settings.
- The quantization algorithm itself follows AQLM without algorithmic innovation; PV-Tuning is an external method.
- Compatibility with complementary inference optimizations such as KV cache quantization and speculative decoding is not evaluated.

## Related Work & Insights

- **Codebook Quantization**: AQLM (additive multi-codebook), GPTVQ (GPTQ + codebook), QuIP#/QTIP (rotation smoothing + lattice codebook / trellis coding), VPTQ (vector quantization).
- **Uniform Quantization Kernels**: INT3/INT4 kernels from GPTQ/AWQ; LUT-GEMM (lookup table acceleration for BCQ format).
- **Lookup-Table-Based Computation**: LUT-GEMM, FigLUT, and related works exploit LUTs at the hardware level to accelerate computation; CodeGEMM can be viewed as LUT-based computation applied to the codebook domain.
- **Key Distinction**: All existing codebook kernels follow a dequantization paradigm (load codebook → reconstruct → multiply); CodeGEMM is the first to propose a codebook-centric paradigm (precompute → lookup → accumulate).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The Psumbook concept is elegant and concise, representing a fundamental improvement to the codebook quantization inference paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 8B/70B models, diverse configuration comparisons, and multi-dimensional evaluation (kernel latency, end-to-end throughput, energy efficiency), though large-batch and multi-architecture experiments are missing.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, figures are intuitive, and complexity analysis is complete; overall writing quality is high.
- **Value**: ⭐⭐⭐⭐ — Offers direct practical value for codebook quantization deployment; the 8.93× speedup on 70B models is highly impressive.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](vision-centric_token_compression_in_large_language_model.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[NeurIPS 2025\] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models](vessa_video-based_object-centric_self-supervised_adaptation_for_visual_foundatio.md)
- [\[NeurIPS 2025\] Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs](matryoshka_pilot_learning_to_drive_black-box_llms_with_llms.md)

<!-- RELATED:END -->
