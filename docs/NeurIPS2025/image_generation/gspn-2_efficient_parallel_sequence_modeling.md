---
title: >-
  [Paper Note] GSPN-2: Efficient Parallel Sequence Modeling
description: >-
  [NeurIPS 2025][Image Generation][efficient attention] GSPN-2 achieves up to 40× speedup over GSPN-1's 2D spatial propagation through algorithm-system co-design — specifically, single-kernel fusion, compact channel propagation, and shared memory optimization — while matching Transformer-level accuracy on ImageNet classification and text-to-image generation at significantly lower computational cost.
tags:
  - NeurIPS 2025
  - Image Generation
  - efficient attention
  - spatial propagation
  - CUDA optimization
  - vision transformer
date: 2026-05-08
content_hash: 0a6865c79640dce0
---

# GSPN-2: Efficient Parallel Sequence Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2512.07884](https://arxiv.org/abs/2512.07884)
**Code**: [Project Page](https://whj363636.github.io/GSPN2/)
**Area**: Image Generation
**Keywords**: efficient attention, spatial propagation, CUDA optimization, vision transformer, image generation

## TL;DR

GSPN-2 achieves up to 40× speedup over GSPN-1's 2D spatial propagation through algorithm-system co-design — specifically, single-kernel fusion, compact channel propagation, and shared memory optimization — while matching Transformer-level accuracy on ImageNet classification and text-to-image generation at significantly lower computational cost.

## Background & Motivation

Vision Transformers are the backbone of nearly all state-of-the-art visual foundation models (Stable Diffusion, CLIP, SigLIP, detection/segmentation pipelines), yet the quadratic complexity of self-attention severely limits deployment at high resolutions or for long video sequences. SigLIP, for instance, is forced to cap inputs at 512×512 to avoid prohibitive latency.

Existing efficient attention alternatives include FlashAttention (fused tiled GEMM), linear attention, and state space models such as Mamba. Among these, the Generalized Spatial Propagation Network (GSPN) takes a distinctive approach by replacing 2D self-attention with **row/column scan propagation**, achieving near-linear complexity in the number of rows or columns ($O(\sqrt{N})$) and up to 84× speedup on 16K-resolution diffusion inference.

Nevertheless, the reference CUDA implementation of GSPN-1 suffers from three critical bottlenecks:

1. **Excessive kernel launch overhead**: Each column step launches a separate small kernel, resulting in thousands of micro-launches that prevent SMs from remaining fully saturated.
2. **Inefficient global memory access**: Each step reloads data from HBM without on-chip reuse or coalesced access, yielding memory bandwidth utilization of only 3–8%.
3. **Redundant per-channel computation**: Maintaining independent propagation weight matrices per channel causes runtime to grow linearly with channel count.

## Core Problem

How can the implementation bottlenecks of GSPN be eliminated at both the algorithmic and CUDA system levels — without sacrificing accuracy — so that the theoretically sub-quadratic complexity translates into real-world speedup?

## Method

### Review of 2D Spatial Propagation

For input $x \in \mathbb{R}^{H \times W \times C}$, GSPN propagates row by row:

$$h_{i,:,c} = w_{i,c} \, h_{i-1,:,c} + \text{Diag}(\lambda_{i,:,c}) \, x_{i,:,c}$$

where $w_{i,c} \in \mathbb{R}^{W \times W}$ is a tridiagonal row-stochastic matrix (row sums equal 1, ensuring numerical stability) and $\lambda_{i,:,c}$ is a learnable scaling vector. Four directional scans (top→bottom, bottom→top, left→right, right→left) together cover dense pairwise connections across the full image, with complexity $O(\max(H,W))$, i.e., $O(\sqrt{N})$ for square images.

### Optimization 1: Single-Kernel Fusion

GSPN-1 launches a separate kernel for each step along the propagation dimension. GSPN-2 merges all propagation steps into a **single unified CUDA kernel**, iterating over all columns via an internal loop. This alone yields a 1.2× speedup by eliminating the scheduling overhead of thousands of micro-launches.

The grid configuration is changed from a flat 1D layout to a 3D (chunk, n, c) indexing scheme, where each block corresponds to a unique (chunk, batch, channel) combination and handles the full column of spatial positions. Each block uses at most 1024 threads; when $H > 1024$, threads cover positions in a strided fashion.

### Optimization 2: Compact Channel Propagation

**GPU concurrency saturation**: When the number of active blocks $k_{\text{chunk}} \times N \times C$ exceeds the hardware concurrency limit (~3500 blocks on A100), runtime transitions from constant to linear growth.

GSPN-2 introduces a two-level mitigation strategy:

1. **Channel-shared weights**: A single propagation matrix $w_i$ replaces the per-channel $w_{i,c}$, so all channels share the same spatial propagation; only $\lambda_{i,:,c}$ retains per-channel modulation. $w_i$ then plays a role analogous to an attention affinity matrix:

$$h_{i,:,c} = w_i \, h_{i-1,:,c} + \lambda_{i,:,c} \odot x_{i,:,c}$$

2. **Compressed proxy dimension**: The input $x \in \mathbb{R}^{N \times C \times H \times W}$ is projected to a low-dimensional proxy space $x_{\text{proxy}} \in \mathbb{R}^{N \times C_{\text{proxy}} \times H \times W}$ ($C_{\text{proxy}} \ll C$, e.g., $C_{\text{proxy}}=8$); propagation is performed in this proxy space and the result is restored via a 1×1 convolution. The active block count drops from $k \times N \times C$ to $k \times N \times C_{\text{proxy}}$, keeping it within the hardware concurrency threshold.

The paper also provides a linear-attention interpretation: unrolling the recurrence yields a block lower-triangular form $H_v = G \, X_v$, where each sub-block $G_{ij}$ is an $N \times N$ attention affinity matrix.

### Optimization 3: Efficient CUDA Extensions

- **Shared memory caching**: The previous hidden state $h_{i-1}$ is cached in SRAM, reducing redundant HBM reads. This is particularly effective for multi-channel workloads; for single-channel cases, L1 cache already suffices and the optimization incurs a minor overhead.
- **2D block design**: blockDim = (H, cSlice), with threadIdx.x indexing spatial positions and threadIdx.y spanning channel slices, enabling a single block to process multiple channels of the same column in parallel.
- **Coalesced memory access**: $x_i$, $h_i$, and $w_i$ are laid out contiguously in memory so that threads within the same warp access adjacent addresses. **This is the single largest contributor** (23.9× speedup).
- **Stream concurrency**: The four directional propagations are dispatched on independent CUDA streams to maximize SM utilization.

### Cumulative Optimization Effect (1024×1024, B=16, C=8)

| Optimization Stage | Time (ms) | Cumulative Speedup |
|---|---|---|
| GSPN-1 Baseline | 71.4 | 1× |
| + Single Kernel | 57.4 | 1.2× |
| + Coalesced Access | 2.4 | 29.8× |
| + Shared Memory | 2.2 | 32.5× |
| + 2D Block | 2.1 | 34.0× |
| + Channel Compression | 1.9 | 37.6× |
| **GSPN-2 Final** | **1.8** | **40.0×** |

## Key Experimental Results

### ImageNet-1K Classification (224²)

| Model | Params (M) | MACs (G) | Top-1 Acc |
|---|---|---|---|
| GSPN-T | 30 | 5.3 | 83.0% |
| **GSPN-2-T** | **24** | **4.2** | **83.0%** |
| GSPN-S | 50 | 9.0 | 83.8% |
| **GSPN-2-S** | **50** | **9.2** | **84.4%** |
| GSPN-B | 89 | 15.9 | 84.3% |
| **GSPN-2-B** | **89** | **14.2** | **84.9%** |
| Swin-T | 29 | 4.5 | 81.3% |
| VMamba-T | 22 | 5.6 | 82.2% |
| MambaVision-B | 98 | 15.0 | 84.2% |

- GSPN-2-T achieves 83.0% with the fewest parameters (24M) and MACs (4.2G), surpassing all SSMs and most Transformers at the same scale.
- GSPN-2-S achieves 84.4% with 50M parameters, a +0.6% gain over GSPN-S, outperforming MambaOut-Small (84.1%) and UniFormer-B (83.9%).
- GSPN-2-B achieves 84.9% with 89M parameters while reducing MACs by 1.7G compared to GSPN-B.

### Memory Bandwidth Utilization

| Configuration | GSPN-1 | GSPN-2 |
|---|---|---|
| 128×128, C=32 | 98 GB/s (4.9%) | 1865 GB/s (**93.3%**) |
| 256×256, C=64 | 76 GB/s (3.8%) | 1842 GB/s (**92.1%**) |
| 512×512, C=128 | 64 GB/s (3.2%) | 1840 GB/s (**92.0%**) |

GSPN-2 consistently achieves 91–93% of A100's theoretical peak bandwidth across all configurations, compared to only 3–8% for GSPN-1.

### Text-to-Image Generation (SDXL Framework)

- 4K image generation: **32×** speedup over the SDXL baseline.
- 16K image generation: improved from GSPN-1's 84× to **93×** (single A100).
- COCO benchmark: FID 33.21 / CLIP-T 0.286, approaching the SD-v1.5 baseline.

### Ablation on Proxy Dimension (GSPN-2-Tiny)

| $C_{\text{proxy}}$ | Acc | Throughput (img/s) |
|---|---|---|
| 2 | 83.0% | 1544 |
| 8 | 83.0% | 1387 |
| 32 | 82.8% | 1106 |

Even with aggressive 48:1 compression ($C_{\text{proxy}}=2$), accuracy is unchanged while throughput improves by 1.4×, indicating that spatial dependencies dominate and channel dependencies are weak.

### Training Details and Design Choices

Key design decisions in the ImageNet experiments:
- Propagation weights $w_i$ are **shared across channels** in all GSPN modules.
- Compressed proxy dimension $C_{\text{proxy}} = 2$ (48:1 compression ratio); the saved parameters are redistributed to deeper/wider networks.
- A **Local Perception Unit (LPU)** is integrated at the beginning of each block and within the FFN to enhance local feature extraction.
- The **MESA** technique is applied to alleviate overfitting, contributing approximately +0.2% accuracy.

### COCO Text-to-Image Generation Benchmark (1024×1024)

| Model | FID (↓) | CLIP-T (↑) |
|---|---|---|
| SD-v1.5 (baseline) | 32.71 | 0.290 |
| Mamba (w/ norm) | 50.30 | 0.263 |
| Mamba2 (w/ norm) | 37.02 | 0.273 |
| Linfusion (w/ norm) | 36.33 | 0.285 |
| GSPN-1 | 30.86 | 0.307 |
| **GSPN-2** | **33.21** | **0.286** |

GSPN-2 closely approaches the SD-v1.5 baseline on COCO and substantially outperforms Mamba/Linfusion, though it falls slightly short of GSPN-1. The paper attributes this to the channel-sharing and proxy compression design, which trades a marginal quality loss for substantial inference acceleration (93× vs. 84× on 16K images).

### Variation in Optimization Gains Across Workloads

The appendix compares the contribution of each optimization under diverse workloads:

**Large batch, small channel (B=256, C=1, 1024²)**:
- Total speedup: 36.8× (143.7ms → 3.9ms)
- Coalesced Access dominates: 34.0×
- Shared memory actually **degrades** performance by 0.9× (L1 cache already sufficient)
- 2D Block yields no meaningful gain (only 1 channel, no parallelism along the y dimension)

**Large channel, small batch (B=1, C=1152, 1024²)**:
- Total speedup: **151.4×** (863.2ms → 5.7ms)
- **Channel compression contributes the most**: 7.8× alone (8× compression reduces 1152→144 channels)
- Confirms that high channel count is a critical bottleneck in GSPN-1

**Conclusion**: Coalesced Access is a foundational optimization across all configurations; channel compression is most impactful in high-channel regimes; shared memory benefits multi-channel workloads but may be detrimental in single-channel settings.

### Unexpected Finding on L1 Cache

Profiling reveals an interesting phenomenon: under certain configurations, relying on L1 cache without explicit shared memory yields performance comparable to using shared memory. The baseline implementation achieves ~35% L1 cache hit rate; enabling shared memory drops the L1 hit rate to ~0%, yet total latency remains essentially unchanged. This indicates that modern GPUs' L1 caches are already highly effective for structured access patterns. Nevertheless, the paper recommends retaining the shared memory implementation for portability and deterministic performance across GPU architectures.

## Highlights & Insights

1. **Coalesced memory access is the keystone**: A single optimization contributes 23.9× speedup — far exceeding the combined effect of all other optimizations — revealing that the fundamental bottleneck of GSPN-1 lies in memory access patterns, not the algorithm itself.
2. **Elegant design of channel sharing and proxy compression**: Grounded in the linear-attention perspective, this design unifies per-channel weights into a shared affinity matrix, simultaneously reducing parameter count and alleviating GPU concurrency pressure, with a clear theoretical justification.
3. **93% bandwidth utilization**: Nearly reaching the A100 hardware ceiling, demonstrating that the CUDA implementation is close to optimal.
4. **Resolution adaptability**: By virtue of the Stability-Context property, GSPN-2 naturally generalizes to arbitrary resolutions without the additional normalization required by Mamba or Linfusion.
5. **Robustness of proxy compression**: The aggressive 48:1 compression ($C_{\text{proxy}}=2$) incurs zero accuracy loss, confirming that spatial propagation is dominated by spatial dependencies while channel dependencies are minimal.

## Limitations & Future Work

1. **Limited gains at low BS×C**: When the product of batch size and channel count is small, SM utilization can drop to 20–30%, yielding modest optimization benefits.
2. **Absence of CLS/register token mechanism**: GSPN-2 cannot serve as a drop-in replacement for ViT architectures that rely on aggregation tokens.
3. **Insufficient validation on long video**: No evaluation is conducted on long-context video datasets.
4. **Limited validation for high-resolution dense prediction**: Experiments primarily use 480–512 pixel images; scalability advantages at higher resolutions remain to be verified.
5. **Slight generation quality regression vs. GSPN-1**: FID 33.21 vs. 30.86, CLIP-T 0.286 vs. 0.307 — the speed–quality tradeoff warrants attention.
6. **No adaptive optimization strategy**: The paper proposes dynamically switching between GSPN-1 and GSPN-2 configurations based on BS×C, but this has not been implemented.

## Related Work & Insights

| Method | Complexity | Core Primitive | Hardware Utilization | Resolution Generalization |
|---|---|---|---|---|
| Self-Attention | $O(N^2)$ | GEMM + Softmax | High (FlashAttn) | Requires positional encoding adaptation |
| FlashAttention | $O(N^2)$ | Fused tiled GEMM | ~90% | Same as above |
| Mamba/SSM | $O(N)$ | Prefix-sum scan | Moderate | Requires additional normalization |
| Linfusion | $O(N)$ | Linear attention | Moderate | Requires additional normalization |
| GSPN-1 | $O(\sqrt{N})$ | Row scan (multi-kernel) | 3–8% | **Natively supported** |
| **GSPN-2** | $O(\sqrt{N})$ | Row scan (single kernel) | **91–93%** | **Natively supported** |

GSPN-2's propagation pattern is neither matrix multiplication nor prefix scan, necessitating a dedicated CUDA implementation. The paper's contribution lies precisely in converting this theoretical advantage into practical speedup through systematic optimization.

## My Notes

- **A paradigm for algorithm-system co-design**: This paper is a canonical example of "theoretically optimal but implementation-limited." GSPN-1 already achieves optimal algorithmic complexity ($O(\sqrt{N})$), yet its CUDA implementation exploits only 3–8% of available bandwidth. GSPN-2's primary contribution is not algorithmic innovation but engineering excellence — translating theoretical advantages into real-world speedup. This algorithm-system co-design paradigm deserves broader adoption across other domains.
- **Memory access patterns determine performance**: Coalesced Access alone contributes 23.9× — the vast majority of the total 40× speedup. This reaffirms that on GPUs, computation is rarely the bottleneck; memory access patterns are. Researchers working on efficient inference or training should prioritize optimizing memory layout and access patterns above all else.
- **An unexpected finding from channel compression**: Zero accuracy loss at $C_{\text{proxy}}=2$ (48:1 compression) implies extremely high inter-channel redundancy in spatial propagation tasks. This opens new directions for downstream applications (e.g., dimensionality reduction in diffusion model U-Nets) — perhaps attention/propagation modules can compress channels far more aggressively than previously assumed.
- **Potential as a ViT attention replacement**: GSPN-2's $O(\sqrt{N})$ complexity, 93% bandwidth utilization, and resolution-agnostic design make it a highly competitive candidate for high-resolution vision models. However, the absence of CLS token support limits its applicability as a drop-in replacement in classification and retrieval tasks.
- **Complementarity with FlashAttention**: FlashAttention optimizes the constant factor of $O(N^2)$ operations, whereas GSPN-2 directly reduces complexity to $O(\sqrt{N})$. The two approaches may each hold an advantage in different resolution regimes: FlashAttention's constant overhead is smaller at low resolutions, while GSPN-2's asymptotic advantage dominates at high resolutions.
- **The speed–quality tradeoff in generation**: GSPN-2's slight regression on COCO (FID 33.21 vs. 30.86; CLIP-T 0.286 vs. 0.307) is the cost of channel sharing and proxy compression. Whether more nuanced compression strategies (e.g., adaptive $C_{\text{proxy}}$) could close this gap is worth exploring.

## Inspirations and Connections

- **Algorithm-system co-design paradigm**: This paper exemplifies deep integration between algorithmic improvement and CUDA engineering. The key insight is that translating theoretical complexity advantages into practical speedup requires careful kernel design (coalesced access, shared memory, block configuration), rather than simply reusing existing primitives.
- **Universality of channel compression**: The proxy dimension ablation suggests that in many vision tasks, spatial dependencies are far stronger than channel dependencies — an insight with broad implications for designing efficient attention mechanisms.
- **Potential for integration with diffusion models**: The 93× speedup on SDXL demonstrates that GSPN-2 is especially well-suited for high-resolution generation; future work could explore integration with purely Transformer-based diffusion architectures such as DiT.
- **Adaptive optimization strategy**: The paper notes that GSPN-1 or GSPN-2 configurations could be selected dynamically based on BS×C; this adaptive approach is worth considering in other efficient model designs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Dual innovations at both the algorithmic level (channel sharing + compressed proxy) and the system level (single kernel + coalesced access).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Detailed step-by-step profiling analysis covering both classification and generation tasks; video and dense prediction benchmarks are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The exposition progresses logically from hardware characteristics to algorithmic design to experimental analysis, with clear and coherent reasoning throughout.
- **Value**: ⭐⭐⭐⭐ — Provides a practical high-efficiency solution for high-resolution vision tasks; the 40× speedup carries genuine engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Accelerating Parallel Diffusion Model Serving with Residual Compression](accelerating_parallel_diffusion_model_serving_with_residual_compression.md)
- [\[NeurIPS 2025\] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling](dico_revitalizing_convnets_for_scalable_and_efficient_diffus.md)
- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](../../ICLR2026/image_generation/locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)

<!-- RELATED:END -->
