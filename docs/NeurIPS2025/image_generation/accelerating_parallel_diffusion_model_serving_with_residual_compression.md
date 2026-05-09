---
title: >-
  [Paper Note] Accelerating Parallel Diffusion Model Serving with Residual Compression
description: >-
  [NeurIPS 2025][Image Generation][parallel inference] This paper proposes CompactFusion, a framework that eliminates communication redundancy in parallel diffusion inference via residual compression—transmitting only the activation differences between adjacent denoising steps rather than full activations. It achieves a 3.0× speedup on 4×L20 GPUs with generation quality significantly superior to DistriFusion, a 6.7× speedup under simulated Ethernet bandwidth, and maintains better quality than DistriFusion even at 100× compression.
tags:
  - NeurIPS 2025
  - Image Generation
  - parallel inference
  - communication compression
  - residual compression
  - diffusion model serving
  - sequence parallelism
date: 2026-05-08
content_hash: 9a09a826e0f01371
---

# Accelerating Parallel Diffusion Model Serving with Residual Compression

**Conference**: NeurIPS 2025
**arXiv**: [2507.17511](https://arxiv.org/abs/2507.17511)
**Code**: [GitHub](https://github.com/Cobalt-27/CompactFusion)
**Area**: Diffusion Models / System Optimization / Model Serving
**Keywords**: parallel inference, communication compression, residual compression, diffusion model serving, sequence parallelism

## TL;DR
This paper proposes CompactFusion, a framework that eliminates communication redundancy in parallel diffusion inference via residual compression—transmitting only the activation differences between adjacent denoising steps rather than full activations. It achieves a 3.0× speedup on 4×L20 GPUs with generation quality significantly superior to DistriFusion, a 6.7× speedup under simulated Ethernet bandwidth, and maintains better quality than DistriFusion even at 100× compression.

## Background & Motivation

**Background**: Diffusion models (e.g., FLUX.1 with 12B parameters) are scaling rapidly, making real-time inference on a single GPU infeasible. Multi-device parallel inference has become a necessity, and the dominant parallelism strategies (Sequence Parallel, Patch Parallel) require exchanging large activation tensors across devices.

**Limitations of Prior Work**: (a) Interconnect bandwidth growth lags far behind compute growth (A100→H100: 6× compute increase vs. only 1.5× NVLink bandwidth increase), making communication a bottleneck; (b) Standard patch parallelism for FLUX.1 requires transmitting ~60 GB of activations per image per GPU, consuming over 45% of inference time on PCIe; (c) Existing methods (DistriFusion, PipeFusion) use "displaced parallelism" to overlap communication with computation by reusing stale activations from the previous step, but suffer from notable quality degradation, undiminished data volume, and integration complexity.

**Key Challenge**: Activations in adjacent diffusion steps are highly similar (temporal redundancy), yet existing methods still transmit full activations—masking redundancy through overlapping rather than eliminating it.

**Goal**
- How to genuinely reduce communication volume rather than merely hiding communication latency?
- How to maintain high generation quality under aggressive compression?

**Key Insight**: Since the activation differences (residuals) between adjacent steps are far smaller than the activations themselves, compressing residuals achieves higher compression ratios with lower error than compressing full activations—eliminating redundancy rather than concealing it.

**Core Idea**: Transmit the "activation delta" (residual) rather than the full activation. The small magnitude of residuals leads to low compression error, enabling aggressive compression exceeding 100× while preserving quality.

## Method

### Overall Architecture
CompactFusion inserts a residual compression module at the communication layer of parallel diffusion inference. At each denoising step, instead of transmitting the full activation $a_t$, each device computes the residual $\Delta_t = a_t - \hat{a}_{t-1}$ (difference from the previous step's reconstructed value), compresses and transmits it, and the receiver reconstructs $\hat{a}_t = \hat{a}_{t-1} + C(\Delta_t)$. Error feedback is incorporated to prevent error accumulation. The entire framework is decoupled from model logic and parallelism strategy, modifying only the communication primitives.

### Key Designs

1. **Residual Compression**

    - **Function**: Transmit compressed inter-step residuals instead of full activations.
    - **Mechanism**: Temporal redundancy in diffusion models implies $\|\Delta_t\| \ll \|a_t\|$; compressing a low-magnitude signal incurs proportionally lower distortion. Empirical validation shows that 1-bit binarization of full activations causes complete image collapse, whereas compressing residuals produces clean reconstructions.
    - **Design Motivation**: Eliminating redundancy is more fundamental than masking it. DistriFusion overlaps communication using stale activations without reducing data volume; CompactFusion directly reduces data volume to less than 1%.

2. **Error Feedback**

    - **Function**: Prevent compression errors from accumulating across steps.
    - **Mechanism**: After each compression step, the untransmitted residual error $e_t = \Delta_t - C(\Delta_t)$ is stored locally and added to the residual of the next step before compression. This "recycles" the error rather than discarding it, preventing the reconstructed state from drifting from the true trajectory. Theoretical analysis (Proposition 3.1) proves that the steady-state error bound of residual compression with error feedback is substantially smaller than naive compression: $v^{\text{residual}}/v^{\text{naive}} = (\sigma_\Delta^2 / \sigma_a^2) \cdot \text{ratio} \ll 1$.
    - **Design Motivation**: Without error feedback, total error grows linearly with the number of steps; with feedback, it converges to a bounded steady state, making the method reliable over 28–50 inference steps.

3. **Low-Rank Strategy for Extreme Compression Ratios**

    - **Function**: Maintain quality at compression ratios exceeding 100×.
    - **Mechanism**: Quantization saturates at 1-bit (16×), and sparsification collapses in the diffusion setting (at 100× sparsity, most values are never updated). Low-rank approximation $X \approx UV^T$ covers all coordinates while substantially reducing communication volume. Subspace iteration replaces SVD (60× faster to meet the ~1ms compression budget). Key insight: diffusion residuals are high-rank, but each transmission is limited to a low-rank subspace—a "high-rank/low-rank mismatch" bottleneck. Solution: quantize the low-rank matrices with INT4 to increase the usable rank (covering more directions) within the same bandwidth budget, trading per-direction precision for directional coverage.
    - **Design Motivation**: Experiments confirm that directional coverage matters more than per-step accuracy—expanding rank coverage (via INT4 quantization) yields better generation quality than improving approximation quality (via more iterations).

4. **System Co-design**

    - Optimized GPU compression kernels: N:M block sparsifier (avoiding the sorting and irregular memory access of TopK)
    - Latency hiding: compression and communication execute in parallel
    - Easy integration: wraps standard communication primitives without modifying model code or the parallel pipeline; core implementation is fewer than 20 lines

### Loss & Training
No training is required—CompactFusion is a purely inference-time compression scheme applied directly to off-the-shelf models.

## Key Experimental Results

### Main Results: FLUX.1-dev on 4×L20 (PCIe)

| Method | Latency | FID↓ | LPIPS↓ |
|--------|---------|------|--------|
| Sequence Parallel (no compression) | 10.89s | baseline | baseline |
| DistriFusion | 8.05s | 9.91 | 0.310 |
| PipeFusion | 9.49s | 6.72 | 0.250 |
| **Compact-1bit (16×)** | **7.46s** | **7.08** | **0.260** |
| **Compact-2bit (8×)** | **7.57s** | **3.26** | **0.114** |
| **Compact-Lowrank (100×)** | 10.60s | 8.68 | 0.275 |

Compact-2bit achieves FID of 3.26 and LPIPS of 0.114, substantially outperforming DistriFusion (9.91 / 0.310).

### Extreme Compression: 100× Still Outperforms DistriFusion

| Compression Method | Ratio | FID↓ |
|-------------------|-------|------|
| DistriFusion | 1× (stale activations) | 9.91 |
| **Compact-Lowrank** | **100.05×** | **8.68** |

Transmitting less than 1% of the original data still yields higher quality than transmitting all stale activations.

### Cross-Hardware / Network Conditions

| Hardware | Method | Speedup |
|----------|--------|---------|
| 4×H20 (NVLink) | CompactFusion | 3.0× |
| 4×L20 (PCIe) | CompactFusion | 3.0× |
| Simulated Ethernet | CompactFusion vs. DistriFusion | **6.7×** |

### Key Findings
- **Residual compression vs. displaced parallelism**: The residual approach comprehensively outperforms DistriFusion/PipeFusion on all metrics—lower latency and higher quality—by avoiding stale activations.
- **Directional coverage > per-step accuracy**: INT4-quantized low-rank matrices (expanded rank) outperform additional subspace iterations (improved accuracy).
- **Communication-intensive strategies become viable**: CompactFusion enables Sequence Parallel to operate efficiently on low-bandwidth networks where DistriFusion fails entirely.
- **Effective on video models**: The method is also validated on CogVideoX-2b.
- **Extreme robustness**: Quality degrades gracefully from 2× to 100×+ compression ratios.

## Highlights & Insights
- The paradigm shift from **"masking redundancy" to "eliminating redundancy"** is the paper's most fundamental insight: DistriFusion attempts to hide an unchanged communication volume via overlapping; CompactFusion directly reduces that volume.
- **Error feedback** resolves the error accumulation problem inherent to residual compression, with both theoretical guarantees and empirical validation. Its transfer from gradient compression to inference-time activation compression is elegant and natural.
- The **"trade precision for coverage"** low-rank strategy is a compelling practical insight: under extreme compression, covering more update directions matters more than precisely approximating fewer directions.
- **Exemplary engineering decoupling**: fewer than 20 lines of core code, multi-framework compatibility, no modifications to model logic—a standard of elegance that systems work should aspire to.

## Limitations & Future Work
- **Warmup required at the first step**: No historical activations exist at step one, requiring full-data transmission (acceptable but non-zero overhead).
- **Extreme low rank may fail at specific steps**: Steps with large changes in noise level (e.g., the initial few steps) may produce larger residuals.
- **Validation limited to Sequence Parallel**: Although generality is claimed, experiments are conducted primarily under SP; Tensor Parallel and other strategies are not validated.
- **Low-rank subspace iteration quality**: Sub-optimal approximation may produce visible degradation in high-frequency details under certain conditions.

## Related Work & Insights
- **vs. DistriFusion**: DistriFusion overlaps computation using stale activations, resulting in quality degradation without reducing data volume; CompactFusion compresses residuals, achieving high quality while reducing data volume to less than 1%.
- **vs. PipeFusion**: PipeFusion is a displaced-parallelism scheme for pipeline parallelism, orthogonal to and composable with the proposed method.
- **vs. gradient compression (PowerSGD, DeepGradComp)**: This work successfully transfers residual compression and error feedback from gradient compression to inference-time activation compression—a compelling example of cross-domain technique transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ — The core idea of residual compression is concise and powerful; though rooted in gradient compression, this is its first application to inference acceleration.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-model (image + video) × multi-hardware (NVLink / PCIe / Ethernet) × multi-compression-ratio × theoretical analysis × human evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, polished figures, seamless integration of theory and experiments.
- Value: ⭐⭐⭐⭐⭐ — Directly deployable in production, delivering 3× speedup with quality improvements, open-sourced, with substantial practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration](tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)
- [\[NeurIPS 2025\] GSPN-2: Efficient Parallel Sequence Modeling](gspn-2_efficient_parallel_sequence_modeling.md)
- [\[CVPR 2026\] CoD: A Diffusion Foundation Model for Image Compression](../../CVPR2026/image_generation/cod_a_diffusion_foundation_model_for_image_compression.md)
- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](../../AAAI2026/image_generation/specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)
- [\[ICCV 2025\] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal](../../ICCV2025/image_generation/compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)

</div>

<!-- RELATED:END -->
