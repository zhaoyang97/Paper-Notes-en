---
title: >-
  [Paper Note] Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution
description: >-
  [ICLR 2026][Image Restoration][Reference-based Super-Resolution] This paper proposes Ada-RefSR, a single-step reference-guided diffusion super-resolution framework based on the "Trust but Verify" principle. It introduces…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "Reference-based Super-Resolution"
  - "diffusion model"
  - "Adaptive Gating"
  - "Implicit Correlation"
  - "Single-step Diffusion"
date: 2026-05-08
content_hash: e955f489c6adc475
---

# Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution

**Conference**: ICLR 2026
**arXiv**: [2602.01864](https://arxiv.org/abs/2602.01864)
**Code**: [https://github.com/vivoCameraResearch/AdaRefSR](https://github.com/vivoCameraResearch/AdaRefSR)
**Area**: Image Restoration
**Keywords**: Reference-based Super-Resolution, diffusion model, Adaptive Gating, Implicit Correlation, Single-step Diffusion

## TL;DR

This paper proposes Ada-RefSR, a single-step reference-guided diffusion super-resolution framework based on the "Trust but Verify" principle. It introduces an Adaptive Implicit Correlation Gating (AICG) mechanism that maximally exploits reliable reference information while suppressing erroneous fusion, incurring only 0.13% additional computational overhead.

## Background & Motivation

Diffusion-based single image super-resolution (SISR) methods (e.g., StableSR, DiffBIR, SeeSR) leverage generative priors to produce visually appealing results, yet they commonly suffer from **hallucination**—fabricating or omitting fine details. Reference-based super-resolution (RefSR) mitigates hallucination by incorporating external reference images to supply complementary high-frequency details.

The core challenge lies in the fact that real-world degradations render the correspondence between low-quality (LQ) inputs and reference (Ref) images unreliable. Existing methods exhibit critical deficiencies in controlling reference utilization:

**PFStorer (global learnable vector)**: Employs a global weight to uniformly control the reference branch, failing to adapt to input pairs of varying alignment quality—pairs with good and poor alignment share identical gating values.

**ReFIR (explicit token similarity)**: Spatial gating based on explicit correlations is susceptible to noise and suffers from long-tail distribution issues, where a majority of identical tokens dominate computation while a minority of critical tokens are neglected.

**Dual failure modes**:
- **Over-reliance on reference**: Incorrectly injects reference cues, causing semantic inconsistency (e.g., bird eyes being copied to non-eye regions).
- **Under-utilization of reference**: Valuable reference information is insufficiently exploited.

## Method

### Overall Architecture

Ada-RefSR comprises two core components:
- **Trust stage**: Maximizes utilization of reference features via Reference Attention (RA).
- **Verify stage**: Adaptively modulates reference contributions via AICG to suppress erroneous fusion.

The framework is built upon the S3Diff single-step diffusion SR backbone. Only the newly introduced reference attention modules are trained; all other components are frozen.

### Key Designs

1. **Trust: Direct Reference Feature Injection**

    - A ReferenceNet (initialized from SD-Turbo, fixed timestep=1) is used to extract multi-scale reference features.
    - Reference Attention (RA) module:
        - $\mathbf{Q} = \mathbf{H}_{src}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{H}_{ref}\mathbf{W}_K$, $\mathbf{V} = \mathbf{H}_{ref}\mathbf{W}_V$
        - $\mathbf{H}_{out} = \text{ZeroLinear}(\text{Softmax}(\frac{\mathbf{QK}^\top}{\sqrt{d}})\mathbf{V}) + \mathbf{H}_{src}$
    - RA weights are initialized by copying from the backbone self-attention; ZeroLinear stabilizes early training.
    - **Design Motivation**: No pre-filtering is applied to ensure all potential LQ-Ref matches are captured.
    - **Limitation**: Indiscriminate fusion may introduce local semantic inconsistencies.

2. **Verify: Adaptive Implicit Correlation Gating (AICG)**

    - **Reference Summary Tokens**: Introduces learnable summary tokens $\mathbf{T}_S \in \mathbb{R}^{M \times d}$ ($M=16$) to compactly summarize reference features:
        - $\mathbf{S} = \mathbf{T}_S \mathbf{W}_K$
        - $\mathbf{K}_{sum} = \text{Softmax}(\frac{\mathbf{SK}^\top}{\sqrt{d}})\mathbf{K} \in \mathbb{R}^{M \times d}$
    - **Implicit Correlation Gating**:
        - $\mathbf{S}_{map} = \text{Softmax}(\frac{\mathbf{Q}\mathbf{K}_{sum}^\top}{\sqrt{d}}) \in \mathbb{R}^{L_q \times M}$
        - $\mathbf{G} = \sigma(\frac{1}{M}\sum_{j=1}^{M}[\mathbf{S}_{map}]_{:,j}) \in \mathbb{R}^{L_q \times 1}$
    - **Gated Modulation**: $\mathbf{H}_{out} = \text{ZeroLinear}(\mathbf{G} \odot \text{RA}(\mathbf{H}_{src}, \mathbf{H}_{ref})) + \mathbf{H}_{src}$
    - **Design Motivation**: Implicit modeling replaces explicit token-to-token similarity, thereby avoiding noise sensitivity.
    - **Key Advantage**: Reuses existing projections and intermediate variables within the RA module, making it extremely lightweight.

3. **Comparison of Core Innovations**

    - PFStorer: Global gating without considering LQ-Ref correlation → unable to adapt to varying alignment quality.
    - ReFIR: Explicit $L_{src} \times L_{ref}$ similarity matrix → high computational cost (+16%) and noise-prone.
    - AICG (Ours): Implicit estimation via $M=16$ summary tokens → only +0.13% overhead, robust.

### Loss & Training

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{rec} + \lambda_2 \mathcal{L}_{per} + \lambda_3 \mathcal{L}_{adv}$$

- $\mathcal{L}_{rec}$: L2 reconstruction loss
- $\mathcal{L}_{per}$: VGG perceptual loss
- $\mathcal{L}_{adv}$: Standard GAN adversarial loss
- Training: 2× A40 GPUs, Adam optimizer, learning rate 5e-5, batch size 16, 11K iterations.
- Data augmentation: 20% of HQ-Ref pairs are randomly replaced with unrelated samples to enhance robustness.

## Key Experimental Results

### Main Results (Table 1)

**WRSR Dataset** (scene-level reference SR):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|--------|-------|-------|--------|------|
| S3Diff | 21.91 | 0.562 | 0.354 | 63.82 |
| SeeSR+ReFIR | 21.83 | 0.567 | 0.344 | 61.96 |
| SUPIR+ReFIR | 21.01 | 0.538 | 0.395 | 76.50 |
| **Ours** | **21.97** | **0.578** | **0.306** | **53.28** |

**Bird Dataset** (retrieval-augmented SR):

| Method | PSNR↑ | LPIPS↓ | FID↓ |
|--------|-------|--------|------|
| S3Diff | 24.84 | 0.290 | 60.95 |
| SeeSR+ReFIR | 23.72 | 0.295 | 52.40 |
| **Ours** | **25.30** | **0.254** | **36.42** |

**Face Dataset** (face reference SR):

| Method | PSNR↑ | LPIPS↓ | FID↓ |
|--------|-------|--------|------|
| DMDNet | 26.90 | 0.232 | 56.63 |
| InstantRestore | 26.22 | 0.207 | 51.23 |
| **Ours** | **27.13** | **0.175** | **42.70** |

### Ablation Study (Table 2 — WRSR & Face Datasets)

| Gating Scheme | WRSR PSNR | WRSR SSIM | Face PSNR | Face SSIM |
|---------------|-----------|-----------|-----------|-----------|
| Vanilla (no gating) | 21.95 | 0.574 | 27.08 | 0.750 |
| Global (PFStorer) | 21.63 | 0.561 | 27.06 | 0.750 |
| ReFIR | 21.78 | 0.567 | 26.94 | 0.747 |
| **AICG (Ours)** | **21.97** | **0.578** | **27.13** | **0.752** |

### Efficiency Comparison (Table 3 — 1024×1024)

| Method | VRAM (GB) | Inference Time (s) |
|--------|-----------|--------------------|
| S3Diff | 7.18 | 0.74 |
| Ada-RefSR | 15.54 | 1.35 |
| SeeSR+ReFIR | 18.95 | **40.23** |

- **Ada-RefSR is approximately 30× faster than SeeSR+ReFIR.**

### Key Findings

1. **AICG consistently outperforms all other gating schemes across datasets**, validating the effectiveness of implicit correlation modeling.
2. **Robustness advantage**: When the LQ-Ref alignment ratio is below 0.7, SUPIR+ReFIR falls below its baseline, whereas the proposed method consistently surpasses its baseline.
3. **Interpretability of summary tokens**: Different tokens capture distinct semantic regions (body parts, sky, grass, bird features, etc.).
4. **$M=16$ summary tokens is optimal**: Both $M=8$ and $M=32$ yield inferior performance.

## Highlights & Insights

- **"Trust but Verify" design philosophy**: First trust (maximize reference utilization), then verify (suppress erroneous fusion)—a logically coherent two-stage strategy.
- **Extreme lightweight design**: AICG incurs only 0.13% additional computational overhead (vs. 16% for ReFIR) by reusing existing projections.
- **Implicit vs. explicit correlation**: Implicit modeling via summary tokens avoids the noise sensitivity inherent in explicit token-to-token similarity computation.
- **Semantic clustering of learnable summary tokens**: Tokens self-organize into semantically meaningful cluster centers.
- **30× speedup**: The single-step diffusion design enables inference speeds far exceeding multi-step methods.

## Limitations & Future Work

1. Model parameter count is approximately twice that of S3Diff (2679M vs. 1327M) due to the ReferenceNet.
2. Handling of extremely irrelevant references leaves room for improvement.
3. The current framework utilizes only a single reference image; extension to multiple references remains unexplored.
4. Patch-level reference guidance may enable finer-grained control.
5. Lighter-weight reference injection strategies (e.g., token pruning, sparse attention) warrant further exploration.

## Related Work & Insights

- **S3Diff**: Single-step diffusion model serving as the SR backbone.
- **ReFIR**: Current state-of-the-art retrieval-augmented RefSR, but with limitations in explicit correlation gating.
- **IP-Adapter / ControlNet**: Alternative reference injection paradigms, but insufficiently fine-grained for the RefSR setting.
- **DETR learnable queries**: Source of inspiration for AICG summary tokens, though the functional roles differ entirely.
- **Insight**: The implicit correlation modeling approach is generalizable to other conditional generation tasks requiring adaptive conditioning (e.g., video editing, virtual try-on).

## Rating

- Novelty: ⭐⭐⭐⭐ (The AICG gating mechanism is novel; the Trust-but-Verify paradigm is clearly articulated.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets, multi-dimensional ablation, efficiency analysis, robustness evaluation, complexity derivation.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, polished figures, complete mathematical derivations.)
- Value: ⭐⭐⭐⭐ (A practical advancement in RefSR; the 30× speedup carries meaningful real-world significance.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](../../CVPR2026/image_restoration/disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](../../CVPR2026/image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](../../CVPR2026/image_restoration/bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[AAAI 2026\] SpatioTemporal Difference Network for Video Depth Super-Resolution](../../AAAI2026/image_restoration/spatiotemporal_difference_network_for_video_depth_super-resolution.md)

</div>

<!-- RELATED:END -->
