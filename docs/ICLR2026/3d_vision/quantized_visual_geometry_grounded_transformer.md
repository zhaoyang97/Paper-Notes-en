---
title: >-
  [Paper Note] Quantized Visual Geometry Grounded Transformer
description: >-
  [ICLR 2026][3D Vision][VGGT] To address the deployment demands of the billion-scale 3D reconstruction model VGGT, this paper proposes QuantVGGT…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "VGGT"
  - "post-training quantization"
  - "3D reconstruction"
  - "Hadamard rotation"
  - "calibration"
date: 2026-05-08
content_hash: 160addc047d863a8
---

# Quantized Visual Geometry Grounded Transformer

**Conference**: ICLR 2026
**arXiv**: [2509.21302](https://arxiv.org/abs/2509.21302)  
**Code**: [https://github.com/wlfeng0509/QuantVGGT](https://github.com/wlfeng0509/QuantVGGT)  
**Area**: 3D Vision / Model Compression
**Keywords**: VGGT, post-training quantization, 3D reconstruction, Hadamard rotation, calibration

## TL;DR
To address the deployment demands of the billion-scale 3D reconstruction model VGGT, this paper proposes QuantVGGT, the first dedicated PTQ framework for VGGT. It resolves heavy-tailed activation distributions caused by special tokens via dual-smoothed fine-grained quantization (Hadamard rotation + channel-wise smoothing), and addresses calibration instability via noise-filtered diverse sampling. At 4-bit quantization, the method achieves 3.7× memory compression and 2.5× inference speedup while retaining 98%+ accuracy.

## Background & Motivation

**Background**: VGGT is a unified 3D reconstruction model with 1.2B parameters that performs depth estimation, point map regression, camera pose prediction, and point tracking in a single forward pass. Despite its outstanding performance, its computational and memory overhead is substantial, limiting practical deployment.

**Limitations of Prior Work**: While PTQ is well-established for LLMs and 2D vision models, applying it to VGGT introduces two unique challenges: (1) data-independent special tokens (camera/register tokens) induce extreme heavy-tailed activation distributions; (2) the semantic complexity of 3D multi-view data makes calibration sample selection highly unstable.

**Key Challenge**: Special tokens are a critical design element for VGGT's multi-task reasoning, yet the distributional gap between these tokens and regular image tokens causes quantization bits to be wasted on outliers.

**Goal**: Design a VGGT-specific PTQ scheme that preserves reconstruction accuracy under low-bit quantization.

**Key Insight**: Distribution analysis reveals that special tokens are the root cause of heavy tails, and inter-frame relationships in multi-view data are the key structural factor for calibration.

**Core Idea**: Global Hadamard rotation disperses the spikes from special tokens, followed by local channel-wise smoothing to reduce residual variance after rotation, combined with frame-aware diverse sampling to construct a robust calibration set.

## Method

### Overall Architecture
QuantVGGT comprises two core components: (1) **DSFQ** (Dual-Smoothed Fine-Grained Quantization) — applies global Hadamard rotation to smooth heavy-tailed distributions, followed by local channel-wise scaling to reduce inter-channel variance, with fine-grained quantization granularity; (2) **NFDS** (Noise-Filtered Diverse Sampling) — filters anomalous samples using deep-layer activation statistics, and constructs a diverse calibration set via frame-aware correlation clustering.

### Key Designs

1. **Pre-Global Rotation (Global Hadamard Rotation)**:

    - **Function**: Disperses activation spikes caused by special tokens.
    - **Mechanism**: Simultaneously left-multiplies both activation $\mathbf{X}$ and weight $\mathbf{W}$ by a random Hadamard matrix $\mathbf{H}$, leveraging the central limit effect to approximate the heavy-tailed distribution as Gaussian. $\mathbf{XW}^\top = (\mathbf{XH})(\mathbf{WH})^\top$
    - **Design Motivation**: The Hadamard transform uniformly redistributes extreme values concentrated in a few channels across all channels.

2. **Post-Local Smooth (Local Channel-wise Smoothing)**:

    - **Function**: Reduces residual inter-channel variance after rotation.
    - **Mechanism**: Computes scaling factors in the rotated space as $\hat{c}_i = \frac{\max(|\mathbf{X}_i\mathbf{H}|)^\alpha}{\max(|\mathbf{W}_i\mathbf{H}|)^{1-\alpha}}$, with $\alpha=0.5$.
    - **Design Motivation**: Rotation only disperses global spikes without eliminating local inter-channel discrepancies. Applying smoothing after rotation is more stable than the reverse order, as prior smoothing would be disrupted by the subsequent rotation.

3. **Fine-Grained Quantization Granularity**:

    - **Function**: Reduces quantization error by decreasing quantization granularity.
    - **Mechanism**: Weights are quantized along the $d_{out}$ dimension; activations are quantized along the token dimension, exploiting the fact that matrix multiplication accumulates only along $d_{in}$.
    - **Design Motivation**: $\mu$-coherence theory shows that finer-grained quantization significantly reduces quantization difficulty.

4. **Noise-Filtered Diverse Sampling (NFDS)**:

    - **Function**: Constructs a robust calibration dataset.
    - **Mechanism**: A two-stage pipeline — (a) computes a noise score for each sample from deep-layer activation statistics (L2 norm of z-scores of mean and variance), filtering out high-scoring anomalous samples; (b) leverages VGGT's inter-frame correlations (normalized similarity vectors $c_t^i$ between the first frame and subsequent frames) for K-means clustering, then uniformly samples from clusters to form the calibration set.
    - **Design Motivation**: Theorem 3.2 proves that calibration sets should sample sub-regions of the data space proportionally; inter-frame relationships are central to VGGT's inductive bias.

## Key Experimental Results

### Main Results (Camera Pose Estimation on CO3Dv2)

| Configuration | W/A bit | Accuracy Retention | Memory Compression | Speedup |
|---|---|---|---|---|
| Full FP16 | 16/16 | 100% | 1× | 1× |
| W8A8 QuantVGGT | 8/8 | ~99% | 2× | 1.5× |
| W4A4 QuantVGGT | 4/4 | ~98% | 3.7× | 2.5× |
| W4A4 SmoothQuant | 4/4 | ~85% | 3.7× | 2.5× |
| W4A4 QuaRot | 4/4 | ~90% | 3.7× | 2.5× |

### Ablation Study

| Component | Accuracy Change | Notes |
|---|---|---|
| Hadamard rotation only | +5% vs. naive | Disperses spikes |
| + Channel-wise smoothing | +3% | Reduces residual variance |
| + Fine-grained quantization | +2% | Finer quantization granularity |
| + NFDS | +2% | Robust calibration |
| Full QuantVGGT | 98% of FP | All components combined |

### Key Findings
- **Special tokens are the primary obstacle to quantization**: The activations of the first 5 tokens (camera + register) are more than 10× larger in magnitude than regular patch tokens.
- **The rotation → smoothing order matters**: Applying smoothing before rotation undermines the benefits of smoothing; rotating first to homogenize the distribution and then smoothing yields greater stability.
- **Frame-aware clustering outperforms label-based clustering**: t-SNE visualizations show that semantic labels for 3D scenes fail to effectively distinguish calibration sub-domains, whereas inter-frame relationships can.
- **4-bit quantization is practically viable**: Empirical measurements on an RTX 4090 confirm a 2.5× inference speedup.

## Highlights & Insights
- **First quantization work for a billion-scale 3D model**: Fills a gap in quantization research within the 3D reconstruction domain.
- **"Global-then-local" dual-smoothing design**: Elegantly resolves the heavy-tail problem in two steps with no additional runtime overhead, as scaling factors can be folded into LayerNorm.
- **Frame-aware calibration in NFDS**: Exploits VGGT's unique inductive bias of "first frame vs. subsequent frames," embodying the principle that deeper model understanding enables better compression.
- **Theoretical contribution of Theorem 3.2**: Provides a formal guiding principle for calibration set construction.

## Limitations & Future Work
- Evaluated only on VGGT; generalizability to other 3D models such as DUSt3R/MASt3R remains unverified.
- A 2% accuracy drop at 4-bit may be insufficient for high-precision applications.
- The noise threshold and cluster count in NFDS require manual tuning.
- Extremely low-bit regimes such as INT2/INT3 are not explored.

## Related Work & Insights
- **vs. SmoothQuant**: Applies only global smoothing without accounting for the influence of VGGT's special tokens.
- **vs. QuaRot**: Applies only Hadamard rotation without subsequent local channel-wise smoothing.
- The proposed approach also offers insights for quantizing other large models containing special tokens, such as [CLS] tokens in VLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ Individual components are not entirely new (Hadamard rotation, SmoothQuant), but their combination and analysis tailored to VGGT are original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark, multi-bit-width evaluation with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis is thorough and visualizations are clear.
- Value: ⭐⭐⭐⭐ First quantization work for a large-scale 3D model with strong practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](../../CVPR2026/3d_vision/ggpt_geometry_grounded_point_transformer.md)
- [\[CVPR 2026\] MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation](../../CVPR2026/3d_vision/mvggt_multimodal_visual_geometry_grounded_transformer_for_multiview_3d_referring.md)
- [\[CVPR 2026\] Fast Spatial Tracking with Visual Geometry Transformer](../../CVPR2026/3d_vision/fast_spatial_tracking_with_visual_geometry_transformer.md)
- [\[ICLR 2026\] NOVA3R: Non-pixel-aligned Visual Transformer for Amodal 3D Reconstruction](nova3r_non-pixel-aligned_visual_transformer_for_amodal_3d_reconstruction.md)
- [\[ICLR 2026\] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)

</div>

<!-- RELATED:END -->
