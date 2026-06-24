---
title: >-
  [Paper Note] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] This paper proposes Mobile-GS, which achieves 116 FPS real-time Gaussian Splatting rendering on a Snapdragon 8 Gen 3 mobile GPU for the first time, with only 4.6MB of storage and visual quality comparable to the original 3DGS. This is accomplished through depth-aware order-independent rendering (eliminating sorting bottlenecks), neural view-dependent enhancement, first-order SH distillation, neural vector quantization…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Real-time Rendering"
  - "Mobile Deployment"
  - "Order-independent Rendering"
  - "Model Compression"
date: 2026-05-08
content_hash: d997c7cdbfb68a42
---

# Mobile-GS: Real-time Gaussian Splatting for Mobile Devices

**Conference**: CVPR 2025  
**arXiv**: [2603.11531](https://arxiv.org/abs/2603.11531)  
**Code**: [https://xiaobiaodu.github.io/mobile-gs-project/](https://xiaobiaodu.github.io/mobile-gs-project/)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Real-time Rendering, Mobile Deployment, Order-independent Rendering, Model Compression

## TL;DR
This paper proposes Mobile-GS, which achieves 116 FPS real-time Gaussian Splatting rendering on a Snapdragon 8 Gen 3 mobile GPU for the first time, with only 4.6MB of storage and visual quality comparable to the original 3DGS. This is accomplished through depth-aware order-independent rendering (eliminating sorting bottlenecks), neural view-dependent enhancement, first-order SH distillation, neural vector quantization, and contribution-based pruning.

## Background & Motivation
**Background**: 3DGS achieves high-quality novel view synthesis, but its high computational and storage demands make it difficult to run in real-time on resource-constrained mobile devices such as smartphones and AR glasses.

**Limitations of Prior Work**: Alpha blending requires sorting Gaussians by depth, which is the primary computational bottleneck (accounting for a significant portion of inference time). Additionally, 3DGS uses third-order SH, storing a massive number of parameters. Existing lightweight methods (such as Scaffold-GS and Mini-Splatting) still rely on sorting and cannot overcome this speed bottleneck.

**Key Challenge**: Sorting is a prerequisite for correct blending but also acts as the primary speed bottleneck—removing sorting can significantly accelerate rendering but introduces transparency artifacts.

**Goal**: (1) Eliminate sorting bottlenecks to achieve order-independent rendering, (2) compensate for the quality loss caused by order-independent rendering, and (3) compress storage to adapt to mobile devices.

**Key Insight**: A depth-aware weight function can implicitly simulate near-to-far sorting effects, and a view-dependent MLP can compensate for transparency artifacts.

**Core Idea**: Replace sorting with depth-aware weights, compensate using neural networks, and apply multi-dimensional compression to achieve real-time 3DGS on mobile devices.

## Method

### Overall Architecture
A first-order SH student model is distilled from a pre-trained teacher model (Mini-Splatting). During training, depth-aware order-independent rendering and a view-dependent enhancement MLP are learned simultaneously. After training, vector quantization and contribution-based pruning are applied for compression. During inference, the sorting step is completely eliminated, and the model is deployed on mobile devices using Vulkan 2.0.

### Key Designs

1. **Depth-aware Order-independent Rendering**:

    - **Function**: Eliminates the sorting dependency of alpha blending, replacing it with a weighted sum.
    - **Mechanism**: The pixel color is represented as $\mathbf{C} = (1-T)\frac{\sum_i \mathbf{c}_i \alpha_i w_i}{\sum_i \alpha_i w_i} + T\mathbf{c}_{bg}$, where $w_i = \phi_i^2 + \frac{\phi_i}{d_i^2} + \exp(\frac{s_{max}}{d_i})$. Since both the numerator and denominator are summations, they are order-independent. $T = \prod(1-\alpha_j)$ is used to distinguish the foreground from the background.
    - **Design Motivation**: Sorting is an $O(N \log N)$ operation that dominates inference time. In order-independent weighted summation, the inverse depth term naturally suppresses the contributions of distant Gaussians, while the scale terms increase the weights of large Gaussians.

2. **Neural View-dependent Enhancement**:

    - **Function**: Predicts view-dependent opacity and weights using an MLP to compensate for transparency artifacts caused by order-independent rendering.
    - **Mechanism**: Let $\mathbf{F} = \text{MLP}_f(\mathbf{P}_i, s_i, r_i, Y_i)$, $\phi_i = \text{ReLU}(\text{MLP}_\phi(\mathbf{F}))$, and $o_i = \sigma(\text{MLP}_o(\mathbf{F}))$. The inputs include the camera-to-Gaussian direction vector, scale, rotation, and SH coefficients.
    - **Design Motivation**: Order-independent blending tends to cause semi-transparency in spatially overlapping regions. View-dependent opacity can dynamically suppress the transparency of occluded regions.

3. **First-order SH Distillation + Neural Vector Quantization + Contribution Pruning**:

    - **SH Distillation**: Distills third-order (48 coefficients) representations into first-order (12 coefficients) student counterparts, supervised by color distillation loss and scale-invariant depth distortion loss, significantly reducing parameters.
    - **Neural Vector Quantization**: Employs K-Means grouping, multi-codebook sub-vector quantization, and Huffman coding. SH features are further decomposed into diffuse and view-dependent components, which are decoded by a lightweight MLP.
    - **Contribution Pruning**: Jointly evaluates opacity and scale quantiles to filter out low-contribution Gaussians. Gaussians are permanently pruned after cumulative votes reach a predefined threshold.

### Loss & Training
Using Mini-Splatting as the teacher model, the training process lasts for 60K iterations, with vector quantization initiated at 35K. The loss function is defined as $\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_{distill} + 0.1\mathcal{L}_{depth}$.

## Key Experimental Results

### Main Results (Mip-NeRF 360 Dataset, RTX 3090)

| Method | PSNR↑ | Storage↓ | FPS↑ |
|------|-------|---------|------|
| 3DGS | 27.21 | 839.9 MB | 174 |
| SortFreeGS | 27.02 | 851.4 MB | 731 |
| LocoGS-S | 27.02 | 8.5 MB | 292 |
| C3DGS | 27.03 | 30.6 MB | 184 |
| **Mobile-GS** | **27.12** | **4.6 MB** | **1125** |

Mobile devices (Snapdragon 8 Gen 3): Mobile-GS achieves 127 FPS, compared to SortFreeGS at 24 FPS and 3DGS at 8 FPS.

### Ablation Study

| Configuration | PSNR↑ | FPS↑ | Storage↓ |
|------|-------|------|---------|
| Full model | 27.12 | 1125 | 4.6 MB |
| w/o Order-independent Rendering | 27.26 | 684 | 4.5 MB |
| w/o View-dependent Enhancement | 26.68 | 1227 | 4.4 MB |
| w/o Neural Quantization | 27.33 | 841 | 121 MB |
| 0th-order SH Distillation | 27.04 | 1219 | 3.6 MB |
| 2nd-order SH Distillation | 27.13 | 917 | 7.3 MB |

### Key Findings
- Eliminating sorting contributes the most to the speed improvement (from 684 to 1125 FPS), but requires view-dependent enhancement to compensate for quality (without it, the PSNR drops by 0.44 dB).
- First-order SH is the optimal trade-off point between accuracy and efficiency: 0th-order loses 0.08 dB but runs faster, while 2nd-order brings only a 0.01 dB improvement but is 20% slower.
- Neural vector quantization compresses the storage from 121 MB to 4.6 MB ($26\times$ compression) with only a 0.21 dB impact on PSNR.
- Mobile-GS achieves 116 FPS on a mobile device in the Bicycle scene (at $1600 \times 1063$ resolution), representing the first truly real-time mobile 3DGS.

## Highlights & Insights
- **Identifying and completely eliminating sorting as the computational bottleneck**: Instead of merely optimizing the sorting algorithm, this work redesigns the rendering equation using weighted summation. This approach is highly bold and innovative.
- **A three-pronged compression strategy of distillation, quantization, and pruning**: The model size is compressed from 840 MB to 4.6 MB ($182\times$ compression) with almost no quality loss.
- **Vulkan 2.0 deployment** provides a valuable engineering reference for running 3DGS on mobile devices.

## Limitations & Future Work
- Order-independent rendering may exhibit worse performance on semi-transparent objects (e.g., glass, smoke), as the compensation provided by the MLP is limited.
- The training time (1.5 hours) is longer than that of 3DGS (0.5 hours) due to the additional distillation and quantization steps.
- The overhead of MLP inference on mobile environments is not analyzed in depth, which may present a new bottleneck on low-end devices.
- The method is only validated on Snapdragon 8 Gen 3; its actual performance on low-end chips (e.g., Snapdragon 6 series) remains unknown.
- The network architecture choices for the view-dependent enhancement MLP (e.g., number of layers, hidden dimensions) lack a systematic search, suggesting that more optimal configurations may exist.
- The impact of codebook size ($K=256$) and sub-vector dimensions on the trade-off between compression ratio and quality during quantization has not been fully explored.
- The coefficients combining the three terms in the depth-aware weight function $w_i$ are empirically designed and lack rigorous theoretical analysis.

## Related Work & Insights
- **vs SortFreeGS**: Both utilize order-independent rendering, but SortFreeGS still suffers from large storage requirements and lacks compression. Mobile-GS introduces a complete compression pipeline.
- **vs LocoGS**: LocoGS achieves extreme compression (8.5 MB) but offers limited speed (292 FPS). Mobile-GS is both smaller (4.6 MB) and faster (1125 FPS).
- **vs LightGaussian**: LightGaussian only performs SH distillation (from 3rd to 2nd order), whereas this work is more aggressive (from 3rd to 1st order) and integrates quantization.
- **vs Compact3D/C3DGS**: These methods apply codebook compression but retain sorting, resulting in 30.6 MB of storage but no speedup. Mobile-GS addresses both speed and storage simultaneously.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic innovation combining order elimination and comprehensive mobile optimizations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validation on 3 datasets, real-world mobile testing, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation with comprehensive experimental results.
- Value: ⭐⭐⭐⭐⭐ The first truly real-time mobile 3DGS, demonstrating high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Seele: A Unified Acceleration Framework for Real-Time Gaussian Splatting on Mobile Devices](../../CVPR2026/3d_vision/seele_a_unified_acceleration_framework_for_real-time_gaussian_splatting_on_mobil.md)
- [\[CVPR 2025\] SOGS: Second-Order Anchor for Advanced 3D Gaussian Splatting](sogs_second-order_anchor_for_advanced_3d_gaussian_splatting.md)
- [\[CVPR 2025\] FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](fruitninja_3d_object_interior_texture_generation_with_gaussian_splatting.md)
- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)
- [\[CVPR 2025\] ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)

</div>

<!-- RELATED:END -->
