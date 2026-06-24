---
title: >-
  [Paper Note] MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Reframes Gaussian pruning in 3DGS from deterministic removal to probabilistic existence modeling. Implements a masked-rasterization technique that allows unsampled Gaussians to still receive gradients for dynamic contribution assessment, achieving 62-75% Gaussian pruning rates on Mip-NeRF360/T&T/DeepBlending with only a 0.02 PSNR loss.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Gaussian pruning"
  - "probabilistic mask"
  - "rendering efficiency"
  - "masked rasterization"
date: 2026-05-08
content_hash: c1b0b9b60af69679
---

# MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks

**Conference**: CVPR 2025  
**arXiv**: [2412.20522](https://arxiv.org/abs/2412.20522)  
**Code**: [https://github.com/kaikai23/MaskGaussian](https://github.com/kaikai23/MaskGaussian)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Gaussian pruning, probabilistic mask, rendering efficiency, masked rasterization

## TL;DR

Reframes Gaussian pruning in 3DGS from deterministic removal to probabilistic existence modeling. Implements a masked-rasterization technique that allows unsampled Gaussians to still receive gradients for dynamic contribution assessment, achieving 62-75% Gaussian pruning rates on Mip-NeRF360/T&T/DeepBlending with only a 0.02 PSNR loss.

## Background & Motivation

3D Gaussian Splatting performs exceptionally well in novel view synthesis and real-time rendering, but suffers from Gaussian redundancy—a single indoor scene can require millions of Gaussians, leading to massive memory consumption and restricted rendering speeds. Existing pruning methods fall into two categories:

- **Importance score-based methods** (LightGaussian, RadSplat, Mini-Splatting): Design hand-crafted scoring functions that require scanning all training images to compute. Pruning can only be executed once or twice and fails to account for the dynamic evolution of the scene post-pruning.
- **Learnable mask-based methods** (Compact3DGS): Multiply a mask by Gaussian attributes (opacity/scale). However, once the mask becomes 0, opacity/scale is zeroed out $\rightarrow \alpha = 0$, which gets filtered out by the $\alpha$-filter, preventing it from receiving gradients and rendering it permanently unrecoverable.
- **Core Problem**: Deterministic pruning only reflects the importance of the current snapshot, ignoring the scene evolution after pruning. Gaussians that currently seem unimportant might become critical in later training stages (e.g., fine details or transparent objects), but once removed, they are gone irreversibly.

Core motivation: Model Gaussians as probabilistically existing entities, determine which Gaussians participate in rendering during each iteration through sampling, and allow unsampled Gaussians to receive gradients to update their existence probabilities using a specially designed masked-rasterization.

## Method

### Overall Architecture

MaskGaussian maintains an existence probability distribution for each Gaussian (2 learnable mask scores + Gumbel-Softmax sampling). In each iteration: (1) a binary mask is sampled to determine the presence/absence of the Gaussian; (2) all Gaussians (including those masked) undergo normal splatting to compute $\alpha$ and pass through the $\alpha$-filter; (3) the mask is applied to transmittance decay and color accumulation within the masked-rasterization; (4) masked Gaussians, though not affecting the rendering outputs, still participate in the forward pass and receive backward gradients.

### Key Designs

1. **Masked Rasterization (Forward)**:
    - **Function**: Applies the mask during the rasterization process rather than to the Gaussian attributes, ensuring that masked Gaussians do not affect the rendering but still participate in the computation.
    - **Mechanism**: Modifies the $\alpha$-blending equations of 3DGS. Color accumulation becomes $c(\mathbf{x}) = \sum_{i=1}^N \mathcal{M}_i \cdot c_i \cdot \alpha_i \cdot T_i$, and transmittance decay becomes $T_{i+1} = \mathcal{M}_i \cdot (1-\alpha_i) \cdot T_i + (1-\mathcal{M}_i) \cdot T_i$. When $\mathcal{M}_i = 0$, the color contribution is masked and the transmittance is not consumed—the Gaussian acts as "non-existent". Crucially, the mask is not multiplied with opacity/scale, meaning $\alpha_i$ is not zero, and the Gaussian still passes through the $\alpha$-filter to participate in rasterization.
    - **Design Motivation**: Compact3DGS multiplies the mask with opacity, leading to $\alpha=0$ and eliminating any gradient feedback once filtered. Masked-rasterization decouples the mask from Gaussian attributes, keeping the gradient pathway intact.

2. **Masked Rasterization (Backward Gradient Analysis)**:
    - **Function**: Derives the gradient formula for the mask, proving that unsampled Gaussians can receive meaningful gradients to update their existence probabilities.
    - **Mechanism**: The gradient of the mask is $\frac{\partial L}{\partial \mathcal{M}_i} = \alpha_i \cdot T_i \cdot \frac{\partial L}{\partial c(\mathbf{x})} \cdot (c_i - b_{i+1})$, where $b_{i+1}$ is the accumulated color behind the $i$-th Gaussian. Interpretation: (1) $\alpha_i \cdot T_i$ is the weight of the Gaussian's influence on color; (2) $\frac{\partial L}{\partial c(\mathbf{x})} \cdot (c_i - b_{i+1})$ measures the gain of using the Gaussian's color $c_i$ compared to not using it (relying on the background color $b_{i+1}$). If the dot product is positive, it indicates the Gaussian contributes positively, and its existence probability should increase—even if it is currently masked.
    - **Design Motivation**: This gradient formula naturally incorporates $\alpha_i \cdot T_i$ (the importance criterion in score-based methods) and additionally captures the contrast between "current color vs. background color," which score-based methods fail to measure.

3. **Probabilistic Sampling & Sparsified Training**:
    - **Function**: Automatically learns the existence probability of each Gaussian and performs dynamic pruning.
    - **Mechanism**: Each Gaussian has 2 mask scores, obtaining a differentiable binary mask $\mathcal{M}_i \in \{0,1\}$ via Gumbel-Softmax sampling. A squared mask loss $L_m = (\frac{1}{N}\sum_i \mathcal{M}_i)^2$ is used to regularize the average number of sampled Gaussians (performing better than L1). Pruning strategy: During each pruning step, sampling is performed 10 times; Gaussians that are never sampled are considered low-probability and removed. This is executed at each phase of the densification period and every 1000 iterations thereafter.
    - **Design Motivation**: Deterministic masks are irreversible once removed. Probabilistic sampling allows Gaussians to dynamically appear/disappear across different iterations, adapting to the evolution of the scene.

### Loss & Training

- Total Loss: $L = L_{render} + \lambda_m \cdot L_m$
- $L_{render}$: Standard 3DGS rendering loss (L1 + SSIM)
- $L_m = (\frac{1}{N}\sum_i \mathcal{M}_i)^2$: Squared mask regularization (constraining the number of used Gaussians)
- $\lambda_m$: Trade-off coefficient; Ours-$\alpha$ uses 0.1 during 19K-20K iterations, Ours-$\beta$ uses 0.0005 throughout, and Ours-$\gamma$ uses 0.001 throughout.

## Key Experimental Results

### Main Results

| Method | Mip-NeRF360 PSNR↑ | #GS↓(M) | FPS↑ | T&T PSNR↑ | #GS↓(M) | DeepBlend PSNR↑ | #GS↓(M) |
|------|-------------------|---------|------|-----------|---------|----------------|---------|
| 3DGS | 27.45 | 3.204 | 187.8 | 23.74 | 1.825 | 29.53 | 2.815 |
| Compact3DGS | 27.32 | 1.533 | 281.1 | 23.61 | 0.960 | 29.58 | 1.310 |
| RadSplat | 27.45 | 2.184 | 247.8 | 23.61 | 1.053 | 29.55 | 1.515 |
| **MaskGaussian** | **27.43** | **1.205** | **384.7** | **23.72** | **0.590** | **29.69** | **0.694** |

### Ablation Study

| Method | Mip PSNR | #GS(M) | T&T PSNR | #GS(M) | DB PSNR | #GS(M) |
|------|---------|--------|---------|--------|---------|--------|
| Compact3DGS | 27.32 | 1.533 | 23.61 | 0.960 | 29.58 | 1.310 |
| Ours-$\beta$ (Same Config) | **27.44** | 1.520 | **23.66** | **0.740** | **29.76** | **0.913** |
| Ours-$\gamma$ (Strong Comp.) | 27.42 | **1.171** | 23.59 | **0.549** | **29.74** | **0.570** |

### Key Findings

- Achieves pruning rates of 62.4%/67.7%/75.3% (Mip/T&T/DB), delivering rendering speedups of 2.05×/2.19×/3.16× with a PSNR drop of only 0.02.
- **Improves rendering quality on Deep Blending** (29.69 vs. 29.53 of 3DGS), which is attributed to the regularization effect of pruning.
- Under identical training configurations (Ours-$\beta$), MaskGaussian concurrently achieves higher PSNR and employs fewer Gaussians across all datasets.
- Successfully retains fine, transparent structures such as the tire valve and penetrating spokes in the "bicycle" scene, which are lost in Compact3DGS.
- Can be seamlessly integrated with other optimization frameworks like Taming-3DGS to further boost efficiency.
- Multiplying the mask by opacity/scale (Compact3DGS approach) vs. masked-rasterization: the latter yields around 0.3-0.5 higher PSNR.

## Highlights & Insights

- **Intrinsic Difference Between Probabilistic and Deterministic Pruning**: Deterministic pruning relies on "snapshot" evaluations where currently unimportant Gaussians are permanently removed, whereas probabilistic existence allows Gaussians to "revive" during training, adapting to the dynamic evolution of the scene.
- **Elegance and Naturalness of the Gradient Formula**: The mask gradient naturally blends $\alpha_i \cdot T_i$ (the conventional importance criterion) with the "color contribution gain," offering a concise mathematical form with intuitive physical meaning.
- **Core Insight on Mask Application Position**: The mask should not be multiplied with Gaussian attributes (which breaks gradients) but applied during the blending stage of rasterization—this seemingly minor design difference leads to a qualitative leap.
- **Analogy to Vision Transformer Dynamic Pruning**: Drawing inspiration from token pruning in ViTs where active and inactive tokens simultaneously receive gradients, presenting a solid example of cross-domain knowledge transfer.

## Limitations & Future Work

- Training time is slightly longer than 3DGS due to the overhead of mask sampling and masked-rasterization.
- The setting of $\lambda_m$ heavily impacts the final pruning rate and visual quality, requiring hyperparameter tuning for different scenes or requirements.
- Only validated on vanilla 3DGS; not yet extended to variants like 2DGS or Scaffold-GS.
- Probabilistic sampling introduces randomness, which could compromise training stability, especially in the early stages.
- Modifications to CUDA kernels increase engineering complexity, making it less easily deployable compared to simple post-processing pruning.

## Related Work & Insights

- **Core Distinction from Compact3DGS**: Compact3DGS multiplies the mask directly onto the opacity/scale, which is deterministic—once mask=0, the Gaussian permanently loses gradients. MaskGaussian applies the mask inside rasterization, ensuring dynamic probabilistic evaluation.
- **Distinction from LightGaussian/RadSplat**: Score-based methods evaluate importance solely on static snapshots at specific moments. MaskGaussian adaptively tracks the dynamic evolution of scenes through iterative probabilistic sampling.
- **Insight**: The key to 3DGS pruning is not just "evaluating which Gaussians are important," but rather "continuously re-evaluating importance as the scene evolves." Probabilistic sampling combined with gradient backpropagation serves as an effective framework for achieving this.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ A paradigm shift from deterministic to probabilistic pruning. Both the mathematical derivation of masked-rasterization and its CUDA implementation display significant depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison across three datasets with detailed ablation studies (mask application position, loss formulations, hyperparameters), though lacks evaluations on large-scale outdoor scenes.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem statement, rigorous gradient derivation, and highly convincing qualitative comparisons on fine details (e.g., tire valves, withered vines).
- **Value**: ⭐⭐⭐⭐ Opens up new avenues for 3DGS efficiency optimization; masked-rasterization holds broader application prospects as a general masking interface.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting](cob-gs_clear_object_boundaries_in_3dgs_segmentation_based_on_boundary-adaptive_g.md)
- [\[CVPR 2025\] EigenGS: Representation from Eigenspace to Gaussian Image Space](eigengs_representation_from_eigenspace_to_gaussian_image_space.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[CVPR 2025\] Evolving High-Quality Rendering and Reconstruction in a Unified Framework with Contribution-Adaptive Regularization](evolving_high-quality_rendering_and_reconstruction_in_a_unified_framework_with_c.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](../../ICCV2025/3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)

</div>

<!-- RELATED:END -->
