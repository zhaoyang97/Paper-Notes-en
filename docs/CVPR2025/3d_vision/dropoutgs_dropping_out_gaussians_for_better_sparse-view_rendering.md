---
title: >-
  [Paper Note] DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering
description: >-
  [CVPR 2025][3D Vision][Sparse-view rendering] DropoutGS mitigates overfitting in sparse-view 3DGS through Random Dropout Regularization (RDR), and compensates for high-frequency details lost in low-complexity models using an Edge-guided Splitting Strategy (ESS). Serving as a plug-and-play module, it can be integrated with various 3DGS methods, achieving SOTA performance on LLFF, DTU, and Blender.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sparse-view rendering"
  - "3D Gaussian Splatting"
  - "Dropout regularization"
  - "Edge-guided splitting"
  - "Overfitting mitigation"
date: 2026-05-08
content_hash: 22c260bbef31dd43
---

# DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering

**Conference**: CVPR 2025  
**arXiv**: [2504.09491](https://arxiv.org/abs/2504.09491)  
**Code**: [https://xuyx55.github.io/DropoutGS/](https://xuyx55.github.io/DropoutGS/)  
**Area**: 3D Vision  
**Keywords**: Sparse-view rendering, 3D Gaussian Splatting, Dropout regularization, Edge-guided splitting, Overfitting mitigation

## TL;DR
DropoutGS mitigates overfitting in sparse-view 3DGS through Random Dropout Regularization (RDR), and compensates for high-frequency details lost in low-complexity models using an Edge-guided Splitting Strategy (ESS). Serving as a plug-and-play module, it can be integrated with various 3DGS methods, achieving SOTA performance on LLFF, DTU, and Blender.

## Background & Motivation

1. **Background**: 3DGS achieves real-time rendering via differentiable rasterization, performing exceptionally well under dense views. However, acquiring a large number of training views is costly in real-world scenarios, making sparse-view 3DGS a challenging problem.
2. **Limitations of Prior Work**: (a) Methods based on depth priors (e.g., DRGS, DNGaussian) are sensitive to depth accuracy, where errors can propagate and amplify, causing artifacts; (b) depth estimation requires additional computational modules; (c) 3DGS easily over-parameterizes under sparse inputs, leading to a continuous decrease in training loss but an increase in test loss (classic overfitting).
3. **Key Challenge**: The imbalance between model complexity (number of Gaussians) and the amount of training data (number of views) is the fundamental cause of overfitting. Using 10k Gaussians under 3 views leads to severe overfitting, whereas using 1k Gaussians reduces overfitting but sacrifices high-frequency details.
4. **Goal**: To mitigate overfitting while preserving the model's ability to represent high-frequency details—a typical trade-off.
5. **Key Insight**: The authors observe two key phenomena through pilot experiments: (a) fewer Gaussians lead to less overfitting but worse high-frequency details; (b) models with fewer Gaussians exhibit larger Gaussian scales, tending to cover larger regions to understand the 3D structure. This inspires the strategy of "first utilizing dropout to gain the generalization benefits of a low-complexity model, then employing edge-guided splitting to restore high-frequency details."
6. **Core Idea**: Dropping out Gaussians randomly during training to achieve an ensemble effect of multiple low-complexity subnetworks, thereby mitigating overfitting, and then applying an edge-guided splitting strategy to refine large-scale Gaussians to recover high-frequency details.

## Method

### Overall Architecture
DropoutGS consists of two complementary modules: (1) Random Dropout Regularization (RDR) randomly deactivates Gaussians during training, using the full-model rendering result as the target to supervise subnetworks, achieving an effect similar to ensemble learning; (2) Edge-guided Splitting Strategy (ESS) identifies and splits large-scale Gaussians with high edge scores in the late stage of training to recover high-frequency details near boundaries. The overall method can be seamlessly integrated into various 3DGS frameworks.

### Key Designs

1. **Random Dropout Regularization (RDR)**:

    - **Function**: Reduces the effective model complexity through random Gaussian deactivation, mitigating overfitting under sparse views.
    - **Mechanism**: Deactivates Gaussians randomly with probability $p$. The rendering result is formulated as $\hat{C} = \sum_{i \in \mathcal{N}} r_i \cdot \alpha_i \prod_{j=1}^{i-1}(1 - r_j \cdot \alpha_j) c_i$, where $r_i \sim \text{Bernoulli}(1-p)$. The key design is that the supervision signal for RDR is the **full-model rendering result** $C$ rather than the ground-truth (GT) image: $\mathcal{L}_{RDR} = \| C - \hat{C} \|_1 + \text{SSIM}(C, \hat{C})$. Consequently, gradients only backpropagate through local regions affected by dropout, forcing neighboring Gaussians to compensate for the deactivated ones.
    - **Design Motivation**: Supervising the post-dropout submodels directly with GT images would cause gradients of all Gaussians to cancel each other out (global optimization), leading to suboptimal results. In contrast, supervising with the full-model rendering limits gradients to local regions affected by dropout, which is more effective. From the perspective of ensemble learning, training multiple low-complexity subnetworks and ensembling them at inference time (by using all Gaussians) yields consistent performance gains.

2. **Edge-guided Splitting Strategy (ESS)**:

    - **Function**: Identifies large-scale Gaussians in boundary regions and splits them into smaller ones, recovering the loss of high-frequency details caused by RDR.
    - **Mechanism**: (a) Utilizes an edge detector to obtain pixel-level edge probabilities $E(I)$ for each input view; (b) projects Gaussians onto the edge map and calculates the single-view edge score $\mathcal{E}'_i = \alpha_i \prod_j^{i-1}(1-\alpha_j) \cdot \sum_p E(I) \mathcal{M}^i(p)$; (c) accumulates and normalizes across views to obtain the final edge score $\mathcal{E}_i$; (d) splits Gaussians that satisfy both a large scale $S_i \geq \mathcal{S}_{thr}$ and a high edge score $\mathcal{E}_i \geq \mathcal{E}_{thr}$.
    - **Design Motivation**: RDR encourages Gaussians to grow larger to cover wider areas and learn 3D structures, but large Gaussians cannot capture high-frequency details such as edges. ESS selectively splits large Gaussians only in boundary areas, restoring details without increasing the risk of overfitting.

3. **Theoretical Model Interpretation from the Ensemble Learning Perspective**:

    - **Function**: Explains theoretically why dropout-style regularization is effective for 3DGS.
    - **Mechanism**: Each rendering after dropout is equivalent to the output of a low-complexity submodel. Optimizing multiple submodels during training is equivalent to implicitly training an ensemble of subnetworks. At inference time, utilizing all Gaussians corresponds to ensembling all submodels via their geometric mean. As shown in the visualization in Fig 6, rendering from a single submodel contains artifacts, but the results improve significantly as the number of ensembled submodels increases.
    - **Design Motivation**: Dropout has been mathematically proven in deep learning to mitigate overfitting by approximating the geometric mean of an exponential number of subnetworks. Generalizing this theory to 3DGS provides solid theoretical support for the method's effectiveness.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{gs} + \lambda_{depth} \mathcal{L}_{depth} + \lambda_{RDR} \mathcal{L}_{RDR}$
- $\mathcal{L}_{gs}$ is the standard 3DGS loss (L1 + D-SSIM).
- $\mathcal{L}_{depth}$ is the optional depth regularization (from DNGaussian).
- No compensation strategy is used for dropout (experiments show that compensation does not significantly boost performance and may introduce negative effects).
- Trained for 6k iterations, using randomly initialized point clouds.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | DropoutGS | DNGaussian | FreeNeRF | 3DGS |
|--------|------|------|-----------|------------|----------|------|
| LLFF | 3-view | PSNR↑ | **19.35** | 19.12 | 19.63 | 16.46 |
| LLFF | 3-view | LPIPS↓ | **0.282** | 0.294 | 0.308 | 0.401 |
| DTU | 3-view | PSNR↑ | **20.22** | 18.91 | 19.92 | 14.74 |
| DTU | 3-view | SSIM↑ | **0.830** | 0.790 | 0.787 | 0.672 |
| Blender | 8-view | PSNR↑ | **24.476** | 24.305 | 24.259 | 22.226 |
| Blender | 8-view | LPIPS↓ | **0.085** | 0.088 | 0.098 | 0.114 |

### Ablation Study

| Method | w/ DropoutGS | PSNR | LPIPS | SSIM | Initialization |
|------|-------------|------|-------|------|--------|
| 3DGS† | ✗ | 16.46 | 0.401 | 0.440 | Random |
| 3DGS† | ✓ | **18.05** | 0.326 | 0.545 | Random |
| FSGS | ✗ | 19.86 | 0.222 | 0.670 | MVS |
| FSGS | ✓ | **20.53** | - | - | MVS |
| CoR-GS | ✗ | 20.45 | 0.196 | 0.712 | MVS |
| CoR-GS | ✓ | - | - | - | MVS |

### Key Findings
- **Most significant improvement on DTU**: DropoutGS achieves a PSNR of 20.22 under DTU 3-view, which is 1.31 dB higher than DNGaussian, demonstrating that the prevention of overfitting is more pronounced in object-centric scenes.
- **Choice of RDR supervision signal is critical**: Using the full-model rendering result rather than the GT image as the supervision target ensures that gradients only act on local regions (those affected by dropout), thereby guiding neighboring Gaussians to learn more effectively.
- **Compatible with various 3DGS methods**: Uniform improvements are obtained when combined with 3DGS, FSGS, and CoR-GS, proving its versatility as a plug-and-play module.
- **Depth map quality improvement**: The depth maps generated by DropoutGS are smoother and more accurate than those of DNGaussian, indicating that overfitting mitigation also improves geometric quality.
- **No compensation strategy used**: Unlike DropGaussian, experiments on DropoutGS demonstrate that the compensation strategy does not yield significant improvements and may negatively affect pixels untouched by dropout.

## Highlights & Insights
- **A "coarse-to-fine" problem-solving paradigm**: Leveraging RDR first to harvest the generalization advantages of a low-complexity model (smooth but lacking details), and then using ESS to selectively recover high-frequency details around boundary regions. This coarse-to-fine framework could be transferred to other overfitting scenarios.
- **Exquisite choice of supervision signal**: Supervising the dropout submodels with the full-model rendering (rather than GT) serves as a key innovation. This restricts gradients to local regions affected by dropout, preventing global gradients from canceling each other out. The visualization of gradient maps (Fig 5) intuitively demonstrates this difference.
- **Convincing pilot study on model complexity and data volume**: Fig 3 illustrates the optimal mapping between different Gaussian counts and different view counts, while Fig 4 displays that highly complex models tend to have smaller Gaussian scales (overfitting to details of training views). This provides a solid experimental basis for the proposed design.

## Limitations & Future Work
- **FreeNeRF still achieves higher PSNR on LLFF**: In the LLFF 3-view setting, FreeNeRF (19.63) outperforms DropoutGS (19.35) in terms of PSNR, though DropoutGS is superior in LPIPS. This indicates that frequency regularization still holds advantages in forward-facing scenes.
- **No in-depth discussion on edge detector choices**: ESS relies on an edge detector, but the impacts of different detectors are not thoroughly analyzed.
- **Tuning of the dropout probability $p$**: As a hyperparameter, $p$ may require dataset-specific tuning, lacking an adaptive mechanism.
- **Extreme sparse scenarios (2-view) not tested**: Experiments are only conducted on 3-view settings and above; performance under extreme sparsity remains unknown.

## Related Work & Insights
- **vs DropGaussian**: Two concurrent works independently proposed applying dropout to sparse-view 3DGS. The innovations of DropoutGS lie in (a) using the full-model rendering instead of the GT as supervision, and (b) incorporating ESS to compensate for high-frequency details. DropGaussian employs opacity compensation factors and progressive dropping rates. The two methods are complementary.
- **vs DNGaussian**: DNGaussian utilizes depth prior regularization, whereas DropoutGS tackles the problem from the perspective of overfitting. The two are compatible, with DropoutGS further boosting the performance of DNGaussian.
- **vs FreeNeRF**: FreeNeRF resolves the sparse-view issue of NeRF using frequency regularization. DropoutGS transfers similar regularization ideas to the discrete representation of 3DGS.
- **Ensemble learning inspiration**: Interpreting dropout as an implicit submodel ensemble provides a theoretical framework for regularizing 3DGS, inspiring further exploration of regularization techniques in the future.

## Rating
- Novelty: ⭐⭐⭐⭐ The supervision signal design of RDR and the compensation strategy of ESS are key innovations. The pilot study analysis is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets + compatibility validation + depth map visualization + gradient map analysis + point-cloud visualization.
- Writing Quality: ⭐⭐⭐⭐ In-depth motivation analysis, convincing pilot study, and clear method explanation.
- Value: ⭐⭐⭐⭐ A generic, plug-and-play regularization module showing significant improvements on DTU and demonstrating strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](../../CVPR2026/3d_vision/dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2025\] MAtCha Gaussians: Atlas of Charts for High-Quality Geometry and Photorealism From Sparse Views](matcha_gaussians_atlas_of_charts_for_high-quality_geometry_and_photorealism_from.md)
- [\[CVPR 2025\] Learnable Infinite Taylor Gaussian for Dynamic View Rendering](learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)
- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
