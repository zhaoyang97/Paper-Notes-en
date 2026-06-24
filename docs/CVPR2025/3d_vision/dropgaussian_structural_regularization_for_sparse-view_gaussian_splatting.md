---
title: >-
  [Paper Note] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][Sparse-view Novel View Synthesis] DropGaussian proposes a simple regularization method without additional priors. By randomly dropping Gaussians during 3DGS training and introducing an opacity compensation factor, it ensures that occluded, distant Gaussians receive larger gradients and visibility. Coupled with a progressive dropping rate strategy, it effectively mitigates overfitting under sparse-view conditions and achieves performance comparable to pr…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sparse-view Novel View Synthesis"
  - "3D Gaussian Splatting"
  - "Overfitting Regularization"
  - "Dropout"
  - "Prior-Free Method"
date: 2026-05-08
content_hash: 529aaf1342160208
---

# DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2504.00773](https://arxiv.org/abs/2504.00773)  
**Code**: [https://github.com/DCVL-3D/DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release)  
**Area**: 3D Vision  
**Keywords**: Sparse-view Novel View Synthesis, 3D Gaussian Splatting, Overfitting Regularization, Dropout, Prior-Free Method

## TL;DR
DropGaussian proposes a simple regularization method without additional priors. By randomly dropping Gaussians during 3DGS training and introducing an opacity compensation factor, it ensures that occluded, distant Gaussians receive larger gradients and visibility. Coupled with a progressive dropping rate strategy, it effectively mitigates overfitting under sparse-view conditions and achieves performance comparable to prior-based methods without increasing computational complexity.

## Background & Motivation

1. **Background**: 3DGS has demonstrated advantages in real-time rendering and high-quality image reconstruction for novel view synthesis. However, its performance severely degrades under sparse views (e.g., only 3 input views) due to overfitting to the training views.
2. **Limitations of Prior Work**: Existing methods mainly mitigate overfitting by introducing prior information: (a) utilizing monocular depth estimation as external supervision (DNGaussian), though depth scales across different views are inconsistent and errors propagate; (b) using 2D generative model priors (FSGS), which incurs high computational costs and unstable optimization; (c) employing optical flow regularization (CoR-GS), which requires an additional pre-trained model.
3. **Key Challenge**: Under sparse views, Gaussians that are far from the camera and occluded by front Gaussians receive very little gradient feedback due to their extremely limited visibility. Consequently, their attributes (scale, color, opacity) cannot be adequately updated, ultimately leading to overfitting onto a small number of training views.
4. **Goal**: To mitigate sparse-view overfitting through simple modifications to 3DGS itself, without relying on any external priors.
5. **Key Insight**: The authors observe that overfitting primarily occurs during the later stages of training rather than the early stages, and distant Gaussians lack gradient updates due to low visibility (low transmittance $T_i$). Reminiscent of Dropout in neural networks, randomly dropping Gaussians can "expose" the occluded Gaussians, allowing them to receive more gradients.
6. **Core Idea**: To increase the visibility and gradient feedback of the remaining Gaussians by randomly dropping Gaussians, and to progressively increase the dropping rate to match the characteristic that overfitting only becomes significant in the later training stages.

## Method

### Overall Architecture
DropGaussian introduces only a single modification to the standard 3DGS framework: randomly dropping a portion of Gaussians during training and using all Gaussians for rendering during testing. The entire method does not introduce any additional modules or external priors, making it a plug-and-play regularization technique. The framework is divided into two parts: (1) random Gaussian dropping with a compensation factor; (2) progressive dropping rate scheduling.

### Key Designs

1. **DropGaussian with Compensation**:

    - **Function**: Randomly removes a portion of Gaussians, allowing the remaining Gaussians (especially the occluded, distant ones) to receive larger gradients and higher visibility.
    - **Mechanism**: Define a dropping rate $r$ (e.g., $r=0.1$ indicates dropping 10%). A random mask $M(i)$ is generated for each Gaussian. The opacity of the retained Gaussians is multiplied by a compensation factor to get $\tilde{o}_i = M(i) \cdot o_i$, where $M(i) = \frac{1}{1-r}$ for the retained Gaussians and $M(i) = 0$ for the dropped Gaussians. This ensures the expectation of total color contribution per pixel remains unchanged.
    - **Design Motivation**: Under sparse views, the transmittance $T_i$ of distant Gaussians is very low (as they are occluded by front Gaussians), and their limited visibility leads to insufficient gradients. After randomly dropping front Gaussians, the $T_i$ of back Gaussians increases, enhancing their visibility and enabling more adequate gradient updates. The compensation factor ensures that the expected value of the overall color contribution during training remains unchanged.

2. **Progressive Dropping Schedule**:

    - **Function**: Dynamically adjusts the dropping rate based on training progress, strengthening regularization during the later stages of training.
    - **Mechanism**: The dropping rate increases linearly with iterations: $r_t = \gamma \cdot \frac{t}{t_{total}}$, where $\gamma$ is a scale factor (ranging from 0 to 1), $t$ is the current iteration, and $t_{total}$ is the total number of iterations.
    - **Design Motivation**: The authors experimentally find that overfitting primarily occurs during the later stages of training—namely, the PSNR continues to rise in the early stages, but the novel-view PSNR begins to drop in the later stages. Over-dropping Gaussians at the beginning of training could disrupt normal scene structure learning, while intensifying dropping in the later stage suppresses overfitting more purposefully.

3. **Prior-Free Design Philosophy**:

    - **Function**: Achieves regularization solely through simple modifications to 3DGS itself, without requiring any external models or extra computation.
    - **Mechanism**: The method does not introduce any external priors such as depth estimation, optical flow, or diffusion models, nor does it add any auxiliary losses or new modules. Regularization is achieved solely by modifying the Gaussian selection logic during rendering. Gaussians are dropped during training and all Gaussians are utilized during inference.
    - **Design Motivation**: Prior-based methods suffer from issues like error propagation (inaccurate depths), heavy computational overhead (generative models), and unstable training. DropGaussian demonstrates a key insight: performance degradation in sparse-view scenarios is inherently an overfitting issue that can be solved with classic regularization concepts, and does not necessarily require external priors.

### Loss & Training
- Uses the standard 3DGS color reconstruction loss: $\mathcal{L}_{color} = \mathcal{L}_1(\hat{I}, I) + \lambda \mathcal{L}_{D-SSIM}(\hat{I}, I)$, with $\lambda = 0.2$
- Trained for 10,000 iterations, performing densification once every 100 iterations
- The densification gradient threshold is set to $5 \times 10^{-4}$
- Run on an NVIDIA RTX 3090Ti

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | DropGaussian | 3DGS | FSGS | CoR-GS |
|--------|------|------|-------------|------|------|--------|
| LLFF | 3-view | PSNR↑ | **20.76** | 19.22 | 20.43 | 20.45 |
| LLFF | 3-view | SSIM↑ | **0.713** | 0.649 | 0.682 | 0.712 |
| LLFF | 6-view | PSNR↑ | **24.74** | 23.80 | 24.09 | 24.49 |
| Mip-NeRF360 | 12-view | PSNR↑ | **19.74** | 18.52 | 18.80 | 19.52 |
| Mip-NeRF360 | 24-view | PSNR↑ | **24.13** | 22.80 | 23.70 | 23.39 |
| Blender | 8-view | PSNR↑ | **25.42** | 21.56 | 24.64 | 24.43 |

### Ablation Study

| Configuration | PSNR (LLFF-3view) | Explanation |
|------|-------------------|------|
| Full Model (progressive) | **20.76** | Progressive dropping, optimal |
| Fixed drop rate | ~20.4 | Fixed dropping rate, slightly lower performance |
| w/o compensation factor | ~20.2 | Without compensation factor, color bias |
| Vanilla 3DGS | 19.22 | Without DropGaussian |

### Key Findings
- **Simple method rivals complex prior-based methods**: DropGaussian achieves a PSNR of 20.76 on LLFF 3-view, surpassing FSGS (20.43) and CoR-GS (20.45) which utilize depth/optical flow priors. This proves that overfitting is the main cause of degradation under sparse views.
- **Progressive dropping outperforms fixed dropping**: This validates the observation that "overfitting primarily occurs in the later stages"—progressively increasing the dropping rate better matches this characteristic.
- **Distant Gaussians receive larger gradients**: Analysis indicates that DropGaussian significantly improves the gradient distribution of distant Gaussians, with a notable increase in the number of distant Gaussians possessing large gradients.
- **Still effective under more views**: Even in 6-view, 9-view, and 24-view settings, DropGaussian remains competitive or even optimal, suggesting that the regularization effect is not solely limited to extremely sparse scenes.

## Highlights & Insights
- **Extremely Simple Design Philosophy**: The entire method only requires modifying a single line of rendering code (random dropping + compensation), without introducing any new parameters, modules, loss functions, or external models. This Occam's Razor style solution is elegant—proving that understanding the essence of the problem is more crucial than stacking complex modules.
- **Analogy from Dropout to DropGaussian**: Migrating the classic Dropout regularization in neural networks to 3D Gaussian representations. The core insight is: while Dropout prevents co-adaptation of neurons, DropGaussian prevents gradient imbalance caused by the excessive occlusion between Gaussians.
- **High Composability**: As a plug-and-play regularization, it can be combined with other 3DGS methods (such as FSGS and CoR-GS), yielding consistent improvements in all cases.

## Limitations & Future Work
- **Hyperparameter Sensitivity**: The optimal value of the scale factor $\gamma$ is dataset-dependent, requiring manual tuning. Future work can explore adaptive mechanisms to dynamically adjust $\gamma$.
- **Lack of Theoretical Guarantee**: Although experiments validate the effectiveness of the method, there is a lack of theoretical analysis regarding the relationship between the dropping rate and the degree of overfitting.
- **No Significant Gain in Dense-View Scenes**: When the training views are sufficient, overfitting is no longer the primary issue, and the improvement brought by DropGaussian is limited.
- **Perceptual Metrics such as CLIP-I/DINO Unreported**: Evaluated only using PSNR/SSIM/LPIPS, without involving more perceptual quality metrics.

## Related Work & Insights
- **vs DNGaussian**: DNGaussian utilizes a depth prior for regularization, whereas DropGaussian is completely prior-free. On LLFF 3-view, DropGaussian (20.76) surpasses DNGaussian (19.12), indicating that prior-free methods can be more effective than prior-based ones.
- **vs FSGS**: FSGS uses generative model priors to enhance under-covered regions, incurring high computational costs. DropGaussian achieves better performance (LLFF: 20.76 vs 20.43) with zero additional computational cost.
- **vs CoR-GS**: CoR-GS uses optical flow to regularize pixel correspondences, requiring a pre-trained optical flow model. DropGaussian achieves comparable performance with a much simpler approach.
- **Relationship with DropoutGS**: Both independently proposed similar dropout concepts for sparse-view 3DGS, but the technical details differ—DropGaussian utilizes a compensation factor to maintain color consistency, while DropoutGS employs an RDR loss in conjunction with an edge-guided splitting strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ Migrating the Dropout concept to 3DGS is a simple yet powerful insight, with the progressive dropping rate strategy backed by experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three standard datasets + multiple view settings + comparisons with feed-forward methods + gradient distribution analysis.
- Writing Quality: ⭐⭐⭐⭐ The method is intuitive and easy to understand, the motivation analysis is well-conceived, and the illustrations are clear.
- Value: ⭐⭐⭐⭐⭐ A minimalist method achieving SOTA performance, carrying extremely high practical value as a plug-and-play component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2025\] SPARS3R: Semantic Prior Alignment and Regularization for Sparse 3D Reconstruction](spars3r_semantic_prior_alignment_and_regularization_for_sparse_3d_reconstruction.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](../../ECCV2024/3d_vision/cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[CVPR 2025\] Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives](speedy-splat_fast_3d_gaussian_splatting_with_sparse_pixels_and_sparse_primitives.md)

</div>

<!-- RELATED:END -->
