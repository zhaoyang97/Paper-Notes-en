---
title: >-
  [Paper Note] Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos
description: >-
  [CVPR 2025][3D Vision][Dynamic Scene Segmentation] This paper proposes Layered Motion Fusion (LMF), which integrates predictions from 2D motion segmentation models into the dynamic and semi-static layers of layered Neural Radiance Fields. Together with a test-time refinement strategy, this work demonstrates for the first time that 3D methods can outperform 2D baselines in dynamic object segmentation for egocentric videos, improving dynamic object segmentation mAP by 30.5%.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dynamic Scene Segmentation"
  - "NeRF"
  - "Egocentric Videos"
  - "Motion Segmentation Fusion"
  - "Test-time Refinement"
date: 2026-05-08
content_hash: 9a73956b7bf9b47f
---

# Layered Motion Fusion: Lifting Motion Segmentation to 3D in Egocentric Videos

**Conference**: CVPR 2025  
**arXiv**: [2506.05546](https://arxiv.org/abs/2506.05546)  
**Code**: None  
**Area**: 3D Vision / Video Understanding  
**Keywords**: Dynamic Scene Segmentation, NeRF, Egocentric Videos, Motion Segmentation Fusion, Test-time Refinement

## TL;DR
This paper proposes Layered Motion Fusion (LMF), which integrates predictions from 2D motion segmentation models into the dynamic and semi-static layers of layered Neural Radiance Fields. Together with a test-time refinement strategy, this work demonstrates for the first time that 3D methods can outperform 2D baselines in dynamic object segmentation for egocentric videos, improving dynamic object segmentation mAP by 30.5%.

## Background & Motivation

1. **Background**: 3D techniques (such as NeRF) have made significant progress in semantic fusion for static scenes—methods like Semantic NeRF, N3F, and LERF distill 2D semantics/features into 3D reconstructions to achieve multi-view consistent semantic labels. However, these methods assume the scene is static.

2. **Limitations of Prior Work**: Systematic studies on EPIC Fields show that in egocentric videos (characterized by severe camera motion + dynamic scenes), 3D methods actually perform *worse* than 2D baselines (such as Motion Grouping) in dynamic object segmentation. This means that the classic advantages of 3D fusion (filtering noise, multi-view consistency) fail in dynamic scenes.

3. **Key Challenge**: Long-duration, high-complexity egocentric videos make it difficult for 3D models to fully capture the scene geometry. Without accurate geometry as a carrier, motion cues cannot be effectively fused into 3D representations.

4. **Goal**: (1) How to effectively fuse 2D motion segmentation predictions into the layered representation of dynamic NeRFs? (2) How to overcome the geometric modeling limitations caused by highly complex scenes in long videos?

5. **Key Insight**: The authors observe that while 2D motion segmentation is incomplete (detecting only a portion of moving pixels), its precision is high (the labeled pixels are indeed moving), resembling "sparse but accurate" labels—which is precisely the type of input Semantic NeRF can handle effectively. Furthermore, simultaneously fusing motion information into both the dynamic layer (positive) and the semi-static layer (negative constraints) can yield synergistic effects.

6. **Core Idea**: Distill 2D motion segmentations into the dynamic and semi-static layers of a layered NeRF via Positive Motion Fusion (PMF) and Negative Motion Fusion (NMF), and utilize test-time refinement focusing on specific frames to alleviate the challenges of insufficient geometric modeling.

## Method

### Overall Architecture
The input consists of an egocentric video, corresponding 2D motion segmentation masks (from Motion Grouping), and a trained layered NeRF (including static, semi-static, and dynamic layers). The method is two-fold: (1) **Layered Motion Fusion (LMF)** fuses the motion masks into the semi-static and dynamic layers during training/refinement—PMF encourages the dynamic layer to learn the moving regions, while NMF penalizes the semi-static layer's predictions in these regions. (2) **Test-time Refinement (TR)** focuses the model on a selected subset of frames for fine-tuning during testing, which reduces data complexity and leads to more precise geometric modeling. The two components form a synergistic cycle: better geometry $\rightarrow$ more effective fusion $\rightarrow$ more accurate segmentation.

### Key Designs

1. **Positive Motion Fusion (PMF)**:

    - **Function**: Fuses 2D motion segmentation masks as pseudo-labels into the dynamic layer.
    - **Mechanism**: Defines a pseudo-color $\mathbf{p}_{\text{dy}} = (0,0,1)$ for each 3D point in the dynamic layer. The "mask map" $\hat{M}_{\text{dy}}(\mathbf{u},t)$ of the dynamic layer is rendered via volume rendering. This mask value essentially represents the opacity ratio of the dynamic layer at that pixel: $m_{\text{dy}}(\mathbf{x},t) = \sigma_{\text{dy}} / \sigma$. The PMF loss is formulated as $\mathcal{L}_{\text{PMF}} = \lambda_{\text{PMF}} \frac{1}{|\Omega|} \sum_{\mathbf{u}} \|\hat{M}_{\text{dy}}(\mathbf{u},t) - M(\mathbf{u},t)\|^2$, which directly supervises the dynamic layer using the 2D dynamic mask as the target.
    - **Design Motivation**: While Motion Grouping predictions are incomplete, they are highly precise. The dynamic layer can leverage 3D consistency to fill in missing parts (e.g., occluded moving regions), which aligns with the principles of Semantic NeRF fusing sparse labels.

2. **Negative Motion Fusion (NMF)**:

    - **Function**: Uses the motion mask to constrain the semi-static layer, preventing it from incorrectly "absorbing" dynamic content.
    - **Mechanism**: Similarly renders the mask map of the semi-static layer $\hat{M}_{\text{ss}}(\mathbf{u},t)$, where $m_{\text{ss}}(\mathbf{x},t) = \sigma_{\text{ss}} / \sigma$. Let $\bar{\Omega}$ be the set of dynamic pixels selected after binarizing the 2D motion mask. The NMF loss is $\mathcal{L}_{\text{NMF}} = \lambda_{\text{NMF}} \frac{1}{|\bar{\Omega}|} \sum_{\mathbf{u} \in \bar{\Omega}} \|\hat{M}_{\text{ss}}(\mathbf{u},t)\|^2$, pushing the response of the semi-static layer toward zero at dynamic pixel locations.
    - **Design Motivation**: A competition exists between the semi-static and dynamic layers in layered NeRFs. Without explicit constraints, the semi-static layer might "steal" dynamic content, leading to mis-segmentation. NMF uses 2D motion information as a negative constraint to eliminate this ambiguity. Experiments show that NMF also unexpectedly improves the quality of semi-static segmentation (+8.4%).

3. **Test-time Refinement (TR)**:

    - **Function**: Focuses on a user-selected subset of frames to fine-tune the model, improving local geometric quality.
    - **Mechanism**: Given a set of frames to analyze $\mathcal{T}$, the static layer parameters $W_{\text{st}}$ are frozen, and only the parameters of the semi-static and dynamic layers are optimized: $(W_{\text{ss}}^*, W_{\text{dy}}^*) = \arg\min \sum_{t \in \mathcal{T}} \mathcal{L}(W_{\text{st}}, W_{\text{ss}}, W_{\text{dy}}; I_t, M_t, t)$. Additional temporal context can be provided by sampling $N$ neighboring frames for each frame (forming $\mathcal{T}_N$). The static layer is frozen because it does not contain motion information.
    - **Design Motivation**: Scene variations across long videos are extreme, requiring the model to cover the geometry of all frames within a limited capacity, which leads to insufficient local geometric quality. TR narrows down the optimization target to allow the model to focus on local frames, significantly improving geometric accuracy. Refinement takes about 22 minutes per 100 frames (~13s/frame) and renders at ~5s/frame.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{\text{rgb}} + \mathcal{L}_{\text{PMF}} + \mathcal{L}_{\text{NMF}}$
- RGB loss employs a self-calibrating robust loss (with learned uncertainty $B$), in the form of weighted MSE + uncertainty regularization.
- $\lambda_{\text{PMF}} = 1.1$, $\lambda_{\text{NMF}} = 1.0$
- The base model adheres to NeuralDiff's pre-training settings (20 epochs, lr=5e-4, cosine annealing, RTX A4000).

## Key Experimental Results

### Main Results

| Method | 3D | 2D Fusion | Dyn mAP ↑ | SS mAP ↑ | Dyn+SS mAP ↑ |
|------|----|----|-----------|---------|-------------|
| Motion Grouping (MG) | ✗ | ✓ | 64.27 | 12.78 | 55.53 |
| NeRF-W | ✓ | ✗ | 28.52 | 20.97 | 45.62 |
| NeRF-T | ✓ | ✗ | 44.27 | 24.48 | 64.91 |
| NeuralDiff (ND) | ✓ | ✗ | 55.58 | 25.55 | 69.74 |
| **ND + TR + LMF (Ours)** | ✓ | ✓ | **72.51** | **27.70** | **74.21** |

Compared to NeuralDiff: dynamic segmentation $+30.5\%$, semi-static $+8.4\%$, joint $+6.4\%$. Key breakthrough: **For the first time, a 3D method outperforms the 2D baseline MG** (72.51 vs 64.27), resolving the open challenge posed by EPIC Fields.

### Generalization Across Architectures

| Base Method | Dyn mAP | With TR+PMF | Gain |
|---------|---------|------------|------|
| NeRF-W | 28.52 | 34.20 | +19.9% |
| NeRF-T | 44.27 | 51.11 | +15.4% |
| NeuralDiff | 55.58 | 67.23 | +20.9% |

The LMF method consistently improves the dynamic segmentation performance across three different NeRF architectures.

### Key Findings
- **PMF and NMF Synergistic Effects**: PMF improves dynamic segmentation, NMF improves semi-static segmentation, and their combination performs better than using either alone.
- TR improves dynamic segmentation by about 10.6% (evaluated on a 5-scene subset), which is further enhanced when combined with LMF.
- Increasing the number of neighboring frames $N$ yields diminishing returns for TR: performance is already solid at $N=0$, and marginal improvements are observed at $N=2$ or $N=5$.
- Freezing the static layer is a key decision in TR—the static layer contains no motion information, and fine-tuning it may degrade the geometric foundation.
- The method is generalizable across different 3D architectures (demonstrated on NeRF-W, NeRF-T, and NeuralDiff).

## Highlights & Insights
- **Layered Positive-Negative Fusion Design**: The complementary strategy of PMF raising the dynamic layer and NMF suppressing the semi-static layer is simple yet highly effective, particularly yielding an unexpected boost in semi-static segmentation quality.
- **Deep Insight on Test-time Refinement**: In long videos, the bottleneck of 3D models is not the method design but rather the geometric capacity constraint. The idea of "focusing" the model by narrowing down the optimization frame set is akin to intensive review before an exam.
- **First Demonstration of 3D Outperforming 2D on Dynamic Segmentation**: This directly answers the open question posed by EPIC Fields, boosting confidence in using 3D vision to process dynamic scenes.
- The "sparse but precise" paradigm of pseudo-label fusion can be transferred to other dynamic scene tasks that require 2D-to-3D distillation.

## Limitations & Future Work
- Dependency on the quality of Motion Grouping's 2D outputs—if the 2D model's predictions are completely incorrect, the fusion introduces noise.
- TR requires separate fine-tuning for each test frame set (~13s/frame), making it unsuitable for real-time applications.
- Evaluations are only conducted on the EPIC Fields benchmark, limiting scene types to kitchen environments.
- More efficient 3D representations (e.g., 3D Gaussian Splatting) have not been explored; the training and rendering speeds of current NeRFs remain bottlenecks for practical deployment.
- The binarization threshold and $\lambda$ settings for NMF may require tuning across different scenes.

## Related Work & Insights
- **vs NeuralDiff**: NeuralDiff serves as the base architecture for layered NeRF, but relies solely on RGB reconstruction losses to decompose scene layers, lacking explicit motion supervision. LMF significantly improves the accuracy of three-layer decomposition by introducing 2D motion signals.
- **vs EPIC Fields**: EPIC Fields found that 3D methods lag behind 2D counterparts in dynamic segmentation; this work directly addresses this issue. The key difference lies in the introduction of 2D-to-3D motion distillation.
- **vs Semantic NeRF**: Semantic NeRF fuses semantic labels into static NeRF. LMF extends a similar fusion strategy to layered representations in dynamic scenes, performing both positive and negative fusions.
- This method holds promise for combining with 3D Gaussian Splatting (e.g., SAGA, GARField) to achieve faster dynamic semantic fusion.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of layered positive/negative motion fusion is novel and intuitive, and the test-time refinement strategy is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on the EPIC Fields benchmark with detailed cross-architecture generalization and ablation studies, though scene variety is limited.
- Writing Quality: ⭐⭐⭐⭐ Clear motivational reasoning and rigorous mathematical formulation of the method.
- Value: ⭐⭐⭐⭐ Solves an important open problem in the 3D dynamic segmentation field, carrying strong directional significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)
- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos](hot3d_hand_and_object_tracking_in_3d_from_egocentric_multi-view_videos.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)

</div>

<!-- RELATED:END -->
