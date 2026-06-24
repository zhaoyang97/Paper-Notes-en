---
title: >-
  [Paper Note] GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency
description: >-
  [CVPR 2025][3D Vision][3D Affordance] GEAL proposes a dual-branch architecture that leverages 3D Gaussian Splatting to render point clouds into realistic 2D images, utilizing the generalization capabilities of pre-trained 2D foundation models. Through granularity-adaptive fusion and 2D-3D consistency alignment, it achieves cross-modal knowledge transfer, outperforming existing 3D affordance methods across both standard and corrupted data benchmarks.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Affordance"
  - "Cross-Modal Alignment"
  - "Gaussian Splatting"
  - "Robustness Benchmark"
  - "2D-3D Knowledge Transfer"
date: 2026-05-08
content_hash: c3336198ab58fc8d
---

# GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency

**Conference**: CVPR 2025  
**arXiv**: [2412.09511](https://arxiv.org/abs/2412.09511)  
**Code**: [https://github.com/DylanOrange/geal](https://github.com/DylanOrange/geal)  
**Area**: 3D Vision  
**Keywords**: 3D Affordance, Cross-Modal Alignment, Gaussian Splatting, Robustness Benchmark, 2D-3D Knowledge Transfer

## TL;DR
GEAL proposes a dual-branch architecture that leverages 3D Gaussian Splatting to render point clouds into realistic 2D images, utilizing the generalization capabilities of pre-trained 2D foundation models. Through granularity-adaptive fusion and 2D-3D consistency alignment, it achieves cross-modal knowledge transfer, outperforming existing 3D affordance methods across both standard and corrupted data benchmarks.

## Background & Motivation

**Background**: 3D affordance learning refers to identifying interactable regions on a 3D object (e.g., the handle of a cup can be "grasped") given semantic prompts (text/image). This is crucial for robot manipulation and human-computer interaction. Prior works typically learn affordances directly from point clouds using 3D backbones like PointNet++.

**Limitations of Prior Work**: (1) Poor generalization—labeled data is scarce (PIAD has only 7,012 point clouds), and 3D backbones lack support from large-scale pre-training, leading to a significant performance drop on unseen object categories; (2) Poor robustness—3D backbones focus on geometric encoding and lack resistance to common real-world noise/corruption (sensor errors, data corruption from complex scenes).

**Key Challenge**: Large-scale pre-trained models like DINO and CLIP provide strong semantic understanding and generalization capabilities in the 2D domain, but the 3D domain lacks pre-training of comparable scale. Transferring knowledge from 2D models to 3D tasks is a key challenge. Directly projecting 3D point clouds to 2D results in sparse points, losing semantic and depth information, and failing to utilize 2D backbones effectively.

**Goal**: (1) Establish an effective 3D→2D mapping to leverage 2D foundation models; (2) Design a cross-modal alignment mechanism to achieve 2D→3D knowledge propagation; (3) Systematically evaluate the robustness of 3D affordance.

**Key Insight**: Use 3D Gaussian Splatting (3DGS) to render sparse point clouds into realistic images—preserving semantic and depth information while generating dense image inputs suitable for 2D backbones like DINO.

**Core Idea**: Bridge 3D point clouds and 2D images via 3DGS, allowing the visual priors of DINO to flow into the 3D branch through consistency alignment, thereby enhancing generalization and robustness.

## Method

### Overall Architecture
GEAL adopts a dual-branch architecture: the 3D branch processes point clouds using PointNet++, and the 2D branch processes depth images rendered via 3DGS using DINOv2. Both branches have independent text encoders (RoBERTa). Visual-textual features are integrated via Granularity-Adaptive Fusion, and cross-modal knowledge is propagated through 2D-3D Consistency Alignment. Finally, a shared transformer decoder predicts affordance scores. Training consists of two stages: first training the 2D branch, and then freezing the 2D branch to train the 3D branch. Inference uses only the 3D branch.

### Key Designs

1. **3DGS-driven 3D-2D Mapping**:

    - **Function**: Render realistic 2D multi-view images from sparse point clouds to provide high-quality inputs for the 2D branch.
    - **Mechanism**: Set the Gaussian centers to the point cloud coordinates $\boldsymbol{\mu} = \mathbf{P}$, with manually fixed covariance $\Sigma$ and opacity $\alpha$ (not updated during training to preserve the original geometry). Render depth maps from $V$ preset camera poses, and then apply a predefined colormap to generate pseudo-color images $\mathbf{I} \in \mathbb{R}^{V \times 3 \times H \times W}$. Simultaneously, render the affordance scores as grayscale colors $\mathbf{c} = \mathbf{y}$ to obtain 2D affordance masks $\mathbf{y}_{2D}$, establishing precise 3D→2D correspondences.
    - **Design Motivation**: The $\alpha$-blending of 3DGS provides smoother transitions, more complete occlusion handling, and more accurate depth perception compared to direct projection, making the rendered images more suitable for feature extraction by DINOv2.

2. **Granularity-Adaptive Fusion Module (GAFM)**:

    - **Function**: Adaptively integrate visual and textual features across multiple feature scales.
    - **Mechanism**: Contains two sub-mechanisms. (a) **Flexible-Granularity Feature Aggregation**: Concatenates multi-scale features from the last $m$ layers, computes layer-wise adaptive weights $\mathbf{W} = \text{Softmax}(\mathbf{f}_{con} \cdot \mathbf{W}_g + \sigma \cdot \epsilon)$ using a gating function with noisy softmax, and then performs a weighted sum to obtain the aggregated features. The introduction of noise enhances exploration during training. (b) **Text-Conditioned Visual Alignment**: Employs a transformer block for text-to-visual cross-attention to enhance textual features, which then query back the visual features, embedding query-relevant information into the visual representation.
    - **Design Motivation**: Affordances may span multiple object parts (e.g., "sitting" involves both the seat and the backrest), requiring multi-granularity features. Since layers in the 3D backbone (PointNet++) have different spatial resolutions and feature dimensions, simple concatenation is insufficient, necessitating text alignment prior to aggregation.

3. **Consistency Alignment Module (CAM)**:

    - **Function**: Transfer rich semantic knowledge from the 2D branch to the 3D branch.
    - **Mechanism**: Use Conv1D to reduce the dimension of 3D enhanced features, and then utilize the 3DGS rendering pipeline to render the feature vector of each point as a Gaussian attribute to 2D, obtaining projected features $\mathbf{F}(v) = \sum_{i} \mathbf{f}_i \alpha_i \prod_{j<i}(1-\alpha_j)$. Meanwhile, use Conv2D to upsample 2D enhanced features to the same resolution. Apply MSE loss $\mathcal{L}_{consis} = \text{MSE}(\mathbf{f}_{cam}^{3D-2D}, \mathbf{f}_{cam}^{2D})$ to force the 3D projected features to align with 2D features in a shared space.
    - **Design Motivation**: The 2D branch uses frozen DINOv2 to extract features containing rich semantic priors. Aligning at the feature level rather than distilling at the output level facilitates finer-grained knowledge transfer. The differentiable rendering of 3DGS allows gradients from the 2D alignment loss to propagate back to the 3D branch.

### Loss & Training
2D branch loss: $\mathcal{L}^{2D} = \mathcal{L}_{BCE}^{2D} + \mathcal{L}_{Dice}^{2D}$.  
3D branch loss: $\mathcal{L}^{3D} = \mathcal{L}_{BCE}^{3D} + \mathcal{L}_{Dice}^{3D} + \mathcal{L}_{consis}$.  
Two-stage training: (1) Train the 2D branch for 50 epochs; (2) Freeze the 2D branch (except CAM) and train the 3D branch for 50 epochs. Adam optimizer, lr=1e-4, step learning rate scheduling. DINOv2 is frozen, RoBERTa is fine-tuned. Inference only uses the 3D branch, with no rendering required.

## Key Experimental Results

### Main Results
PIAD Dataset (7,012 point clouds, 23 object categories, 17 affordance types):

| Method | Split | aIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|------|-------|------|------|------|
| IAGNet | Seen | 20.5 | 84.9 | 0.545 | 0.098 |
| LASO | Seen | 19.7 | 84.2 | 0.590 | 0.096 |
| **GEAL** | **Seen** | **22.5** | **85.0** | **0.600** | **0.092** |
| IAGNet | Unseen | 8.0 | 71.8 | 0.352 | 0.127 |
| LASO | Unseen | 8.0 | 69.2 | 0.386 | 0.118 |
| **GEAL** | **Unseen** | **8.7** | **72.5** | **0.390** | **0.102** |

LASO Dataset (19,751 pairs, 8,434 objects):

| Method | Split | aIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|------|-------|------|------|------|
| LASO | Seen | 20.8 | 87.3 | 0.629 | 0.093 |
| **GEAL** | **Seen** | **22.0** | 86.7 | **0.634** | **0.092** |
| LASO | Unseen | 14.6 | 80.2 | 0.507 | 0.119 |
| **GEAL** | **Unseen** | **16.7** | **80.9** | **0.567** | **0.106** |

### Ablation Study
(Based on module contribution analysis described in the paper)

| Component | Function |
|------|------|
| 3DGS Mapping | Converts sparse point clouds into dense inputs that can be processed by DINOv2 |
| GAFM | Multi-granularity feature aggregation + text-conditioned alignment |
| CAM | Core bridge for 2D→3D knowledge transfer |
| Two-stage Training | First solidify 2D knowledge, then transfer to 3D |

### Key Findings
- The improvement on Unseen objects is the most significant: PIAD Unseen MAE decreases from 0.118 to 0.102 (13.6%↓), indicating that 2D priors indeed enhance generalization.
- On LASO Unseen, aIoU increases by 14.4% (14.6→16.7) and SIM increases by 11.8% (0.507→0.567), showing outstanding cross-category generalization.
- The authors also established two corrupted data benchmarks, PIAD-C and LASO-C (7 corruption types × 5 severities), providing the first systematic evaluation of the robustness of 3D affordance learning methods.
- As a frozen 2D backbone, DINOv2 provides stable feature anchors, stabilizing the training of the 3D branch.
- Inference only requires the 3D branch with no extra computational cost—the 2D branch is only utilized during the training phase.

## Highlights & Insights
- **Novel use of 3DGS as a cross-modal bridge**: Instead of using 3DGS for rendering or reconstruction, it is leveraged as a differentiable mapping tool from 3D to 2D, skillfully bypassing the issue where "directly projecting point clouds is too sparse". This paradigm can be generalized to any scenario requiring the transformation of 3D features into 2D representations.
- **Zero extra cost during inference**: Although a complete 2D branch is trained, it is entirely discarded during inference, utilizing only the 3D branch. This "distill during training, independent during inference" design achieves an optimal balance between efficiency and effectiveness.
- **Value of the corruption benchmarks**: PIAD-C and LASO-C represent the first robustness evaluation benchmarks in this field (4,890 object-action pairs, 7 corruptions, 5 severities), filling an critical evaluation gap.

## Limitations & Future Work
- The current 3D backbone is PointNet++, which is relatively basic; transitioning to Point Transformer or 3D foundation models could yield further improvements.
- The coloration of depth maps rendered by 3DGS uses a handcrafted colormap, which may not be the optimal visual feature input.
- Training requires rendering from multiple views ($V$ views), which increases training time and GPU memory consumption.
- The design of text prompts ("Given a depth map of a [object] in [view]") is relatively simple; more sophisticated prompt engineering might be beneficial.
- Only single-object scenes were evaluated; affordance identification in multi-object interaction scenes remains unexplored.

## Related Work & Insights
- **vs LASO**: Also utilizes text conditioning for open-vocabulary affordance detection but relies solely on a 3D backbone. GEAL is significantly stronger on Unseen classes due to 2D priors (aIoU +14.4%).
- **vs IAGNet**: An image-point cloud cross-modal method, but uses hand-crafted image features. GEAL leverages pre-trained DINOv2 to provide stronger semantic understanding.
- **Intersection with the 3DGS community**: Gaussian splatting is used here not for rendering quality, but to establish differentiable 3D-2D correspondences. This opens up new pathways for using 3DGS in downstream visual understanding tasks.
- One could consider expanding GEAL's framework to robot manipulation (from affordance to grasp planning).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of 3DGS as a cross-modal bridge and consistency alignment is novel, although the individual components themselves are not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two standard datasets + two corruption benchmarks + complete baseline comparisons + unseen category evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagram, detailed method explanations, and standardized equations.
- Value: ⭐⭐⭐⭐ The corruption benchmarks represent a substantial contribution, the framework concept is highly transferable, and it has direct utility for robotics applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CrossOver: 3D Scene Cross-Modal Alignment](crossover_3d_scene_cross-modal_alignment.md)
- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)
- [\[CVPR 2025\] 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)
- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)
- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](../../CVPR2026/3d_vision/affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)

</div>

<!-- RELATED:END -->
