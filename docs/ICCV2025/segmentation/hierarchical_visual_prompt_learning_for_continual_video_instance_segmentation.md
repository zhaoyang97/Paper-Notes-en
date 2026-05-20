---
title: >-
  [Paper Note] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation
description: >-
  [ICCV 2025][Segmentation][Continual Learning] This paper introduces Continual Video Instance Segmentation (CVIS) as a new problem formulation, and proposes the Hierarchical Visual Prompt Learning (HVPL) model…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Continual Learning"
  - "Video Instance Segmentation"
  - "Visual Prompt Learning"
  - "Catastrophic Forgetting"
  - "Orthogonal Gradient Correction"
date: 2026-05-08
content_hash: 7204ef023ead2fed
---

# Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation

**Conference**: ICCV 2025
**arXiv**: [2508.08612](https://arxiv.org/abs/2508.08612)  
**Code**: [GitHub](https://github.com/JiahuaDong/HVPL)  
**Area**: Image Segmentation
**Keywords**: Continual Learning, Video Instance Segmentation, Visual Prompt Learning, Catastrophic Forgetting, Orthogonal Gradient Correction

## TL;DR

This paper introduces Continual Video Instance Segmentation (CVIS) as a new problem formulation, and proposes the Hierarchical Visual Prompt Learning (HVPL) model, which mitigates catastrophic forgetting of old categories via forgetting compensation mechanisms at both frame-level and video-level.

## Background & Motivation

Video Instance Segmentation (VIS) requires simultaneously detecting, segmenting, and tracking object instances across consecutive frames. However, existing methods assume a fixed category set, which is unrealistic in practice—new categories emerge continuously. Directly fine-tuning on new categories leads to severe forgetting of previously learned ones, i.e., catastrophic forgetting.

**Limitations of Prior Work**:
1. Traditional VIS methods require retaining video data from all old categories for retraining, incurring prohibitive computational and memory costs.
2. Naively combining continual learning (CL) with VIS can only alleviate forgetting at the frame level, failing to exploit global video context to address video-level forgetting.
3. Knowledge distillation methods are ineffective in CVIS due to large appearance variations across instances; multi-task incremental learning further exacerbates distillation difficulty.

**Core Problem**: A rehearsal-free method is needed that addresses catastrophic forgetting jointly from both frame-level and video-level perspectives, without storing training data from old tasks.

## Method

### Overall Architecture

HVPL adopts a pre-trained Mask2Former as the frame-level detector (with frozen parameters) and introduces two levels of prompt learning on top:
- **Frame-level**: Task-specific frame prompts + Orthogonal Gradient Correction (OGC) module
- **Video-level**: Task-specific video prompts + Video Context Decoder (comprising GSS and MSA layers)

### Key Designs

1. **Task-Specific Frame Prompts + Orthogonal Gradient Correction (OGC)**:

    - For each new task $t$, learnable frame prompts $\mathbf{P}_{\mathrm{frm}}^{t} \in \mathbb{R}^{L_p^f \times D}$ are introduced to encode global instance information for new categories.
    - Core Idea of OGC: the gradient update $\triangle\mathbf{P}$ for frame prompts must satisfy $\triangle\mathbf{P}(\mathcal{O}^{t-1})^\top = \mathbf{0}$, i.e., the gradient is projected onto the orthogonal complement of the old tasks' feature space.
    - The orthogonal basis $\mathbf{V}_0^{t-1}$ is obtained by applying SVD to the representative feature space $\mathcal{O}^{t-1}$ of old tasks, and the corrected gradient is computed as $\triangle\mathbf{P}^* = \triangle\mathbf{P}\hat{\mathbf{V}}_0^{t-1}(\hat{\mathbf{V}}_0^{t-1})^\top$.
    - An elastic threshold $\xi \in [0,1]$ controls the size of the orthogonal subspace, balancing new-task learning and old-task retention (optimal at $\xi=0.7$ empirically).
    - Design Motivation: This ensures that new-task learning does not interfere with attention computation in the old categories' feature space, providing a mathematical guarantee for minimizing frame-level forgetting.

2. **Graph-guided State Space Module (GT-SSM)**:

    - Located in the GSS layer of the Video Context Decoder, it captures structured inter-class relationships across frames.
    - An undirected $\varphi$-connected graph $\mathcal{G}$ is constructed, with noisy edges removed via a minimum spanning tree $\mathcal{G}_T$.
    - Based on the discretized variable $\bar{\mathbf{A}}_j = \exp(\triangle_j \mathbf{A}_j)$ from the state space model, hidden states are computed via graph traversal.
    - A dynamic traversal strategy (Algorithm 1) is proposed, reducing complexity from $\mathcal{O}(N_v^2)$ to $\mathcal{O}(N_v)$.
    - Design Motivation: Conventional Mamba relies on fixed traversal orders and cannot model the intrinsic structural relationships among cross-frame instances. GT-SSM adaptively traverses via graph structure to capture information propagation among semantically related instances.

3. **Task-Specific Video Prompts + MSA Layer**:

    - Learnable video prompts $\mathbf{P}_{\mathrm{vid}}^{t} \in \mathbb{R}^{L_p^v \times D}$ encode global video-level context.
    - The MSA layer uses video prompts as Query and the enhanced features from the GSS layer as Key/Value, capturing task-specific global video context via multi-head self-attention.
    - The output video prompt features $\mathbf{F}_{\mathrm{vid}}^{L_m}$ integrate cross-frame structural relationships and global semantic information.
    - Design Motivation: Frame-level compensation neglects the intrinsic relationships among instances across frames; the absence of global semantic context makes video-level forgetting difficult to address.

### Loss & Training

- Standard detection loss from Mask2Former is adopted.
- The backbone, pixel decoder, and Transformer decoder of Mask2Former are frozen during training.
- Prompt lengths for the first task are set to $L_p^f = L_p^v = 100$; subsequent tasks use $L_p^f = L_p^v = 10$.
- At inference, frame and video prompts from all tasks are concatenated for joint prediction.
- The storage for the old-task feature space $\mathcal{O}^{t-1}$ is less than 0.1M parameters and is negligible.

## Key Experimental Results

### Main Results

| Dataset / Setting | Metric | HVPL | ECLIPSE (CVPR'24) | CoMFormer+NeST | Gain |
|---|---|---|---|---|---|
| OVIS 15-5 | AP | **11.09** | 6.62 | 4.44 | +4.47 |
| OVIS 15-5 | FAP↓ | **9.56** | 28.08 | 32.25 | −18.52 |
| OVIS 15-10 | AP | **11.92** | 6.82 | 5.87 | +5.10 |
| YT-VIS 2021 30-10 | AP | **43.24** | 29.68 | 28.70 | +13.56 |
| YT-VIS 2021 20-4 | AP | **35.05** | 30.52 | 15.37 | +4.53 |
| YT-VIS 2019 20-5 | AP | **34.79** | 32.62 | 21.39 | +2.17 |
| YT-VIS 2019 20-2 | AP | **31.68** | 30.24 | 10.29 | +1.44 |

### Ablation Study

| Configuration | AP (30-10) | FAP↓ | Description |
|---|---|---|---|
| Base (fine-tune Mask2Former only) | 27.53 | 96.48 | No anti-forgetting mechanism |
| +TFP (frame prompts) | 30.02 | 74.88 | Frame prompts partially alleviate forgetting |
| +TFP+TVP (+video prompts+MSA) | 39.07 | 31.05 | Video-level compensation yields large gains |
| +TFP+TVP+GSS | 40.37 | 30.40 | GSS further models cross-frame relationships |
| Full (+OGC) | **43.24** | **15.69** | OGC orthogonal gradient correction finalizes the framework |

### Key Findings

- HVPL has only 0.92M trainable parameters, far fewer than the 41.92M of knowledge distillation methods, yet substantially outperforms them.
- Forgetting rates FAP and FAR1 are significantly reduced; e.g., FAP on OVIS 15-5 is only 9.56% (vs. 28.08% for ECLIPSE).
- Performance is particularly strong in multi-task incremental settings (e.g., YT-VIS 2021 20-4 with 6 tasks), consistently leading at each incremental step.
- $\xi=0.7$ is the optimal threshold for balancing new- and old-task learning; performance remains stable for $\xi \in [0.2, 1.0]$.

## Highlights & Insights

- **New Problem Formulation**: The paper is the first to formally define the CVIS problem, filling a gap in the application of continual learning to video instance segmentation.
- **Hierarchical Forgetting Compensation**: The two-tier design—frame-level (OGC orthogonal projection) + video-level (GT-SSM + video prompts)—is systematic and well-motivated.
- **Rigorous Theoretical Derivation**: The orthogonal gradient condition is derived from attention equivalence, providing strong theoretical guarantees.
- **GT-SSM**: Combining Mamba with graph structures for cross-frame relationship modeling is a novel contribution.
- **Extreme Parameter Efficiency**: HVPL with 0.92M parameters substantially outperforms methods with 42M parameters.

## Limitations & Future Work

- Only ResNet-50 is used as the backbone; performance under stronger backbones (e.g., Swin-L) is not validated.
- The rehearsal-free CVIS setting excludes old-task data replay; incorporating a small replay buffer may further improve performance in practice.
- The minimum spanning tree construction in GT-SSM relies on cosine similarity, which may be insufficiently robust for scenes with drastic appearance changes.
- Integration with large language models for large-scale continual tasks remains unexplored (noted by the authors as future work).

## Related Work & Insights

- HVPL shares the prompt learning paradigm with ECLIPSE (CVPR'24) but adds video-level compensation, yielding substantial performance gains.
- The orthogonal gradient projection idea originates from OGD-series methods in continual learning; applying it to prompt learning is an innovative contribution.
- GT-SSM integrates Vision Mamba and graph neural networks, offering inspiration for other tasks requiring cross-frame structural modeling.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce the CVIS problem; hierarchical prompt learning framework is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets with multiple settings and thorough ablations; experiments with stronger backbones are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and notation is well-defined.
- **Value**: ⭐⭐⭐⭐⭐ Pioneering significance for the application of continual learning to video segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](cavis_context-aware_video_instance_segmentation.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[ICCV 2025\] Region-based Cluster Discrimination for Visual Representation Learning](region-based_cluster_discrimination_for_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
