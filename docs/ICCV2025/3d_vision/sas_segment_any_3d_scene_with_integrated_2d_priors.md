---
title: >-
  [Paper Note] SAS: Segment Any 3D Scene with Integrated 2D Priors
description: >-
  [ICCV 2025][3D Vision][Open-vocabulary 3D segmentation] This paper proposes SAS, a framework that for the first time integrates the complementary capabilities of multiple 2D open-vocabulary models to learn better 3D representations. It aligns feature spaces across models via Model Alignment via Text, and quantifies per-category model recognition capability using diffusion-synthesized images through Annotation-Free Model Capability Construction. These components jointly guide multi-model feature fusion and 3D distillation, achieving substantial improvements over prior work on ScanNet v2, Matterport3D, and nuScenes.
tags:
  - ICCV 2025
  - 3D Vision
  - Open-vocabulary 3D segmentation
  - multi-model fusion
  - knowledge distillation
  - diffusion models
  - model capability construction
date: 2026-05-08
content_hash: 4bc6ec82ea560a39
---

# SAS: Segment Any 3D Scene with Integrated 2D Priors

**Conference**: ICCV 2025
**arXiv**: [2503.08512](https://arxiv.org/abs/2503.08512)
**Code**: [Project Page](https://peoplelu.github.io/SAS.github.io)
**Area**: 3D Vision
**Keywords**: Open-vocabulary 3D segmentation, multi-model fusion, knowledge distillation, diffusion models, model capability construction

## TL;DR

This paper proposes SAS, a framework that for the first time integrates the complementary capabilities of multiple 2D open-vocabulary models to learn better 3D representations. It aligns feature spaces across models via Model Alignment via Text, and quantifies per-category model recognition capability using diffusion-synthesized images through Annotation-Free Model Capability Construction. These components jointly guide multi-model feature fusion and 3D distillation, achieving substantial improvements over prior work on ScanNet v2, Matterport3D, and nuScenes.

## Background & Motivation

3D scene understanding is a fundamental task for autonomous driving, virtual reality, and robotic manipulation. Traditional closed-set methods are limited to fixed category sets and cannot recognize unseen classes, motivating growing interest in open-vocabulary 3D understanding.

The dominant approach transfers capabilities of 2D open-vocabulary models (e.g., LSeg, SEEM) to 3D models via feature distillation. However, a fundamental problem exists:

**Error propagation from a single teacher model**: 2D models make mistakes on certain categories (e.g., SEEM misidentifies "picture" as "wall"), and distillation causes the 3D model to inherit the same errors.

**The intuitive solution—multi-model fusion—faces two key challenges**:
- **Misaligned feature spaces**: Different 2D models (e.g., LSeg uses CLIP, SEEM uses its own encoder) have incompatible image-text feature spaces and cannot be directly fused.
- **Difficulty in quantifying model capability**: Knowing which model performs better on which categories requires test images and annotations, which is impractical in zero-shot settings.

**Core Insight**: Different 2D models have complementary recognition capabilities across categories. If per-category capability can be quantified for each model, more reliable model features can be selected for each point, correcting single-model misidentifications.

## Method

### Overall Architecture

SAS consists of four stages: (1) aligning the feature spaces of multiple 2D models via text bridging; (2) evaluating per-category model capabilities using diffusion-synthesized images; (3) fusing point features from multiple models guided by the constructed capability estimates; and (4) transferring the fused knowledge to a 3D network via superpoint distillation and temporal ensembling self-distillation.

### Key Designs

1. **Model Alignment via Text**: Different 2D models reside in different feature spaces. The core mechanism uses text as a unified bridge. For each mask output by SEEM, a pre-trained captioner (TAP) first generates a descriptive caption (including color, shape, etc.), after which nouns in the caption are replaced with SEEM's predicted labels to improve semantic accuracy. A CLIP text encoder then uniformly encodes these captions. This aligns LSeg (natively aligned to CLIP) and SEEM (bridged to CLIP via text) into a shared CLIP feature space. Pixel-to-point correspondences are used to map pixel features to 3D points, yielding aligned point features $F^{2D}_L$ and $F^{2D}_S$.

2. **Annotation-Free Model Capability Construction**: This is the most innovative component. The challenge is evaluating model capabilities without test data or annotations. The solution proceeds as follows:

    - Stable Diffusion is used to generate $m$ synthetic images for each category in a predefined vocabulary.
    - Cross-attention maps $M_x$ from the diffusion model localize the target object, and SAM refines them into precise masks serving as pseudo-labels $\mathbf{M}^{Pseudo}_{i,j}$.
    - LSeg and SEEM are applied to the synthetic images to obtain their respective mask predictions.
    - mIoU measures each model's recognition capability per category:
    $S^{LSeg}_j = \frac{1}{m}\sum_{i=1,...,m} \mathbf{mIoU}(\mathbf{M}^{Pseudo}_{i,j}, \mathbf{M}^{LSeg}_{i,j})$
    - A capability vector for each model is constructed as $S_L = [S^{LSeg}_1, ..., S^{LSeg}_K]$.

3. **Capability-Guided Feature Fusion**:

    - CLIP encodes the vocabulary to obtain text features $F_{text}$.
    - The predicted category for each point under LSeg and SEEM is computed as $\mathcal{P}_{LSeg}$ and $\mathcal{P}_{SEEM}$.
    - The sum of capability scores for the two candidate categories serves as the probability of each model making a correct prediction.
    - Temperature-controlled softmax weighting fuses the features:
    $F^{2D}_{fusion} = \frac{\exp(\mathcal{P}_{LSeg}/\tau)}{\exp(\mathcal{P}_{LSeg}/\tau) + \exp(\mathcal{P}_{SEEM}/\tau)} F^{2D}_L + \frac{\exp(\mathcal{P}_{SEEM}/\tau)}{\exp(\mathcal{P}_{LSeg}/\tau) + \exp(\mathcal{P}_{SEEM}/\tau)} F^{2D}_S$

4. **Superpoint Distillation + Temporal Ensembling Self-Distillation**:

    - Superpoint distillation: Semantically consistent superpoints are extracted, their features averaged, and distillation is performed at the superpoint level, smoothing inconsistent 2D predictions.
    - Temporal ensembling self-distillation: An EMA accumulates historical outputs $\hat{F}^{3D} = \alpha \hat{F}^{3D} + (1-\alpha) F^{3D}$; pseudo-labels derived from historical predictions supervise the current model. This is more stable than GGSD's Mean-teacher, which risks training collapse when a variable student model supervises the teacher.

### Loss & Training

Distillation stage:
- **Superpoint distillation loss**: $\mathcal{L} = \mathcal{L}_p + \mathcal{L}_{sp}$, computing cosine similarity loss at both point and superpoint levels.
- **Self-distillation loss**: $\mathcal{L} = \mathcal{L}^{ST}_p + \mathcal{L}^{ST}_{sp}$, using cross-entropy loss.

Training runs for 100 epochs: superpoint distillation only for the first 70 epochs, with self-distillation added for the final 30. Different pre-built vocabularies are used for indoor and outdoor scenes. For nuScenes, where point clouds are predominantly road-centric, superpoints are not used (each point is treated as its own superpoint).

## Key Experimental Results

### Main Results

| Dataset | Metric (mIoU) | Ours | Prev. SOTA (OV3D) | Gain |
|--------|------------|------|-----------------|------|
| ScanNet v2 | mIoU | 61.9 | 57.3 | +4.6 |
| Matterport3D | mIoU | 48.6 | 45.8 | +2.8 |
| nuScenes | mIoU | 47.5 | 44.6 (Seal=45.0) | +2.5 |

SAS achieves state-of-the-art performance among zero-shot methods on all three datasets. The mIoU of 61.9 on ScanNet v2 approaches that of some older fully supervised methods (e.g., PointConv at 61.0).

### Ablation Study

| Configuration | ScanNet v2 | Matterport | nuScenes | Note |
|------|-----------|-----------|---------|------|
| 2D LSeg features | 51.2 | 38.6 | - | LSeg only |
| 2D SEEM features | 47.3 | 40.2 | 37.8 | SEEM only |
| 2D direct addition | 48.6 | 39.1 | 34.1 | Naive fusion ineffective |
| 2D linear fusion | 49.9 | 39.4 | 34.8 | Limited gain |
| 2D proposed fusion | 55.5 | 43.6 | 40.0 | Capability-guided significantly better |
| 3D pixel-point distillation | 56.7 | 45.1 | 45.4 | Distillation surpasses 2D features |
| 3D superpoint distillation | 59.2 | 46.3 | 45.4 | Structural information beneficial |
| 3D +self-distillation (full) | 61.9 | 48.6 | 47.5 | Further gains |

Extending to 3 2D models (+ODISE): ScanNet 62.5 (+0.6), Matterport 49.8 (+1.2), demonstrating scalability.

Long-tail evaluation: SAS significantly outperforms OpenScene under Matterport K=40/80/160 category settings (e.g., K=160: 8.1 vs. 5.8).

### Key Findings

- SEEM outperforms LSeg on Matterport3D but is weaker on ScanNet v2, confirming genuine complementarity between the two models.
- Naive addition or linear fusion can perform worse than the best single model, demonstrating the necessity of capability-guided fusion.
- The distilled 3D model consistently outperforms 2D fused features, reflecting gains from 3D spatial consistency.
- Superpoint distillation and self-distillation each contribute approximately 2–3 mIoU points independently.
- The method's advantage is more pronounced under long-tail categories (K=160), reflecting strong open-vocabulary generalization.

## Highlights & Insights

- **Core innovation**: Using diffusion-synthesized images to quantify teacher model capabilities resolves the fundamental challenge of evaluating models in zero-shot settings—an elegant and broadly applicable idea.
- Text-bridged alignment is an elegant solution to the feature space incompatibility problem across multiple models.
- Temporal ensembling self-distillation is more stable than GGSD's Mean-teacher, avoiding training collapse caused by supervising the teacher with a varying student.
- At inference time, only the 3D model output is used directly, without 2D–3D ensembling, yielding higher efficiency.

## Limitations & Future Work

- Synthetic image quality is bounded by Stable Diffusion's generation capability, which may be insufficient for fine-grained categories.
- Cross-attention map localization is not always precise, and SAM refinement may also introduce errors.
- Only 2–3 2D models are fused; fusion strategies and efficiency for larger model ensembles remain unexplored.
- The quality of superpoint extraction (e.g., color/normal clustering) affects distillation performance.
- The possibility of using 3D geometric information to guide 2D feature selection has not been explored.

## Related Work & Insights

- OpenScene established the 2D→3D feature distillation paradigm; SAS makes a significant contribution along the direction of improving 2D teacher features.
- GGSD's Mean-teacher self-distillation inspired SAS's temporal ensembling design.
- Cross-attention maps from Stable Diffusion are cleverly repurposed for object localization, extending the utility of diffusion models as general-purpose tools.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to propose annotation-free model capability construction; highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three major datasets, long-tail evaluation, multi-task extension, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ The multi-teacher fusion paradigm has broad implications for open-vocabulary 3D understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Online Segment Any 3D Thing as Instance Tracking](../../NeurIPS2025/3d_vision/online_segment_any_3d_thing_as_instance_tracking.md)
- [\[ICCV 2025\] Scene Coordinate Reconstruction Priors](scene_coordinate_reconstruction_priors.md)
- [\[ICCV 2025\] Find Any Part in 3D](find_any_part_in_3d.md)
- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)

</div>

<!-- RELATED:END -->
