---
title: >-
  [Paper Note] GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation
description: >-
  [ICLR 2026][3D Vision][Open-vocabulary 3D segmentation] GeoPurify is proposed as a framework that purifies noisy features projected from 2D VLMs into 3D by distilling geometric priors from a 3D self-supervised teacher model, achieving performance on par with or superior to full-data SOTA open-vocabulary 3D segmentation using only ~1.5% of training data.
tags:
  - ICLR 2026
  - 3D Vision
  - Open-vocabulary 3D segmentation
  - knowledge distillation
  - geometric priors
  - VLM feature purification
  - data efficiency
date: 2026-05-08
content_hash: 8b8695f0cb977378
---

# GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation

**Conference**: ICLR 2026
**arXiv**: [2510.02186](https://arxiv.org/abs/2510.02186)
**Code**: [Available](https://github.com/tj12323/GeoPurify)
**Area**: 3D Vision
**Keywords**: Open-vocabulary 3D segmentation, knowledge distillation, geometric priors, VLM feature purification, data efficiency

## TL;DR

GeoPurify is proposed as a framework that purifies noisy features projected from 2D VLMs into 3D by distilling geometric priors from a 3D self-supervised teacher model, achieving performance on par with or superior to full-data SOTA open-vocabulary 3D segmentation using only ~1.5% of training data.

## Background & Motivation

Open-vocabulary 3D scene understanding aims to enable models to recognize objects described by arbitrary text. The core challenge lies in a fundamental trade-off when transferring 2D VLM semantics to 3D:

**Training-free methods**: Directly project multi-view 2D predictions onto 3D point clouds and aggregate them, resulting in severe geometric inconsistencies.

**Training-based methods**: Learn point-level 3D-semantic mappings, but require large-scale annotated data.

**Key hypothesis**: When VLM features are transferred from 2D to 3D, geometric information is not destroyed but becomes latent, and can be recovered through efficient means rather than being learned from scratch.

## Method

### Overall Architecture

GeoPurify operates in two phases:

- **Training phase**: A Student Affinity Network learns 3D structure by mimicking a frozen 3D SSL teacher model (Sonata) via contrastive distillation. No 3D semantic labels are required.
- **Inference phase**: A frozen 2D VLM (X-Decoder) generates initial 3D features, which are then refined through geometry-aware pooling using the pretrained Student network.

### Key Designs

**(1) Semantic Initialization from a General-Purpose VLM**

X-Decoder is adopted instead of the conventional "segment-then-match" pipeline (e.g., LSeg, OpenSeg, SAM+CLIP). X-Decoder follows a "segment-as-understanding" paradigm, and its unified vision-language embedding space provides a higher semantic ceiling. Per-point features are sampled from all visible views and aggregated via weighted averaging.

**(2) Geometric Contrastive Distillation**

Teacher model: frozen Sonata (a 3D self-supervised foundation model) providing a robust geometric target space. Student model: a trainable sparse 3D CNN outputting 128-dimensional geometric embeddings.

An efficient hybrid negative sampling strategy is employed:

- **Macro-negatives** (48 samples): globally least-similar points, enabling learning of overall scene structure.
- **Micro-negatives** (16 samples): spatially neighboring points with the most dissimilar features, for fine-grained boundary discrimination.

InfoNCE contrastive loss with temperature $\tau = 0.07$ and 4096 anchor points per scene.

**(3) Geometry-Guided Pooling (Inference)**

1. The Student network generates geometric embeddings for each voxel.
2. A sparse affinity matrix $A$ is constructed using K-nearest neighbors with sharpened softmax ($\alpha = 1/20$).
3. Iterative pooling: $F^{(t+1)} = A \cdot F^{(t)}$, for $T=18$ iterations.
4. Refined voxel features are mapped back to the original points.

### Loss & Training

- **Loss**: InfoNCE contrastive loss, temperature 0.07
- **Optimizer**: AdamW, lr $10^{-3}$, cosine annealing, 50 epochs
- **Training scale**: Only 20 scenes (~1.6% of ScanNetV2), without 3D semantic labels
- **Subset selection**: Joint scoring based on Shannon entropy (semantic complexity) and category count (semantic richness), with K-Means clustering to ensure environmental diversity
- **Hardware**: Single NVIDIA L40 GPU

## Key Experimental Results

### Main Results: Open-Vocabulary 3D Semantic Segmentation

| Method | Training Data | ScanNetV2 mIoU | ScanNetV2 mAcc | Matterport3D mIoU | Matterport3D mAcc |
|--------|---------------|----------------|----------------|--------------------|--------------------|
| OpenScene-3D | 100% | 51.6 | 63.1 | 40.5 | 48.8 |
| CUA-O3D (3D) | 100% | 54.1 | 64.1 | 41.3 | 49.5 |
| OV3D | 100% | 57.3 | 72.9 | 45.8 | 62.4 |
| CUA-O3D (same data) | ~1.5% | 18.1 | 26.4 | 14.0 | 20.5 |
| **GeoPurify** | **~1.5%** | **55.1** | **72.5** | **40.2** | **62.4** |

### Cross-Dataset Transfer

| Direction | OpenScene | CUA-O3D | **GeoPurify** |
|-----------|-----------|---------|---------------|
| ScanNetV2 → Matterport3D mIoU | 36.0 | 37.4 | **40.5** |
| Matterport3D → ScanNetV2 mIoU | 36.5 | 38.6 | **54.9** |

### Ablation Study

| Component | Setting | mIoU | mAcc |
|-----------|---------|------|------|
| No geometric purification | Direct 2D feature aggregation | 50.2 | 68.1 |
| + GeoPurify | Full framework | **55.1** | **72.5** |
| 2D backbone | LSeg | 48.6 | 61.6 |
| 2D backbone | LSeg + GeoPurify | 51.2 | 63.0 |
| Sampling strategy | Macro-negatives only | 53.5 | 70.8 |
| Sampling strategy | Hybrid (full) | **55.1** | **72.5** |
| Pooling iterations | T=1 / T=18 / T=36 | 52.3 / **55.1** / 55.1 | 70.2 / **72.5** / 72.4 |
| Training scenes | 10 / 20 / 50 | 54.7 / **55.1** / 55.0 | 72.4 / **72.5** / 72.5 |

### Key Findings

1. **Extreme data efficiency**: ~1.5% of data achieves parity with full-data competitors (55.1 vs. 54.1); under the same data budget, CUA-O3D collapses to 18.1.
2. **Geometric purification gain of +4.9 mIoU**: from 50.2 to 55.1.
3. **Micro-negatives are critical**: contributing +1.6 mIoU in boundary accuracy.
4. **Saturation at 20 scenes**: significant improvement from 10 to 20 scenes, with convergence thereafter.
5. **Strong cross-dataset transfer advantage**: Matterport3D → ScanNetV2 reaches 54.9 mIoU, outperforming CUA-O3D by 16.3 points.

## Highlights & Insights

- **Recovering latent structure vs. learning from scratch**: The core hypothesis is highly insightful — transferring features from 2D to 3D does not destroy geometric information.
- **Robustness of the decoupled design**: Semantics are delegated to the VLM while geometry is handled by the Student network, each operating independently.
- **Class-agnostic geometric priors**: The learned geometric associations are independent of semantic categories, yielding strong cross-dataset transferability.
- **Data selection strategy**: Scene selection based on Shannon entropy is more efficient than random selection.

## Limitations & Future Work

1. **mIoU vs. mAcc trade-off**: Geometry-guided pooling improves recall but may introduce semantic bleeding at boundaries.
2. **Performance ceiling bounded by VLM quality**: Convergence at 20 scenes indicates the bottleneck lies in VLM semantic quality rather than geometric learning.
3. **Over-smoothing in iterative pooling**: Performance degrades for $T > 18$.
4. **Outdoor scenes unexplored**: Validation is limited to indoor benchmarks.

## Related Work & Insights

- **OpenScene**: Large-scale 3D knowledge distillation; GeoPurify matches it with 1.5% of the data.
- **CUA-O3D**: Full-data training SOTA; GeoPurify significantly surpasses it in the low-data regime.
- **Sonata**: 3D self-supervised teacher model providing geometric priors.
- **Insight**: Decoupling semantic and geometric learning may be a key paradigm for data-efficient 3D scene understanding.

## Rating

- Novelty: 4/5 — The hypothesis of recovering latent geometric structure and the decoupled framework design are highly original.
- Technical Depth: 4/5 — The combination of contrastive distillation and geometry-guided pooling is elegantly designed.
- Experimental Thoroughness: 5/5 — Three major benchmarks, cross-dataset evaluation, and comprehensive ablation studies.
- Value: 5/5 — Reaching SOTA with 1.5% of training data demonstrates strong practical deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](../../CVPR2026/3d_vision/jopp3d_joint_open_vocabulary_semantic_segmentation.md)
- [\[ICLR 2026\] CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D](core-3d_context-aware_open-vocabulary_retrieval_by_embeddings_in_3d.md)
- [\[AAAI 2026\] Retrieving Objects from 3D Scenes with Box-Guided Open-Vocabulary Instance Segmentation](../../AAAI2026/3d_vision/retrieving_objects_from_3d_scenes_with_box-guided_open-vocabulary_instance_segme.md)
- [\[ICLR 2026\] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)
- [\[CVPR 2026\] LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds](../../CVPR2026/3d_vision/lightsplat_fast_and_memory-efficient_open-vocabulary_3d_scene_understanding_in_f.md)

</div>

<!-- RELATED:END -->
