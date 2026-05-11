---
title: >-
  [Paper Note] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation
description: >-
  [CVPR 2026][Segmentation][Open-vocabulary 3D semantic segmentation] This paper proposes GeoGuide, a hierarchical geometric guidance framework for open-vocabulary 3D semantic segmentation. It leverages geometric priors from pretrained 3D models to correct geometric bias in 2D-to-3D knowledge distillation via three complementary modules: uncertainty-based superpoint distillation, instance-level mask reconstruction, and inter-instance relation consistency. GeoGuide achieves state-of-the-art performance of 64.8 mIoU on ScanNet v2.
tags:
  - CVPR 2026
  - Segmentation
  - Open-vocabulary 3D semantic segmentation
  - geometric priors
  - 2D-to-3D distillation
  - superpoint aggregation
  - instance-level consistency
date: 2026-05-08
content_hash: d551edecc4182b24
---

# GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.26260](https://arxiv.org/abs/2603.26260)
**Code**: N/A
**Area**: Segmentation
**Keywords**: Open-vocabulary 3D semantic segmentation, geometric priors, 2D-to-3D distillation, superpoint aggregation, instance-level consistency

## TL;DR

This paper proposes GeoGuide, a hierarchical geometric guidance framework for open-vocabulary 3D semantic segmentation. It leverages geometric priors from pretrained 3D models to correct geometric bias in 2D-to-3D knowledge distillation via three complementary modules: uncertainty-based superpoint distillation, instance-level mask reconstruction, and inter-instance relation consistency. GeoGuide achieves state-of-the-art performance of 64.8 mIoU on ScanNet v2.

## Background & Motivation

1. **Background**: Open-vocabulary 3D semantic segmentation aims to segment arbitrary categories beyond the training set. Due to the scarcity of 3D point-text paired data, dominant approaches distill knowledge from pretrained 2D open-vocabulary models (e.g., CLIP, LSeg, OpenSeg) into 3D models. Two main paradigms are "2D-to-3D distillation" (projecting pixel-level features onto point clouds via geometric correspondences) and "point-text alignment" (aligning 3D features with text embeddings via contrastive learning).
2. **Limitations of Prior Work**: Both paradigms essentially train 3D models to replicate 2D feature representations, suffering from two core issues: (a) **suppression of intrinsic 3D geometric learning** — aligning 3D features to 2D representation spaces inhibits the learning of 3D geometric structures; (b) **inheritance of 2D prediction errors** — 2D models are prone to incorrect object masks due to occlusion and viewpoint changes (as shown in Figure 1), and 3D models inherit these errors, learning incorrect segmentation patterns.
3. **Key Challenge**: How to effectively preserve intrinsic 3D geometric information during 2D-to-3D knowledge distillation? Naively incorporating pretrained 3D features does not necessarily improve performance, as heterogeneous supervision signals from different modalities introduce training instability.
4. **Goal**: Three levels of geometric-semantic consistency are addressed: (a) **intra-superpoint consistency** — points within the same superpoint should share semantic labels, but 2D projection frequently leads to inconsistencies; (b) **intra-instance consistency** — single-view 2D predictions cover only partial instances, resulting in fragmented instance semantics in 3D space; (c) **inter-instance relation consistency** — multi-view feature aggregation introduces feature distribution drift among instances of the same category.
5. **Key Insight**: Pretrained 3D models (e.g., Sonata) have already learned strong geometric priors from large-scale point cloud data — objects of the same category exhibit similar geometric representations. The key challenge is how to exploit these priors during distillation to correct 2D-level errors.
6. **Core Idea**: By exploiting geometric priors from pretrained 3D models, GeoGuide guides the 2D-to-3D distillation process through three-level (superpoint, instance, inter-instance) geometric-semantic consistency modeling, ensuring that distilled 3D features preserve geometric structure while maintaining open-vocabulary semantic capability.

## Method

### Overall Architecture

Given a scene point cloud $\mathbf{P} \in \mathbb{R}^{N \times 3}$ and multi-view RGB images $\mathcal{I}$, GeoGuide extracts two types of features in parallel: (1) a frozen pretrained 3D backbone extracts geometric features $\mathbf{F}_{3d}^G \in \mathbb{R}^{N \times C_1}$; (2) a frozen 2D open-vocabulary segmentation model extracts pixel-level semantic features $\mathbf{F}_{2d}^M$. Camera parameters establish 2D-3D correspondences, projecting 2D features onto the point cloud to obtain $\mathbf{F}_{2d} \in \mathbb{R}^{N \times C}$. A lightweight MLP adapter maps 3D geometric features into the same semantic space to produce $\mathbf{F}_{3d}^{\text{sem}}$. Three hierarchical modules (USD, IMR, IIRC) guide the distillation process from local to global. At inference time, only the 3D point cloud is required, and all auxiliary modules are discarded.

### Key Designs

1. **Uncertainty-based Superpoint Distillation (USD)**:

    - **Function**: Leverages geometric consistency within superpoints to promote semantic coherence, and adaptively suppresses noisy 2D features.
    - **Mechanism**: Superpoints $\{Q_i\}_{i=1}^{N_Q}$ are first obtained via normal-based segmentation. Superpoint-level features are computed by mean-pooling 3D geometric and 2D semantic features within each superpoint. Differences between superpoint-level and point-level features are concatenated and passed through an MLP to predict per-point 2D feature reliability weights: $\mathcal{W} = \text{MLP}(\text{concat}[(\mathbf{S}_{3d}^G - \mathbf{F}_{3d}^G); (\mathbf{S}_{2d} - \mathbf{F}_{2d})])$. These weights are used for weighted pooling of 2D features within each superpoint to obtain refined superpoint semantic features $\overline{\mathbf{S}}_{2d}$. Cosine similarity distillation losses $\mathcal{L}_{sp}$ are computed at both superpoint and point levels.
    - **Design Motivation**: Conventional mean pooling is overly sensitive to prediction noise and amplifies bias. By incorporating 3D geometric information to estimate uncertainty weights, erroneous 2D predictions (e.g., incorrect projections from occluded or boundary-ambiguous regions) are suppressed while discriminative correct features are preserved.

2. **Instance-level Mask Reconstruction (IMR)**:

    - **Function**: Enforces semantic consistency at the instance level and recovers the complete mask for each instance.
    - **Mechanism**: Category-agnostic 3D instance segmentation yields instance masks $\{M_i\}_{i=1}^{N_M}$. A portion of each mask is randomly occluded to produce an incomplete mask $\overline{M}_i$. Features of the corresponding points are indexed from $\mathbf{F}_{3d}^{\text{sem}}$ and pooled; a linear layer then produces a mask feature $\overline{\mathbf{F}}_i^{\text{mask}}$. This feature is used to compute cosine similarity with the global $\mathbf{F}_{3d}^{\text{sem}}$ to predict the reconstructed mask: $\hat{M}_i = \text{sigmoid}(\cos(\overline{\mathbf{F}}_i^{\text{mask}}, \mathbf{F}_{3d}^{\text{sem}}))$. A BCE loss $\mathcal{L}_{\text{mask}}$ constrains the reconstruction to be consistent with the original mask.
    - **Design Motivation**: Superpoints typically cover only local regions of an instance. By reconstructing complete masks from partial ones, the model is encouraged to learn similar semantic features for all points within the same instance, achieving instance-level topological completeness.

3. **Inter-Instance Relation Consistency (IIRC)**:

    - **Function**: Aligns semantic relationships across instances with geometric affinity, mitigating viewpoint-induced semantic drift.
    - **Mechanism**: 3D geometric and semantic features within each instance mask are aggregated into mask-level embeddings $\mathbf{F}_{\text{mask}}^G$ and $\mathbf{F}_{\text{mask}}^{\text{sem}}$. Pairwise similarity matrices are computed: $\mathbf{P}_{\text{sim-m}}^G = \mathbf{F}_{\text{mask}}^G {\mathbf{F}_{\text{mask}}^G}^T$ and $\mathbf{P}_{\text{sim-m}}^{\text{sem}} = \mathbf{F}_{\text{mask}}^{\text{sem}} {\mathbf{F}_{\text{mask}}^{\text{sem}}}^T$. Analogous geometric and semantic similarity matrices are computed at the superpoint level. An MSE loss aligns semantic similarity with geometric similarity: $\mathcal{L}_{\text{sim}} = \text{MSE}(\mathbf{P}_{\text{sim-m}}^G, \mathbf{P}_{\text{sim-m}}^{\text{sem}}) + \text{MSE}(\mathbf{P}_{\text{sim-sp}}^G, \mathbf{P}_{\text{sim-sp}}^{\text{sem}})$.
    - **Design Motivation**: Pretrained 3D models ensure that objects of the same category have similar geometric representations, but this prior is not automatically preserved during 2D-to-3D distillation. By enforcing alignment between semantic and geometric similarity matrices, the degradation of inter-instance geometric consistency during distillation is prevented.

### Loss & Training

Total loss: $\mathcal{L}_{\text{final}} = \lambda_1 \mathcal{L}_{\text{sp}} + \lambda_2 \mathcal{L}_{\text{mask}} + \lambda_3 \mathcal{L}_{\text{sim}}$

At inference, only 3D point cloud input is required. The adapter output $\mathbf{F}_{3d}^{\text{sem}}$ is compared against CLIP text embeddings via cosine similarity for classification. All three auxiliary modules are discarded, introducing no additional inference overhead.

## Key Experimental Results

### Main Results

**Open-vocabulary 3D semantic segmentation (mIoU / mAcc):**

| Method | ScanNet v2 mIoU | nuScenes mIoU | Matterport3D mIoU |
|--------|----------------|--------------|-------------------|
| OpenScene (LS) | 54.2 | 36.7 | 43.4 |
| SAS (stage1) | 59.2 | 45.4 | 46.3 |
| SAS (stage2) | 61.9 | 47.5 | 48.6 |
| **GeoGuide (SAS*)** | **64.8** | **50.3** | **51.9** |

GeoGuide surpasses SAS (stage1) by +5.6 mIoU on ScanNet v2 and by +4.9 mIoU on nuScenes.

**Long-tail evaluation (Matterport3D):**

| Method | K=21 mIoU | K=40 mIoU | K=80 mIoU | K=160 mIoU |
|--------|----------|----------|----------|-----------|
| OpenScene (OS) | 41.1 | 33.4 | 18.1 | 8.9 |
| DMA (OS) | 45.1 | 37.9 | 19.7 | 9.4 |
| **GeoGuide (OS)** | **47.7** | **38.5** | **22.0** | **11.6** |

### Ablation Study

| Module | ScanNet v2 mIoU | mAcc | Note |
|--------|----------------|------|------|
| Full model | 53.4 | 74.8 | Using OpenSeg features |
| w/o USD | decreased | decreased | Loss of intra-superpoint consistency |
| w/o IMR | decreased | decreased | Loss of instance-level consistency |
| w/o IIRC | decreased | decreased | Absence of inter-instance relational constraint |

**Preliminary experiments (validating framework design motivation):**

| Method | mIoU (OpenSeg) | mAcc (OpenSeg) |
|--------|---------------|---------------|
| OpenScene (train entire 3D network) | 47.5 | 70.7 |
| Frozen 3D backbone + MLP adapter | 50.4 | 75.2 |

Freezing the pretrained 3D backbone and using a lightweight adapter alone yields +2.9 mIoU, though the gain is inconsistent across different 2D models (a slight drop is observed with LSeg), indicating that naive distillation can corrupt geometric priors learned by 3D pretraining.

### Key Findings

- **Complementarity of the three modules**: USD addresses local superpoint consistency, IMR addresses intra-instance completeness, and IIRC addresses inter-instance relations, resolving distillation bias from micro to macro levels respectively.
- **Robust generalization across 2D models**: GeoGuide consistently improves performance regardless of whether LSeg, OpenSeg, or SAS features are used, demonstrating that the method addresses fundamental geometric inconsistency rather than relying on specific 2D features.
- **Concurrent improvement in mIoU and mAcc**: This indicates that the method not only improves segmentation coverage but also enhances per-class prediction accuracy, with geometric consistency modeling helping the network learn more discriminative and category-specific features.
- GeoGuide exhibits the least performance degradation in long-tail scenarios, attributable to the IIRC module maintaining semantic consistency among same-category instances.

## Highlights & Insights

- **Core insight — "preserving geometric priors"**: Rather than training the 3D model to fully replicate 2D features, GeoGuide freezes the pretrained 3D backbone and trains only a lightweight adapter, while actively protecting and exploiting geometric priors throughout distillation via three modules. This "conservative distillation" strategy is worth referencing in other cross-modal transfer scenarios.
- **Hierarchical consistency modeling**: The three-level consistency design — superpoint (local) → instance (regional) → inter-instance (global) — forms a complete local-to-global semantic alignment mechanism, with each level having a clear motivation and complementary role.
- **Zero additional inference overhead**: All three auxiliary modules are used only during training and completely discarded at inference, making GeoGuide identical in inference efficiency to baseline methods such as OpenScene, which is highly practical.
- The uncertainty-guided weighted aggregation idea is broadly applicable to any scenario requiring fusion of multiple potentially noisy feature sources.

## Limitations & Future Work

- **Dependency on class-agnostic instance segmentation quality**: The effectiveness of the IMR and IIRC modules is constrained by the accuracy of 3D instance segmentation methods such as Mask3D.
- **No self-distillation strategy**: Methods such as SAS can further improve performance through costly self-distillation, which GeoGuide does not adopt, yet still surpasses SAS.
- **Over- and under-segmentation of superpoints and instances**: Superpoint quality directly affects the USD module, and improper grouping may occur in geometrically ambiguous regions.
- **Cross-domain generalization**: Although cross-domain experiments from ScanNet to Matterport3D are conducted, large-scale indoor-to-outdoor transfer remains challenging.
- Stronger 3D pretrained models (e.g., PointMAE v2) could be explored to obtain better geometric priors.

## Related Work & Insights

- **vs. OpenScene**: OpenScene trains the entire 3D network to align with multi-view 2D features, which corrupts geometric priors. GeoGuide freezes the 3D backbone and leverages geometric priors to guide distillation, surpassing OpenScene by +5.9 mIoU on ScanNet v2 with OpenSeg features.
- **vs. SAS**: SAS integrates multiple 2D open-vocabulary models to reduce single-model bias but still neglects 3D geometric structure. Using the same 2D features, GeoGuide further improves by +5.6 mIoU on ScanNet v2 without requiring costly self-distillation.
- **vs. GGSD**: GGSD adopts a mean-teacher framework to enhance distillation but still lacks explicit geometric constraints. GeoGuide achieves more comprehensive improvements through hierarchical geometric-semantic consistency.
- This work suggests that in cross-modal distillation, the pretrained priors of the target modality should not be discarded but rather used as a "teacher" to guide the distillation process.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-level geometric consistency modeling is well-motivated, though the technical novelty of individual modules is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three major benchmarks, long-tail and cross-domain evaluation, complete ablation, and preliminary experiments that sufficiently validate the motivation.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly derived, module design logic is coherent, and figures and tables are informative.
- **Value**: ⭐⭐⭐⭐ Introducing hierarchical geometric constraints into open-vocabulary 3D segmentation is a meaningful direction; surpassing SOTA without self-distillation is practically significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Universal 3D Shape Matching via Coarse-to-Fine Language Guidance](universal_3d_shape_matching_via_coarse-to-fine_language_guidance.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation](pca-seg_revisiting_cost_aggregation_for_openvocabulary_semantic_and_part_segmentat.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)

</div>

<!-- RELATED:END -->
