---
title: >-
  [Paper Note] ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes
description: >-
  [CVPR 2026][Autonomous Driving][4D Semantic Instance Segmentation] This paper formally defines the temporally sparse 4D indoor semantic instance segmentation (4DSIS) task and proposes ReScene4D…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "4D Semantic Instance Segmentation"
  - "Temporal Consistency"
  - "Indoor Scene Change"
  - "Contrastive Learning"
  - "Spatio-Temporal Queries"
date: 2026-05-08
content_hash: ca514c9a72b12167
---

# ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes

**Conference**: CVPR 2026
**arXiv**: [2601.11508](https://arxiv.org/abs/2601.11508)  
**Code**: [Project Page](https://www.easteine.com/rescene4d/)  
**Area**: Autonomous Driving / 3D Vision / Instance Segmentation
**Keywords**: 4D Semantic Instance Segmentation, Temporal Consistency, Indoor Scene Change, Contrastive Learning, Spatio-Temporal Queries

## TL;DR

This paper formally defines the temporally sparse 4D indoor semantic instance segmentation (4DSIS) task and proposes ReScene4D, which extends a 3D instance segmentation architecture to the 4D domain via three temporal information sharing strategies—spatio-temporal contrastive loss, spatio-temporal mask pooling, and spatio-temporal decoder serialization. The method achieves state-of-the-art performance on the 3RScan dataset and introduces a new t-mAP metric that jointly evaluates segmentation quality and temporal identity consistency.

## Background & Motivation

1. **Background**: 3D semantic instance segmentation (3DSIS) has achieved strong performance on static scenes, with representative query-based Transformer methods including Mask3D, SPFormer, and Relation3D. Meanwhile, 4D LiDAR panoptic segmentation approaches (Mask4D, Mask4Former) have advanced on densely sampled autonomous driving sequences. Large-scale self-supervised point cloud encoders (Sonata, Concerto) have further pushed state-of-the-art results across multiple 3D tasks.

2. **Limitations of Prior Work**: (a) 3DSIS methods process each observation independently, ignoring temporal identity continuity—the same chair observed in two scans is split into two separate instances. (b) 4D LiDAR methods rely on high-frequency dense sampling and small inter-frame motion assumptions (optical flow tracking, motion models), which break down in indoor settings where observations may be separated by days, months, or years, with substantial changes in object positions, appearances, or topology. (c) Change detection methods identify differences but do not establish semantic or instance-level correspondences. (d) No existing metric jointly evaluates segmentation quality and temporal identity consistency.

3. **Key Challenge**: 4D understanding of indoor environments faces a unique challenge—observations are temporally sparse (intervals ranging from days to years), yet scene changes can be substantial (objects moved, added, or removed), rendering methods that rely on dense observations or motion models entirely ineffective. Maintaining instance identity consistency across temporally distant scans without dense observations remains an open problem.

4. **Goal**: (1) Formally define the temporally sparse 4DSIS task; (2) design temporal information sharing strategies that do not require dense observations; (3) propose a metric that jointly evaluates segmentation accuracy and temporal consistency.

5. **Key Insight**: Sharing information across temporal observations—even when the scene has changed—benefits not only 4DSIS but also single-stage 3DSIS. Geometric and semantic priors can be leveraged flexibly rather than requiring hard geometric alignment.

6. **Core Idea**: Jointly predict instance masks across all temporal stages via spatio-temporal queries, and achieve temporally consistent instance segmentation without dense observations through three progressive temporal information sharing strategies (contrastive loss, spatio-temporal masking, and spatio-temporal serialization).

## Method

### Overall Architecture

ReScene4D extends the mask Transformer architecture of Mask3D to 4D. The input is a sequence of $T$ 3D scans of the same scene, registered to a unified coordinate system while retaining temporal dimension labels (4D voxelization). A feature backbone independently extracts hierarchical features for each temporal stage. Spatio-temporal queries iteratively refine themselves via joint masked attention across all temporal stages, and a mask module predicts temporally consistent instance masks and semantic labels across the sequence. Three temporal information sharing modules (contrastive loss, ST masking, ST serialization) operate at different levels of the architecture to promote cross-temporal consistency.

### Key Designs

1. **Spatio-Temporal 4D Input and Architecture Adaptation**:

    - **Function**: Represents multiple scans as a unified 4D point cloud, enabling end-to-end joint prediction.
    - **Mechanism**: Registers sequential point clouds to a global coordinate system to form $\mathcal{P} \in \mathbb{R}^{N \times 4}$ (x, y, z, t). A key distinction from LiDAR methods is that the temporal dimension is **not** collapsed into 3D; points from different temporal observations remain independent during voxelization. Spatio-temporally shared queries jointly predict instance masks across all stages without a separate matching step. Positional encodings use 4D Fourier features. Unmatched predictions are penalized by an elevated no-object semantic loss ($\lambda_{noobj}=0.2$) to suppress cross-temporal duplicate predictions.
    - **Design Motivation**: LiDAR methods collapse multiple frames into 3D, losing temporal discriminability and forcing points assumed to be spatially aligned to share instance labels—an assumption that fails under sparse indoor observations.

2. **Cross-Temporal Contrastive Loss**:

    - **Function**: Enhances instance discriminability and propagates temporal information at the feature level.
    - **Mechanism**: Supervised contrastive learning is applied on pooled superpoint features. Instance annotations are used to construct a binary relation matrix $R_{GT} \in \{0,1\}^{S \times S}$ (superpoints of the same instance form positive pairs; different instances form negative pairs) sampled across the entire temporal sequence. An InfoNCE loss $\mathcal{L}_{cont} = -\frac{1}{|S^+|}\sum_{i \in S^+}\log\frac{\sum_{j \in P(i)}\exp(L_{ij})}{\sum_k \exp(L_{ik})}$ encourages the network to learn temporally consistent feature representations—features of the same instance at different time stages should be similar, while features of different instances should be well-separated.
    - **Design Motivation**: This is the lightest-weight temporal sharing strategy, injecting cross-temporal identity consistency signals into feature learning at training time solely through the loss function, without modifying the network architecture.

3. **Spatio-Temporal Mask Pooling**:

    - **Function**: Enables queries from different temporal stages to guide each other to attend to the same spatio-temporal positions.
    - **Mechanism**: During query refinement in masked attention, auxiliary masks from different stages are temporally pooled via logical OR, allowing queries to attend to temporally aligned voxel positions. At coarser resolution levels, the probability of voxel overlap is high, promoting temporal information sharing; at finer resolution levels, overlap decreases and masks naturally revert to independent refinement. Spatially aligned points are not forced to share labels in the final mask.
    - **Design Motivation**: Geometric alignment priors are exploited flexibly rather than strictly enforced—when voxel overlap exists, it guides attention; when it does not, there is no adverse effect. This adapts naturally to indoor scenes where most objects are static while some move.

4. **Spatio-Temporal Decoder Serialization**:

    - **Function**: Enables the decoder to attend to both spatial and temporal neighbors simultaneously, enriching contextual information.
    - **Mechanism**: For the PTv3 backbone (Sonata/Concerto), point clouds from all temporal stages are merged in the decoder, and four space-filling curves (Z-order, Hilbert, etc.) are applied to generate serialization patterns, which are randomly mixed with the original spatial serialization. The encoder retains the fixed spatial serialization consistent with pretraining (with frozen parameters) to avoid domain shift.
    - **Design Motivation**: PTv3's serialization-based attention mechanism inherently supports expanding the effective receptive field by modifying the serialization order. Extending serialization from purely spatial to spatio-temporal allows the decoder to leverage complementary information from other time steps during feature refinement.

### Loss & Training

- The primary loss follows Mask3D's mask prediction objective (semantic classification and binary mask losses).
- A cross-temporal contrastive loss $\mathcal{L}_{cont}$ is added.
- Unmatched queries are penalized with a higher no-object weight.
- Training data is mixed: 3RScan two-stage sequences and ScanNet single scans at a ratio of 1.0:0.8.
- For the PTv3 backbone, the encoder is frozen (utilizing pretrained weights) while the decoder is trained from scratch.

## Key Experimental Results

### Main Results

**4DSIS Evaluation (3RScan dataset)**:

| Method | t-mAP | t-mAP50 | t-mAP25 | mAP | mAP50 | mAP25 |
|--------|-------|---------|---------|-----|-------|-------|
| Mask4D | 1.3 | 2.9 | 8.7 | 2.1 | 5.5 | 21.2 |
| Mask4Former | 17.0 | 38.9 | 59.1 | 21.7 | 45.6 | 66.3 |
| Mask3D + Semantic Matching | 20.1 | 32.9 | 38.6 | 25.9 | 42.3 | 73.9 |
| Mask3D + Geometric Matching | 20.7 | 43.1 | 62.4 | 29.7 | 54.1 | 70.9 |
| ReScene4D (Mink.) | 31.6 | 49.5 | 61.6 | 39.2 | 60.7 | 74.1 |
| ReScene4D (Sonata) | 33.2 | 50.7 | 63.3 | 40.9 | 62.8 | 79.1 |
| **ReScene4D (Concerto)** | **34.8** | **52.5** | **66.8** | **43.3** | **64.3** | **81.9** |

**Single-Stage 3DSIS Performance (4D predictions evaluated per stage independently)**:

| Method | Stage | mAP | mAP50 | mAP25 |
|--------|-------|-----|-------|-------|
| Mask3D (3D only) | - | 46.4 | 68.5 | 78.5 |
| Mask3D + Geometric Matching | 2 | 21.9 | 46.4 | 68.4 |
| **ReScene4D (C)** | 1 | **47.8** | 68.4 | **82.0** |
| **ReScene4D (C)** | 2 | **48.3** | **69.8** | **83.0** |

### Ablation Study

**Ablation of Temporal Information Sharing Strategies (Concerto backbone)**:

| Contrastive Loss | ST-Serialization | ST-Masking | t-mAP | t-mREC | Ambiguous | Rigid Change | Non-Rigid Change |
|-----------------|-----------------|-----------|-------|--------|-----------|-------------|-----------------|
| × | × | × | 28.4 | 41.8 | 20.4 | 44.9 | 62.1 |
| ✓ | × | × | 34.1 | 49.6 | 42.8 | 48.4 | 63.2 |
| × | ✓ | × | 32.9 | 48.8 | 43.2 | 40.9 | 67.0 |
| × | × | ✓ | 32.4 | 48.5 | 42.3 | 40.2 | 70.7 |
| ✓ | ✓ | × | **34.8** | 52.1 | 47.2 | 48.6 | 66.5 |

### Key Findings

- **LiDAR methods degrade severely on sparse indoor 4D scenes**: Mask4D achieves only 1.3 t-mAP, as its LiDAR-specific backbone trained from scratch performs poorly on the limited 3RScan data. Mask4Former, which relies on dense observations and smooth motion assumptions, also underperforms.
- **Backbone choice determines the optimal temporal strategy**: The best strategy for the Concerto backbone is contrastive loss + ST serialization; for Sonata it is ST serialization + ST masking; Minkowski benefits most from contrastive loss. Differences in feature representations and latent spaces across backbones lead to different optimal temporal strategies.
- **4D joint inference in turn improves 3D performance**: ReScene4D's single-stage mAP (47.8/48.3) surpasses dedicated Mask3D training (46.4), indicating that temporal information sharing functions as a form of data and observation augmentation.
- **Different change types favor different strategies**: Contrastive loss is most effective for ambiguous instances and rigid changes (distinguishing visually similar instances via negative pairs), while ST masking is most effective for non-rigid changes (assisting objects with large local geometric changes but small displacement via spatial alignment).

## Highlights & Insights

- **The t-mAP metric design** is carefully crafted—it uses min-IoU across temporal stages, ensuring that identity inconsistency at any stage is penalized; an iterative assignment strategy handles ambiguous instance groups (e.g., a set of visually identical chairs that swap positions should not be counted as errors). This metric can be directly adopted by the broader indoor 4D understanding community.
- **The "no collapsing, no hard alignment, no explicit matching" design philosophy**: Unlike LiDAR methods that collapse multiple frames into 3D, ReScene4D maintains 4D independence while sharing information through flexible strategies, achieving far greater robustness to sparse observations and large scene changes than hard-alignment approaches.
- **Mixing ScanNet single-scan data during training** is a practical technique—since the model does not require an explicit temporal bottleneck, it can process both single-stage and multi-stage inputs simultaneously, leveraging the larger ScanNet dataset to improve semantic coverage.

## Limitations & Future Work

- Constrained by the scale and annotation quality of the 3RScan dataset: only 17% of validation instances undergo changes, and temporal annotations are inconsistent (primarily focused on foreground objects), limiting thorough validation of temporal strategies.
- Currently supports only sequences of length $T=2$; scalability to longer sequences ($T>2$) remains unverified.
- The PTv3 encoder is used frozen without end-to-end fine-tuning due to computational constraints; the authors note that fine-tuning may yield further improvements.
- Segmentation of small objects (e.g., pillows) remains difficult—the non-systematic annotation of small objects in 3RScan causes the model to underperform on them.
- Larger-scale and more diverse 4D indoor scene datasets are needed to advance this research direction.

## Related Work & Insights

- **vs. Mask4D**: Mask4D propagates instance queries across consecutive scans but lacks explicit temporal information sharing—later scans cannot correct earlier predictions or align instance features within a shared context. ReScene4D's joint spatio-temporal query refinement and information sharing strategies directly address this limitation.
- **vs. Mask4Former / SP2Mask4D**: These methods operate on stacked scans to enforce spatial alignment—suitable for the minimal inter-frame changes in LiDAR settings but ill-suited for the substantial changes in indoor scenes. ReScene4D does not assume alignment but instead flexibly exploits geometric priors.
- **vs. RescanNet**: Inductively associates instances step by step, relying on ground-truth segmentation initialization and handcrafted segmentation/registration/matching pipelines. ReScene4D jointly predicts consistent instance masks across all temporal stages end-to-end.
- **vs. MORE**: Jointly reconstructs and relocates objects but relies on ground-truth mask filtering and heuristic matching. ReScene4D requires none of these intermediate steps.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Unifies a new task definition, a new metric, and a new method, opening up the indoor 4DSIS direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablation across three backbones × three strategies with thorough baseline comparisons, though limited by reliance on a single dataset (3RScan only).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear and rigorous; the design rationale for the t-mAP metric (including a toy example) is well-argued.
- **Value**: ⭐⭐⭐⭐ — Provides a systematic task definition and benchmark method for long-term dynamic understanding of indoor scenes, with direct applicability to digital twins, facility management, and related domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2026\] CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation](cyclebev_regularizing_view_transformation_networks_via_view_cycle_consistency_fo.md)
- [\[CVPR 2026\] SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors](saber_spatially_consistent_3d_universal_adversarial_objects_for_bev_detectors.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)

</div>

<!-- RELATED:END -->
