---
title: >-
  [Paper Note] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations
description: >-
  [NeurIPS 2025][3D Vision][open-vocabulary] This paper proposes OpenLex3D, a tiered evaluation benchmark for open-vocabulary 3D scene representations. Built upon Replica, ScanNet++, and HM3D, it provides language annotations 13× richer than the original labels, supporting evaluation on two tasks: open-set 3D semantic segmentation and object retrieval.
tags:
  - NeurIPS 2025
  - 3D Vision
  - open-vocabulary
  - 3D scene understanding
  - benchmark
  - semantic segmentation
  - object retrieval
date: 2026-05-08
content_hash: 8e0933eabb9de39b
---

# OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations

**Conference**: NeurIPS 2025
**arXiv**: [2503.19764](https://arxiv.org/abs/2503.19764)
**Code**: [Project Page](https://openlex3d.github.io/)
**Area**: 3D Vision / Open-Vocabulary Understanding
**Keywords**: open-vocabulary, 3D scene understanding, benchmark, semantic segmentation, object retrieval

## TL;DR
This paper proposes OpenLex3D, a tiered evaluation benchmark for open-vocabulary 3D scene representations. Built upon Replica, ScanNet++, and HM3D, it provides language annotations 13× richer than the original labels, supporting evaluation on two tasks: open-set 3D semantic segmentation and object retrieval.

## Background & Motivation
**State of the Field**: Open-vocabulary language models have greatly advanced 3D scene understanding, enabling natural language interaction with 3D environments. Existing methods such as LERF, OpenScene, and ConceptFusion have demonstrated impressive demonstrations.

**Limitations of Prior Work**: Existing evaluations still rely on closed-set semantic annotations (e.g., 20 categories in ScanNet, 88 in Replica), which fail to capture the richness and ambiguity of real-world natural language queries.

**Root Cause**: Methods claim to be "open-vocabulary" yet are evaluated under closed-set protocols, leading to an overestimation of their true capabilities. For instance, a method may correctly respond to the query "chair" but fail on "office swivel chair" or "armchair."

**Starting Point**: Constructing an annotation dataset that genuinely captures linguistic diversity by introducing synonym categories and fine-grained descriptions.

**Core Idea**: Re-evaluate open-vocabulary 3D methods using annotations 13× richer in linguistic variety, thereby exposing the true limitations of existing approaches.

## Method

### Overall Architecture
OpenLex3D organizes its evaluation into three tiers:

1. **Tier 1: Standard Semantic Segmentation** — Uses original dataset categories as a baseline.
2. **Tier 2: Synonym-Expanded Segmentation** — Augments each category with synonyms (e.g., chair → office chair / swivel chair / armchair), testing robustness to linguistic variation.
3. **Tier 3: Free-Form Retrieval** — Users provide arbitrary natural language descriptions (e.g., "the red cushion near the window"), testing fine-grained retrieval capability.

### Key Designs

1. **Annotation Pipeline**

    - Annotations are performed at the mesh face level, not via 2D projection.
    - Each object is associated with a canonical name, multiple synonyms, appearance descriptions, and positional descriptions.
    - Annotator guidelines are provided to ensure consistency.
    - The final per-scene label count is on average **13× that of the original dataset**.

2. **Evaluation Tasks**

    - **Open-Set 3D Semantic Segmentation**: Given a text query, predict semantic labels on 3D point clouds or meshes.
    - **Object Retrieval**: Given a natural language description, localize the target object in the 3D scene.

3. **Evaluation Metrics**

    - mIoU (standard for semantic segmentation), reported separately for Tier 1/2/3.
    - Recall@K (for object retrieval), with $K = 1, 3, 5$.
    - Feature Precision: measures semantic consistency in feature space.

### Data Scale

| Dataset | # Scenes | Original Categories | OpenLex3D Labels |
|--------|--------|-----------|------------------|
| Replica | 8 | 88 | ~1150 |
| ScanNet++ | 10 | 100 | ~1300 |
| HM3D | 10 | 30 | ~390 |

## Key Experimental Results

### Main Results — 3D Semantic Segmentation mIoU (%)

| Method | Replica T1 | Replica T2 | Replica T3 | ScanNet++ T1 | ScanNet++ T2 |
|------|-----------|-----------|-----------|-------------|-------------|
| LERF | 41.2 | 28.7 | 15.3 | 38.5 | 24.1 |
| OpenScene | 45.8 | 31.2 | 18.6 | 42.3 | 27.8 |
| ConceptFusion | 43.5 | 29.8 | 16.9 | 40.1 | 25.6 |
| LangSplat | 48.1 | 33.5 | 20.2 | 44.7 | 29.3 |
| OpenMask3D | 50.3 | 35.1 | 22.4 | 46.9 | 31.7 |

### Object Retrieval Recall@1 (%)

| Method | Replica Standard | Replica Synonyms | Replica Free-Form |
|------|-------------|---------------|-----------------|
| LERF | 52.3 | 38.1 | 21.5 |
| OpenScene | 58.7 | 42.5 | 25.8 |
| ConceptFusion | 55.1 | 40.2 | 23.1 |
| LangSplat | 61.2 | 45.8 | 28.3 |
| OpenMask3D | 64.5 | 48.2 | 30.7 |

### Ablation Study — Impact of Tier Level on Performance (Average mIoU Drop)

| Method | T1→T2 Drop | T1→T3 Drop |
|------|-----------|-----------|
| LERF | -30.3% | -62.9% |
| OpenScene | -31.9% | -59.4% |
| ConceptFusion | -31.5% | -61.1% |
| LangSplat | -30.4% | -58.0% |
| OpenMask3D | -30.2% | -55.5% |

### Key Findings
- **All methods exhibit a 55–63% mIoU drop from T1 to T3**, revealing that existing approaches are highly vulnerable to linguistic variation.
- Synonym queries (T2) alone cause approximately 30% performance degradation, indicating overfitting to closed-set label names.
- OpenMask3D achieves the best overall performance, yet substantial room for improvement remains (T3 mIoU ~22%).
- Segmentation methods generally perform poorly on Feature Precision, suggesting that semantically similar queries are not mapped to nearby regions in feature space.
- 3D Gaussian Splatting-based methods (LangSplat) outperform NeRF-based methods on fine-grained descriptions.

## Highlights & Insights
- **High diagnostic value**: Exposes the "false prosperity" of open-vocabulary 3D methods under closed-set evaluation.
- **Practical tiered design**: T1/T2/T3 progressively increase difficulty, facilitating identification of specific weaknesses in each method.
- **High annotation quality**: Mesh face-level annotations with multi-annotator review.
- **Community infrastructure**: Public dataset, evaluation code, and leaderboard are released.

## Limitations & Future Work
- The number of scenes is relatively limited (28 scenes), potentially insufficient to cover all indoor/outdoor variations.
- Annotations primarily target indoor scenes; outdoor and large-scale scenes are not covered.
- Free-form descriptions (T3) involve considerable annotator subjectivity, which may introduce scoring bias.
- Dynamic scenes and temporal queries are not included.

## Related Work & Insights
- **ScanRefer / ReferIt3D**: Pioneering works in 3D referring expression localization.
- **LERF (ICCV 2023)**: CLIP feature fields.
- **OpenScene (CVPR 2023)**: Open-vocabulary 3D segmentation.
- Insight: Open-vocabulary evaluation needs to move beyond "category names" toward natural language descriptions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First genuinely open-vocabulary 3D evaluation benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple methods, datasets, and tiers.
- Writing Quality: ⭐⭐⭐⭐ Well-motivated with in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed evaluation infrastructure for the community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[NeurIPS 2025\] From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes](from_objects_to_anywhere_a_holistic_benchmark_for_multi-level_visual_grounding_i.md)

<!-- RELATED:END -->
