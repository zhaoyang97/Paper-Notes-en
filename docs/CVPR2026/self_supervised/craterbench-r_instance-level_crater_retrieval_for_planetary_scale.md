---
title: >-
  [Paper Note] CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale
description: >-
  [CVPR 2026][Self-Supervised Learning][crater retrieval] This work is the first to formalize crater analysis as an instance-level image retrieval problem. It introduces the CraterBench-R benchmark (~25K Mars crater IDs, 50K gallery, 5K queries), and through systematic diagnosis reveals that single-vector pooling imposes an accuracy ceiling while supervised metric learning consistently degrades performance. A training-free instance token aggregation method is proposed—selecting K seed tokens via top-K attention or FPS and performing cosine nearest-neighbor residual assignment—to compress 196 ViT patch tokens into K representative tokens for late interaction matching. At K=64, the method matches full-token accuracy with substantially reduced storage. A practical two-stage pipeline (single-vector coarse retrieval + instance token re-ranking) recovers 89–94% of full-pipeline accuracy.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - crater retrieval
  - instance-level retrieval
  - ViT patch token
  - training-free token aggregation
  - two-stage retrieval
date: 2026-05-08
content_hash: b310e143340c8074
---

# CraterBench-R: Instance-Level Crater Retrieval for Planetary Scale

**Conference**: CVPR 2026
**arXiv**: [2604.06245](https://arxiv.org/abs/2604.06245)
**Code**: [https://hf.co/datasets/jfang/CraterBench-R](https://hf.co/datasets/jfang/CraterBench-R)
**Area**: Planetary Science / Image Retrieval
**Keywords**: crater retrieval, instance-level retrieval, ViT patch token, training-free token aggregation, two-stage retrieval

## TL;DR
This work is the first to formalize crater analysis as an instance-level image retrieval problem. It introduces the CraterBench-R benchmark (~25K Mars crater IDs, 50K gallery, 5K queries), and through systematic diagnosis reveals that single-vector pooling imposes an accuracy ceiling while supervised metric learning consistently degrades performance. A training-free instance token aggregation method is proposed—selecting K seed tokens via top-K attention or FPS and performing cosine nearest-neighbor residual assignment—to compress 196 ViT patch tokens into K representative tokens for late interaction matching. At K=64, the method matches full-token accuracy with substantially reduced storage. A practical two-stage pipeline (single-vector coarse retrieval + instance token re-ranking) recovers 89–94% of full-pipeline accuracy.

## Background & Motivation

**State of the Field**: Mars orbital imagery contains millions of crater structures. Deep learning efforts have focused on detection—predicting locations and diameters—without providing visual representations suitable for association.

**Practical Need**: Scientific workflows depend on **association**—deduplication of the same crater across images, cross-observation matching, and morphological analogy discovery. These tasks are fundamentally **retrieval** problems, not detection problems.

**Core Challenge**: Martian crater appearance is highly complex—varying degradation states (pristine vs. heavily eroded), diverse infill mechanisms (dunes/dust/lava), and dramatic illumination variation across orbital passes—resulting in extreme structural and photometric variability.

**Representation Bottleneck Findings**: (1) Single-vector global descriptors (CLS/GeM pooling) over-compress spatial detail, imposing a hard accuracy ceiling. (2) Supervised metric learning (three commonly used losses) consistently degrades retrieval accuracy, including late interaction accuracy—attributed to only 2 views per ID, yielding insufficient positive diversity. (3) Retaining all 196 patch tokens for late interaction achieves high accuracy but is infeasible at planetary scale due to storage and computation costs.

**Core Idea**: Training-free instance token aggregation—post-hoc compression from frozen ViT features—avoids fine-tuning degradation while preserving spatial detail.

## Method

### Key Designs

1. **CraterBench-R Benchmark**:

    - ~25K crater IDs, each with 2 gallery views (~50K gallery images)
    - 5K manually verified query images (1,000 crater IDs × 5 views) with cross-scale and context variation
    - Mars CTX imagery with a complete evaluation protocol
    - Diameter range: 1.0–401 km (median 1.5 km; 69% below 2 km)
    - Gallery provided in two standard crops: 2× and 3× diameter context, explicitly evaluating robustness to context variation
    - Queries manually verified to exclude degraded samples (pure background, severe artifacts, etc.)
    - Evaluation metrics: Recall@K (K=1,5,10) and mAP; cluster-tolerant relevance to handle co-visible crater cases

2. **Baseline Diagnosis (30 frozen backbones)**:

    - Self-supervised ViTs—especially domain-specific pretrained MarsDINO—perform best, outperforming general-purpose models with 79× more parameters
    - ViT-B/16 MarsDINO (85M parameters): R@1=.374, mAP=.553—best single-vector result
    - Same architecture with DINO: R@1=.304 → domain-specific pretraining yields +7.0 R@1 gain
    - MAE (.022) and CLIP (.058) perform extremely poorly under the same ViT-B/16 architecture → pretraining objective matters more than architecture
    - Single-vector pooling (CLS/GeM) constitutes an insurmountable accuracy ceiling
    - Supervised metric learning (Triplet/ArcFace/SupCon): all three losses **consistently degrade** retrieval accuracy
        - Triplet performs best yet still reduces CLS mAP from .368 to .318 and LI mAP from .602 to .530
        - Root cause: only 2 views per ID → insufficient positive diversity → full-backbone fine-tuning corrupts the token-level structure required by late interaction

3. **Instance Token Aggregation (Training-Free; Core Contribution)**:

    - **Step 1 — Seed Selection**: Select K seed indices $\mathcal{S}=\{s_1,\ldots,s_K\}$ via attention-based selection (top-K by CLS→patch attention weights) or FPS (farthest point sampling in cosine space)
    - **Step 2 — Assignment**: Non-seed tokens are assigned to their nearest seed by cosine similarity, forming clusters $C_k$
    - **Step 3 — Aggregation**: Seed and cluster tokens are merged in residual form:
    $\mathbf{z}_k = \ell_2\left(\mathbf{t}_{s_k} + \frac{1}{\max(|C_k|, \epsilon)}\sum_{i \in C_k} \mathbf{t}_i\right)$
    - **Why residual rather than centroid**: The residual formulation preserves the seed's identity, maintaining discriminability even for small clusters; k-means centroids blur local morphological detail
    - Output: K instance tokens used for ColBERT-style late interaction matching:
    $s_{\mathrm{LI}}(q,g) = \frac{1}{K_q}\sum_{i=1}^{K_q}\max_{1 \leq j \leq K_g} \langle \mathbf{t}_i^q, \mathbf{t}_j^g \rangle$
    - Training-free → avoids the fine-tuning degradation trap
    - At K=16: mAP is +17.9 pts above raw token selection; at K=64: ≈ full 196-token accuracy with 3× storage reduction

4. **Two-Stage Planetary-Scale Retrieval Pipeline**:

    - Stage 1: Single-vector FAISS coarse retrieval of top-S candidates (millisecond-level)
    - Stage 2: Instance token late interaction re-ranking
    - Offline aggregation complexity: $O(NK)$ per image; online matching complexity: $O(K^2D)$ per candidate
    - At S=100: recovers 89–94% of full-pipeline accuracy
    - At S=500: recovers ~96%

## Key Experimental Results

### Main Results — Frozen Backbone Single-Vector Retrieval

| Model | Params | Pooling | R@1 | R@5 | mAP |
|-------|--------|---------|-----|-----|-----|
| EfficientNet-B0 | 4M | GAP | .150 | .214 | .248 |
| ResNet-50 | 24M | GeM | .142 | .217 | .244 |
| ViT-S/16 DINO | 22M | CLS | .273 | .360 | .420 |
| ViT-B/8 DINO | 86M | GeM | .304 | .379 | .461 |
| ViT-B/14 DINOv2 | 87M | Max | .240 | .323 | .377 |
| ViT-7B/16 DINOv3_sat | 6.7B | Max | .330 | .416 | .505 |
| ViT-B/16 MAE | 86M | GeM | .022 | .042 | .043 |
| ViT-B/16 CLIP | 86M | GeM | .058 | .091 | .107 |
| ViT-S/16 MarsDINO | 22M | GeM | .269 | .356 | .412 |
| **ViT-B/16 MarsDINO** | **85M** | **CLS** | **.374** | **.472** | **.553** |

### Ablation Study — Instance Token Aggregation

| Configuration | mAP | Notes |
|---------------|-----|-------|
| Single-vector (best backbone) | .553 | MarsDINO CLS pooling ceiling |
| Raw attention selection K=16 | .444 | Token selection only, no aggregation |
| **Instance token aggregation K=16** | **.623** | **+17.9 pts, substantial gain** |
| Raw attention selection K=64 | .716 | Accuracy increases with more tokens |
| **Instance token aggregation K=64** | **.760** | **Approaches full-token accuracy** |
| Full 196-token late interaction | .744 (MarsDINO) | Complete upper bound |
| Supervised Triplet fine-tuning | .318 (CLS) | **Degraded**, below frozen .368 |

### Key Findings
- **"Fine-tuning degradation"** is the most important negative result in this paper—in the few-view regime (only 2 views per ID), brute-force learning underperforms frozen features with post-hoc processing
- Residual assignment (vs. k-means centroids) retains more local morphological detail → stronger discriminability for crater edges and textures
- Self-supervised ViT > CLIP > ImageNet pretraining → domain-specific pretraining is the key factor for retrieval performance
- Attention-based seed selection provides the largest advantage at low K (K=16: +14 mAP over random); the gap narrows at high K
- Pretraining objective matters more than parameter count: 22M ViT-S/16 DINO (.420 mAP) outperforms 86M DeiT-B/16 (.303) and 134M VGG-16 (.068)

## Highlights & Insights
- **Task Redefinition**: The paradigm shift from detection (outputting coordinates) to retrieval (outputting similarity-ranked matches) directly addresses the authentic needs of planetary science workflows
- **"Supervised Degradation" Finding and Explanation**: In the few-view regime, metric learning lacks sufficient positive diversity → fine-tuning degrades general-purpose representations → frozen features with post-hoc processing is the correct strategy for such regimes
- **Generality of Training-Free Token Aggregation**: Not limited to craters—applicable to any scenario requiring efficient retrieval over frozen ViT features (remote sensing change detection, scene deduplication, geo-localization)
- **Methodological Contribution to GeoAI**: The pipeline of late interaction + deterministic compression + two-stage search is domain-agnostic

## Limitations & Future Work
- Only 2 views per ID → more views may restore the effectiveness of supervised methods
- Currently limited to Mars CTX → generalization to the Moon and other planetary bodies remains to be verified
- Seed token selection is attention-based → alternative saliency metrics may be superior
- The optimal value of K may vary with crater size and type

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First crater retrieval benchmark + training-free token aggregation + supervised degradation finding
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 30 backbones + 3 metric learning losses + K-value ablation + two-stage parameter analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain from problem definition → diagnosis → solution → experiments
- Value: ⭐⭐⭐⭐ Dual contributions to planetary science and GeoAI, with a generalizable retrieval methodology

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[ICCV 2025\] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)](../../ICCV2025/self_supervised/a_tokenlevel_text_image_foundation_model_for_document_unders.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)

<!-- RELATED:END -->
