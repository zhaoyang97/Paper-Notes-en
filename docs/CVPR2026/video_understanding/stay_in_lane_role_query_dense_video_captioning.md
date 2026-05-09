---
title: >-
  [Paper Note] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning
description: >-
  [CVPR 2026][Video Understanding][Dense Video Captioning] ROS-DVC introduces three complementary components for DETR-based dense video captioning (DVC): role-specific query initialization (separate localization and captioning queries), a cross-task contrastive alignment loss, and an overlap suppression loss. Without pretraining or LLMs, it achieves a CIDEr of 39.18 on YouCook2, surpassing DDVC which relies on GPT-2.
tags:
  - CVPR 2026
  - Video Understanding
  - Dense Video Captioning
  - Role-Specific Queries
  - Overlap Suppression
  - DETR
  - Cross-Task Contrastive Alignment
date: 2026-05-08
content_hash: 11cc30c5b7ab346d
---

# Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning

**Conference**: CVPR 2026
**arXiv**: [2603.11439](https://arxiv.org/abs/2603.11439)
**Code**: [github.com/MMAI-Konkuk/ROS-DVC](https://github.com/MMAI-Konkuk/ROS-DVC)
**Area**: Video Understanding
**Keywords**: Dense Video Captioning, Role-Specific Queries, Overlap Suppression, DETR, Cross-Task Contrastive Alignment

## TL;DR

ROS-DVC introduces three complementary components for DETR-based dense video captioning (DVC): role-specific query initialization (separate localization and captioning queries), a cross-task contrastive alignment loss, and an overlap suppression loss. Without pretraining or LLMs, it achieves a CIDEr of 39.18 on YouCook2, surpassing DDVC which relies on GPT-2.

## Background & Motivation

**State of the Field**: Dense video captioning (DVC) requires simultaneously localizing multiple temporal event segments in a video and generating natural language descriptions for each. PDVC pioneered the use of DETR architecture in DVC, enabling end-to-end joint optimization. Subsequent methods (CM2, MCCL, E2DVC, DDVC) have built upon this foundation.

**Limitations of Prior Work**: DETR-based DVC employs shared learnable queries to jointly drive both localization and captioning, leading to two critical issues: (1) **Multi-task interference** — localization requires broad temporal context to predict precise boundaries, while captioning demands dense attention to semantic details of key frames; the attention distribution of shared queries becomes ambiguous (Figure 1b, top); (2) **Prediction overlap** — multiple queries capture highly overlapping temporal segments, producing redundant descriptions (Figure 1a).

**Root Cause**: A fundamental conflict between the unified nature of queries and the divergent requirements of sub-tasks — a single query cannot simultaneously optimize temporal localization and semantic captioning.

**Paper Goals**: Enable queries to "stay in their lane" — localization queries focus on temporal boundaries while captioning queries focus on semantic content — while reducing temporal overlap among predictions.

**Starting Point**: Intervene at both query initialization and loss function levels — physically separating query spaces, maintaining consistency via contrastive alignment, and penalizing redundancy via overlap suppression.

**Core Idea**: Split DETR queries into two independently initialized groups for localization and captioning; bridge semantic consistency between the two groups via a contrastive loss; suppress prediction overlap via IoU-based penalties.

## Method

### Overall Architecture

Video frames → pretrained CLIP ViT-L/14 for feature extraction → Transformer encoder producing frame-level features → DETR decoder (receiving two independently initialized role-specific query sets) → localization queries output event segments via Hungarian matching; captioning queries output event descriptions after CTCA alignment. Localization queries are additionally constrained by OSL to reduce overlap.

### Key Designs

1. **Role-Specific Query Initialization**:
    - Function: Splits the standard DETR query set into localization queries $\{q_{\text{loc}}^j\}_{j=1}^K$ and captioning queries $\{q_{\text{cap}}^j\}_{j=1}^K$, each initialized from independent embedding spaces.
    - Mechanism: Both query groups share visual grounding in the decoder cross-attention layers (referencing the same visual positions via reference points), while maintaining independent representation spaces. Localization queries learn to attend broadly to temporal context for boundary prediction; captioning queries learn to attend densely to key frames for semantic capture. Unlike DDVC (which derives captioning queries from localization queries via MLP), this approach achieves true physical separation.
    - Design Motivation: Fully independent embedding spaces allow each query group to be optimized independently by its respective objective, avoiding gradient direction conflicts. Attention distribution visualizations in Figure 1b confirm that separated queries exhibit distinctly differentiated attention patterns.

2. **Cross-Task Contrastive Alignment Loss (CTCA)**:
    - Function: Ensures that localization and captioning queries at corresponding positions refer to the semantics of the same event.
    - Mechanism: After Hungarian matching, for the set of matched indices $\mathcal{M}$, $(q_{\text{cap}}^j, q_{\text{loc}}^j)$ are treated as positive pairs and $(q_{\text{cap}}^j, q_{\text{loc}}^{j'})$ as negative pairs. The loss is: $\mathcal{L}_{\text{CTCA}}=-\sum_{j\in\mathcal{M}}\log\frac{\exp(\text{sim}(\tilde{q}_{\text{cap}}^j,\tilde{q}_{\text{loc}}^j)/\tau)}{\sum_{j'}\exp(\text{sim}(\tilde{q}_{\text{cap}}^j,\tilde{q}_{\text{loc}}^{j'})/\tau)}$
    - Design Motivation: Query space separation no longer guarantees semantic consistency automatically; CTCA explicitly bridges localization and captioning queries through contrastive learning, endowing localization queries with semantic awareness.

3. **Overlap Suppression Loss (OSL)**:
    - Function: Penalizes excessive temporal overlap among predicted events to reduce redundant predictions.
    - Mechanism: Based on pairwise temporal IoU $P_o(i,j)$ between predicted boundaries $B_i, B_j$, a GT-aligned weight $\alpha=\gamma\cdot P_g+(1-\gamma)\cdot(1-P_g)$ is introduced ($\gamma\leq0.5$), yielding the final loss $L_{\text{OSL}}=-\alpha\cdot\log(\beta-P_o)$. Predictions with high GT alignment receive smaller penalties (large $P_g$ → small $\alpha$), preventing erroneous suppression of genuinely consecutive events.
    - Design Motivation: Directly optimizing overlap during training is more effective than NMS post-processing; GT-modulated penalties distinguish "legitimate overlap corresponding to GT" from "redundant and spurious overlap."

### Loss & Training

The total loss is:
$$\mathcal{L}_{\text{total}}=\lambda_{\text{giou}}\mathcal{L}_{\text{giou}}+\lambda_{\text{cls}}\mathcal{L}_{\text{cls}}+\lambda_{\text{cap}}\mathcal{L}_{\text{cap}}+\lambda_{\text{ec}}\mathcal{L}_{\text{ec}}+\lambda_{\text{CTCA}}\mathcal{L}_{\text{CTCA}}+\lambda_{\text{OSL}}\mathcal{L}_{\text{OSL}}+\lambda_{\text{CG}}\mathcal{L}_{\text{CG}}$$
where $\mathcal{L}_{\text{CG}}$ is an auxiliary cross-entropy loss from the Concept Guider (not used at inference). Hyperparameters: $\gamma=0.25$, $\beta=1.0$, $N_C=30$. A 2-layer deformable transformer decoder is used, with 50 query pairs for YouCook2 and 10 for ActivityNet.

## Key Experimental Results

### Main Results

| Dataset | Metric | ROS-DVC | DDVC (GPT-2) | MCCL | E2DVC | PDVC |
|---------|--------|---------|--------------|------|-------|------|
| YouCook2 | CIDEr↑ | **39.18** | 38.75 | 36.09 | 34.26 | 29.69 |
| YouCook2 | SODA_c↑ | **7.06** | 6.68 | 5.21 | 5.39 | 4.92 |
| YouCook2 | BLEU4↑ | **2.10** | 1.92 | 2.04 | 1.68 | 1.40 |
| ActivityNet | CIDEr↑ | **35.04** | — | 34.92 | 33.63 | 29.97 |
| ActivityNet | SODA_c↑ | **6.45** | — | 6.16 | 6.13 | 5.92 |

| Dataset | Localization Metric | ROS-DVC | E2DVC | PDVC |
|---------|---------------------|---------|-------|------|
| YouCook2 | Recall↑ | **29.34** | 24.36 | 22.89 |
| YouCook2 | F1↑ | **32.03** | 28.64 | 26.81 |
| ActivityNet | Recall↑ | **55.35** | 54.67 | 53.27 |

### Ablation Study

| Configuration | CIDEr | Notes |
|---------------|-------|-------|
| Baseline (E2DVC) | 34.26 | Shared queries |
| + Role separation | 36.14 (+1.88) | Query decoupling alone is effective |
| + Role separation + CTCA | 37.92 (+3.66) | Cross-task alignment preserves semantic consistency |
| + Role separation + CTCA + OSL | **39.18 (+4.92)** | Overlap suppression further reduces redundancy |
| OSL without GT modulation (fixed penalty) | ~38.4 | GT modulation prevents erroneous suppression of legitimate overlap |

### Key Findings

- The three components yield incrementally increasing contributions (+1.88, +1.78, +1.26), with the combined effect being optimal (+4.92).
- Role separation outperforms shared queries + CTCA (soft constraint), indicating that physical separation of representation spaces is superior to soft alignment.
- Recall and Precision are closely matched, suggesting the event count predictor produces estimates closer to ground truth.
- Without any LLM, ROS-DVC surpasses DDVC with GPT-2 (CIDEr +0.43), demonstrating the method's lightweight efficiency.

## Highlights & Insights

- The "let each query do its own job" philosophy is simple yet effective — addressing DVC multi-task interference at the level of DETR query design.
- The GT-modulated design of OSL is elegant — the adaptive penalty strength via $\alpha$ distinguishes legitimate overlap from redundant overlap.
- The Concept Guider provides zero-overhead auxiliary enhancement — an MLP predicts event concept vectors during training to enrich query representations, then is discarded at inference.
- The method does not rely on external memory banks or LLMs, making it lightweight and transferable.

## Limitations & Future Work

- Validation is limited to YouCook2 and ActivityNet; performance on longer or more complex video scenarios remains untested.
- Role separation doubles the query parameters (2K vs. K), which may introduce additional overhead with large query sets.
- CTCA's global contrastive objective may be insufficiently sensitive to distinguishing extremely short or long events; temporally-aware contrastive strategies are worth exploring.
- Comprehensive comparison with the latest LLM-based DVC methods (e.g., those using LLaMA) is lacking.
- The Concept Guider's concept vocabulary ($N_C=30$) is fixed, and generalization to out-of-domain videos remains to be verified.

## Related Work & Insights

- **vs. PDVC**: The pioneering DVC method with shared queries. ROS-DVC decouples queries and adds loss constraints on top of it, achieving a CIDEr gain of +9.49.
- **vs. DDVC**: Achieves CIDEr 38.75 using GPT-2 for caption generation. ROS-DVC surpasses it without any LLM (39.18), demonstrating that query design matters more than model capacity.
- **vs. E2DVC**: An improved end-to-end DVC baseline. ROS-DVC achieves CIDEr +4.92 over it.
- **vs. MCCL**: Uses an external memory bank to enhance caption diversity. ROS-DVC achieves higher CIDEr without any additional memory.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of role-separated queries and OSL GT modulation is novel; the three components are complementary and incrementally effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two standard datasets, component-wise incremental ablation, and multi-baseline multi-metric comparisons.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the "Stay in your Lane" title is apt; method diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Offers direct practical improvements for DVC; the role-separation concept is transferable to other DETR-based multi-task architectures.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](videochatm1_collaborative_policy_planning_for_vide.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[AAAI 2026\] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction](../../AAAI2026/video_understanding/explicit_temporal-semantic_modeling_for_dense_video_captioning_via_context-aware.md)

<!-- RELATED:END -->
