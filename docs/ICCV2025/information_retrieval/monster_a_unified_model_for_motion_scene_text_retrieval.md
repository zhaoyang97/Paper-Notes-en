---
title: >-
  [Paper Note] MonSTeR: a Unified Model for Motion, Scene, Text Retrieval
description: >-
  [ICCV 2025][Information Retrieval & RAG][tri-modal retrieval] This paper proposes **MonSTeR**—the first **tri-modal retrieval model for motion, scene…
tags:
  - "ICCV 2025"
  - "Information Retrieval & RAG"
  - "tri-modal retrieval"
  - "motion-scene-text"
  - "higher-order relationship modeling"
  - "contrastive learning"
  - "human-scene interaction evaluation"
date: 2026-05-08
content_hash: 59153d06089d4dd0
---

# MonSTeR: a Unified Model for Motion, Scene, Text Retrieval

**Conference**: ICCV 2025
**arXiv**: 2510.03200  
**Code**: [GitHub](https://github.com/colloroneluca/MonSTeR)  
**Area**: Information Retrieval
**Keywords**: tri-modal retrieval, motion-scene-text, higher-order relationship modeling, contrastive learning, human-scene interaction evaluation
**Authors**: Luca Collorone, Matteo Gioia et al. (Sapienza University of Rome, Technion/NVIDIA)

## TL;DR

This paper proposes **MonSTeR**—the first **tri-modal retrieval model for motion, scene, and text**—which constructs a unified latent space via higher-order relationship modeling inspired by topological deep learning. By capturing intrinsic dependencies among all three modalities, MonSTeR substantially outperforms baselines that rely solely on unimodal representations across multiple retrieval tasks, and can further serve as an evaluation tool for human-scene interaction models.

## Background & Motivation

When humans navigate complex environments, they must balance their intentions against the affordances offered by the surroundings. For instance, the intent "sit on a chair" accompanied by the corresponding motion becomes implausible if no chair is present in the scene. This observation reveals a strong intrinsic consistency among **intent (text), motion (action), and environment (scene)**.

Nevertheless, a clear gap exists in prior work:

**Text-motion retrieval models** (TMR, MoPa) cannot incorporate environmental context, leaving motion representation divorced from scene information.

**Evaluation of human-scene interaction (HSI) models** lacks a global coherence/realism metric; existing approaches typically decompose evaluation into independent measures such as collision detection and goal-object distance, neglecting path plausibility and motion credibility.

**Existing multimodal alignment methods** either perform only pairwise alignment or process all modalities through a shared encoder without explicit cross-modal interaction modeling.

The core challenge lies in effectively representing many-to-many relationships among three modalities within a unified latent space—the same text can correspond to multiple motions, and the same motion may carry different meanings across different scenes.

## Method

### Overall Architecture

MonSTeR is built upon three types of encoders:

1. **Unimodal encoders**: Transformer-based variational autoencoders that separately encode text $t$, motion $m$, and scene $s$, producing latent variables $v_t$, $v_m$, $v_s$.
2. **Cross-modal encoders**: The output tokens of the unimodal encoders are concatenated pairwise and fed into cross-modal encoders to produce joint latent variables $v_{st}$, $v_{mt}$, $v_{ms}$.

### Data Representation

- **Text** $t \in \mathbb{R}^{768}$: DistilBERT features
- **Scene** $s \in \mathbb{R}^{N \times 6}$: colored point cloud (x, y, z + RGB)
- **Motion** $m \in \mathbb{R}^{T \times 3 \times 22}$: $T$ frames × 3D coordinates × 22 joints

### Topological Modeling of Higher-Order Relationships

Inspired by topological deep learning, the three-modal relationships are modeled as a topological structure comprising nodes, edges, and faces:

- **Nodes** $\mathcal{V} = \{t, s, m\}$: unimodal representations
- **Edges** $\mathcal{E} = \{ts, sm, mt\}$: cross-modal representations
- **Face** $\mathcal{P} = \{tsm\}$: global tri-modal relationship

Higher-order relationships are encoded by aligning unimodal representations with cross-modal ones:
- $(st, m)$: scene-text cross-modal aligned with motion
- $(mt, s)$: motion-text cross-modal aligned with scene
- $(ms, t)$: motion-scene cross-modal aligned with text

### Training Objective

The set of pairs included in the contrastive learning loss is:
$$K = \{(t,s), (m,t), (m,s), (st,m), (mt,s), (ms,t)\}$$

Pairs that could induce degenerate solutions (e.g., $(st, t)$ or $(st, s)$) are excluded to prevent the cross-modal encoders from learning identity mappings.

For each pair $(i,j) \in K$, an $N \times N$ cosine similarity matrix $C_{i,j}$ is computed and aggregated via InfoNCE:
$$\mathcal{L}_{\text{tot}} = \frac{1}{|K|} \sum_{(i,j) \in K} \frac{\mathcal{L}_{\text{NCE}}(C_{i,j})}{N}$$

### Retrieval Inference

The unified latent space supports flexible retrieval across multiple configurations:
- **Bi-modal → uni-modal**: st2m, ms2t, mt2s (given two modalities, retrieve the third)
- **Uni-modal → bi-modal**: m2st, t2ms, s2mt (given one modality, retrieve two)
- **Uni-modal → uni-modal**: t2m, m2t, s2m, m2s, t2s, s2t

## Key Experimental Results

### Main Results: HUMANISE+ Retrieval (All Protocol, mRecall)

| Method | st2m | m2st | ms2t | t2sm | tm2s | s2mt | Avg. |
|--------|------|------|------|------|------|------|------|
| TMR + S | 4.10 | 3.30 | 5.81 | 4.79 | 1.08 | 1.98 | 2.72 |
| MoPa + S | 2.10 | 2.45 | 1.62 | 1.94 | 3.28 | 3.06 | 1.88 |
| **MonSTeR** | **13.91** | **13.14** | **8.46** | **10.39** | **4.09** | **4.45** | **4.80** |

MonSTeR achieves a relative improvement of **209%** over the strongest baseline on st2m, and surpasses the best scene-aware model by **76.47%** in average mRecall.

### Ablation Study: Necessity of Higher-Order Relationship Modeling

| Variant | st2m | m2st | t2m | m2t | Avg. |
|---------|------|------|-----|-----|------|
| **MonSTeR** | 13.91 | 13.14 | 3.62 | 3.11 | 5.63 |
| w/o cross-modal | 5.20 | 3.77 | 4.35 | 3.21 | 3.79 |
| w/o single | 11.91 | 12.93 | 0.22 | 0.29 | 4.36 |
| w tri-modal | 6.14 | 6.00 | 4.16 | 4.37 | 4.14 |

| Variant | Small Batches Avg. |
|---------|--------------------|
| **MonSTeR** | **60.00** |
| w/o cross-modal | 56.07 |
| w/o single | 41.77 |
| w tri-modal | 53.70 |

Key Findings:
- Removing the cross-modal encoder leads to a substantial drop in bi-modal → uni-modal tasks.
- Removing unimodal alignment terms causes near-complete collapse on uni-modal → uni-modal tasks.
- A shared tri-modal encoder is inferior to the separated cross-modal encoder design.

### Path Plausibility Evaluation

After rotating test-set motions from 0 to $\pi$ radians, MonSTeR's FID and Recall degrade consistently as expected; moreover, motions involving collisions receive lower scores—demonstrating that the MonSTeR latent space **internalizes the prior that motion should not penetrate scene objects**.

### User Study

MonSTeR's rankings align with human preferences at a rate of **66.5%** (1,122 annotations, 224 evaluators).

### Downstream Task: Motion Captioning

| Method | BLEU 1 | BLEU 4 | ROUGE L | CIDEr | BERT F1 |
|--------|--------|--------|---------|-------|---------|
| MotionGPT | 42.16 | 17.47 | 40.23 | 11.13 | 22.16 |
| **MonSTeR + GPT2** | **42.93** | **23.59** | **50.85** | **13.70** | **35.57** |

When MonSTeR embeddings are used for motion captioning, the model substantially outperforms MotionGPT on BLEU 4 (+6.12), ROUGE L (+10.62), and BERT F1 (+13.41).

### Zero-Shot Scene Object Placement

Using mt2s scores to localize objects within a $5 \times 5 \times 5$ grid, the average error is only **18 cm** (random baseline: 58.98 cm), demonstrating MonSTeR's precise spatial reasoning capability in a zero-shot setting.

## Highlights & Insights

1. **First tri-modal retrieval system**: The unified latent space for motion, scene, and text has never been explored before; MonSTeR fills this gap.
2. **Topology-inspired higher-order modeling**: Rather than simple pairwise alignment or fully mixed multi-modal encoding, edge-node alignment is used to encode higher-order relationships, grounded in topological theory.
3. **Evaluation capability**: MonSTeR can replace conventional collision detection + distance metrics, providing a holistic consistency score that aligns with human judgment.
4. **Flexible retrieval**: Supports 12 retrieval task combinations, including uni-modal → bi-modal and bi-modal → uni-modal directions.
5. **Zero-shot transfer**: Zero-shot and downstream experiments on object placement and motion captioning validate the rich semantics of the learned latent space.

## Limitations & Future Work

1. **Training on aligned data only**: Cross-modal encoders are trained exclusively on paired data, leaving the potential of unpaired data unexploited.
2. **Static scene assumption**: Human actions are assumed not to alter the scene layout, limiting the modeling of dynamic interactions.
3. **Dataset scale**: HUMANISE+ (19.6K samples) and TRUMANS+ (15 hours of motion capture) remain relatively small.
4. **TMR still leads on t2m in TRUMANS+**: This suggests that when scene information is not discriminative, specialized bi-modal models may have an advantage.

## Related Work & Insights

- **Relationship to TMR/MoPa**: MonSTeR extends text-motion retrieval by incorporating the scene dimension, generalizing it to a tri-modal setting.
- **Difference from CLIP-style multimodal alignment**: CLIP-based methods typically align to a single reference modality, whereas MonSTeR achieves all-to-all alignment.
- **Implications for HSI evaluation**: Traditional collision + distance metrics can be replaced by a single tri-modal consistency score.
- **Connection to topological deep learning**: Applying the topological framework of Bodnar et al. 2021 to multimodal alignment represents a novel cross-domain idea.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First tri-modal retrieval model for motion, scene, and text; the topology-inspired higher-order modeling approach is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across 12 retrieval tasks, thorough ablations, and diverse downstream applications (captioning, object placement, user study).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, an engaging Beatles-inspired introduction, and well-motivated topological framing.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction in tri-modal retrieval and offers practical utility for evaluating human-scene interaction models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching](aligning_information_capacity_between_vision_and_language_via_dense_to_sparse_feature_distillation.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](../../ICLR2026/information_retrieval/g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[NeurIPS 2025\] The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface](../../NeurIPS2025/information_retrieval/the_transparent_earth_a_multimodal_foundation_model_for_the_earths_subsurface.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)

</div>

<!-- RELATED:END -->
