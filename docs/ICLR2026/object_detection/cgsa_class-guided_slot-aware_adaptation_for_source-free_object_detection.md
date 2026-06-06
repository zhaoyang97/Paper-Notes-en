---
title: >-
  [Paper Note] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection
description: >-
  [ICLR 2026][Object Detection][source-free domain adaptation] This paper is the first to introduce Object-Centric Learning (Slot Attention) into Source-Free Domain-Adaptive Object Detection (SF-DAOD). It extracts domain-i…
tags:
  - "ICLR 2026"
  - "Object Detection"
  - "source-free domain adaptation"
  - "object-centric learning"
  - "slot attention"
  - "DETR"
  - "contrastive learning"
date: 2026-05-08
content_hash: f074c05e9457801b
---

# CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection

**Conference**: ICLR 2026
**arXiv**: [2602.22621](https://arxiv.org/abs/2602.22621)  
**Code**: [GitHub](https://github.com/Michael-McQueen/CGSA)  
**Area**: Object Detection
**Keywords**: source-free domain adaptation, object-centric learning, slot attention, DETR, contrastive learning

## TL;DR

This paper is the first to introduce Object-Centric Learning (Slot Attention) into Source-Free Domain-Adaptive Object Detection (SF-DAOD). It extracts domain-invariant object-level structural priors via a hierarchical slot-aware module and drives domain-invariant representations through class-guided contrastive learning, achieving substantial improvements over existing methods across multiple cross-domain benchmarks.

## Background & Motivation

**Domain shift problem**: Object detectors suffer significant performance degradation when deployed under domain shifts caused by weather, camera, or scene variations.

**SF-DAOD constraints**: Only a source-domain pretrained model and unlabeled target-domain data are available; source data cannot be accessed due to privacy or copyright restrictions.

**Limitations of prior work**: Mainstream SF-DAOD methods (SFOD/PETS/A2SFOD) focus on pseudo-label threshold tuning or teacher-student framework improvements, neglecting the shared object-level structural information across cross-domain data.

**Potential of Slot Attention**: Object-Centric Learning (OCL) decomposes a scene into discrete "slot" representations, each binding to one object, naturally isolating foreground from background. It has demonstrated strong transferability in segmentation, video prediction, and robotics, but has never been applied to SF-DAOD.

**Natural fit**: DETR-based detectors already employ object queries, making the embedding of slot priors into query space a natural yet unexplored direction.

## Method

### Overall Architecture

A two-stage pipeline: source-domain pretraining (standard detection loss + HSA reconstruction loss) → target-domain adaptation (Teacher-Student + HSA + CGSC).

### HSA (Hierarchical Slot Awareness) Module

1. **Two-stage decomposition**: Stage one applies iterative Slot Attention to extract $n=5$ coarse-grained slots → spatial broadcast MLP decodes and reconstructs → softmax competition ensures each slot binds to a distinct region. Stage two takes the reconstructed features as input and applies Slot Attention again to obtain $n^2=25$ fine-grained slots.
2. **Reconstruction loss**: $\mathcal{L}_{rec} = \|\hat{h}^{(1)} - h\|_2^2 + \|\hat{h}^{(2)} - h\|_2^2$, supervising both stages.
3. **Slot-Aware Queries**: Projected slots are added to object queries: $Q_{aware} = Q_{obj} + f_{map}(z^{(2)})$, providing object-level structural priors to the decoder.

### CGSC (Class-Guided Slot Contrast) Module

1. **Class prototype memory**: Maintains global class prototypes $P_c$ updated via EMA, aggregated from decoder queries by averaging over predicted class assignments.
2. **Weighted slot construction**: Applies attention masks $m_k^{(2)}$ to weight-aggregate raw features, suppressing background slots.
3. **Hungarian matching**: A cosine similarity matrix combined with the Hungarian algorithm establishes one-to-one correspondences between weighted slots and queries, yielding pseudo class labels.
4. **InfoNCE contrastive loss**: Pulls the per-class slot prototype $\bar{z}_c$ closer to the class prototype $P_c$ while pushing apart prototypes of different classes.

### Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{unsup} + \lambda_{con} \mathcal{L}_{con} + \lambda_{rec} \mathcal{L}_{rec}$$

### Theoretical Guarantee

A target-domain risk reduction bound is proven: $\mathbb{E}[\mathcal{R}_T(\theta_{t+1})] \le \mathbb{E}[\mathcal{R}_T(\theta_t)] - c_1 \Delta_t + c_2(\epsilon_{rec} + \sigma^2)$

## Key Experimental Results

### Main Results

| Cross-Domain Setting | SF | Method | mAP |
|---|---|---|---|
| Cityscapes→BDD100K | ✗ | DATR (Source-based DAOD) | 43.3 |
| Cityscapes→BDD100K | ✓ | TITAN (SF-DAOD) | 38.3 |
| Cityscapes→BDD100K | ✓ | **CGSA** | **53.0** |
| Cityscapes→Foggy | ✓ | A2SFOD | 41.2 |
| Cityscapes→Foggy | ✓ | **CGSA** | **49.8** |

### Ablation Study

| Configuration | Cityscapes→BDD100K mAP | Note |
|---|---|---|
| Teacher-Student only | 35.4 | No structural prior |
| +HSA | 45.2 | Slot structural prior is effective |
| +CGSC | 41.8 | Class-guided contrast is effective |
| **+HSA+CGSC (CGSA)** | **53.0** | Both modules are complementary; best overall |

### Key Findings

- CGSA in the source-free setting even surpasses several source-based DAOD methods that require access to source data.
- Training is conducted on 4×A100 GPUs using the RT-DETR detector.
- CGSA consistently leads across multiple cross-domain scenarios (clear→foggy, real→cartoon/watercolor, etc.).
- The 25-slot configuration exceeds the conventional OCL limit of ≤10 slots, while the hierarchical design ensures stable convergence.

## Highlights & Insights

- **First combination of OCL and SF-DAOD**: Opens a new paradigm of leveraging object-level structural priors for domain adaptation.
- The hierarchical slot design elegantly overcomes the traditional slot count limitation (5→25) while maintaining training stability.
- Theoretical generalization analysis is provided — the slot-aware design is not only empirically effective but also theoretically grounded.
- **Surpassing source-based methods under a source-free setting** constitutes compelling experimental evidence.

## Limitations & Future Work

- Validation is limited to driving-scene datasets; generalization to medical, aerial, and industrial domains remains untested.
- The slot count $n=5$ is manually set; an adaptive mechanism may be preferable.
- Hungarian matching depends on detector prediction quality; instability in early training stages may introduce erroneous class labels.
- The two-stage Slot Attention and reconstruction objectives in HSA increase training time and memory overhead.

## Related Work & Insights

- **SFOD/PETS/A2SFOD**: Focus on pseudo-label filtering while ignoring object-level structure.
- **DATR/MRT** (source-based DAOD): Require source data, yet CGSA outperforms them without it.
- **Slot Attention/SAVi**: Originally designed for segmentation and video prediction; introduced to domain-adaptive detection for the first time here.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First combination of OCL and SF-DAOD, opening a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets/cross-domain settings, complete ablation, and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, supported by both theory and experiments.
- Value: ⭐⭐⭐⭐ Provides a new methodological foundation for SF-DAOD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SFUOD: Source-Free Unknown Object Detection](../../ICCV2025/object_detection/sfuod_source-free_unknown_object_detection.md)
- [\[ICML 2026\] Smoothing Slot Attention Iterations and Recurrences](../../ICML2026/object_detection/smoothing_slot_attention_iterations_and_recurrences.md)
- [\[AAAI 2026\] Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](../../AAAI2026/object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)
- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](../../CVPR2026/object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](../../CVPR2026/object_detection/paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)

</div>

<!-- RELATED:END -->
