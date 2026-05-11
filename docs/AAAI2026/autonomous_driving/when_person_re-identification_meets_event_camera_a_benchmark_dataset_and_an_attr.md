---
title: >-
  [Paper Note] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework
description: >-
  [AAAI2026][Autonomous Driving][person re-identification] This paper presents EvReID, the first large-scale RGB-Event person re-identification dataset (1,200 identities / 118,988 image pairs), and proposes TriPro-ReID, a three-stage contrastive learning framework guided by pedestrian attributes. The framework leverages positive-negative attribute prompts and cross-modal prompt fusion to integrate RGB and Event modality features, achieving 69.3% mAP.
tags:
  - AAAI2026
  - Autonomous Driving
  - person re-identification
  - event camera
  - RGB-Event fusion
  - pedestrian attributes
  - benchmark dataset
date: 2026-05-08
content_hash: c3fd6619dcb73743
---

# When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework

**Conference**: AAAI2026
**arXiv**: [2507.13659](https://arxiv.org/abs/2507.13659)
**Code**: [Event-AHU/Neuromorphic_ReID](https://github.com/Event-AHU/Neuromorphic_ReID)
**Area**: Autonomous Driving
**Keywords**: person re-identification, event camera, RGB-Event fusion, pedestrian attributes, benchmark dataset

## TL;DR
This paper presents EvReID, the first large-scale RGB-Event person re-identification dataset (1,200 identities / 118,988 image pairs), and proposes TriPro-ReID, a three-stage contrastive learning framework guided by pedestrian attributes. The framework leverages positive-negative attribute prompts and cross-modal prompt fusion to integrate RGB and Event modality features, achieving 69.3% mAP.

## Background & Motivation

### State of the Field

**Background**: RGB camera-based person ReID faces persistent challenges in illumination variation, motion blur, and privacy protection.

### Root Cause

**Key Challenge**: Although event cameras offer advantages such as low power consumption, high dynamic range, and freedom from motion blur, existing event-based ReID datasets are extremely small in scale (Event-ReID contains only 33 identities / 16,000 samples), making it infeasible to evaluate real-world performance and generalization capability.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing methods focus solely on event feature learning or RGB-Event feature fusion, neglecting mid-level semantic information such as pedestrian attributes (e.g., long hair, wearing glasses).

### Starting Point

**Goal**: How to construct a large-scale, real-world RGB-Event person ReID benchmark dataset, and design a ReID framework that effectively exploits multimodal visual features alongside pedestrian attribute semantics?

## Method

### EvReID Dataset
- Captured using a DVS346 event camera at resolution $346 \times 260$
- 1,200 identities and 118,988 frame pairs (7× more images and 36× more identities than Event-ReID)
- Covers multiple seasons, diverse scenes, and day/night illumination conditions
- RGB modality augmented with 11 types of noise (illumination changes, motion blur, adverse weather) to validate complementary learning
- 70%/30% train/test split with a single-shot evaluation protocol

### TriPro-ReID Framework (Three-Stage Training)

**Stage 1: Text Prompt Alignment**
- Built upon CLIP-ReID; learns per-identity text prompt tokens $[X]_1, ..., [X]_n$
- Visual and text encoders are frozen; only ID-specific prompts are optimized
- Loss: $L_{stage1} = L_{v2t} + L_{t2v}$

**Stage 2: Multimodal Prompt Alignment**
- Introduces Cross-Modal Prompt (CMP): learnable prompt tokens initialized in the RGB branch are projected to the Event branch via a fully connected layer
- CMP propagates synchronously across all Transformer layers, enabling continuous cross-modal feature fusion

**Stage 3: Visual-Modal Tuning with Attribute Prompts**
- Employs pretrained pedestrian attribute recognition model VTFPAR++ to predict attributes
- Constructs Positive-Negative Attribute Prompt (PNAP): positive attributes (e.g., "Male, Jacket, Bald") combined with negative attributes (e.g., "Not Female, Not Short Sleeves")
- PNAP encodings are injected into intermediate ViT layers to dynamically modulate visual features
- Loss: $L_{stage3} = L_{id} + L_{tri} + L_{v2t} + L_{t2v}$

## Key Experimental Results

**On EvReID (V+E modalities):**

### Main Results

| Method | mAP | Rank-1 | Rank-5 |
|--------|-----|--------|--------|
| CLIMB-ReID | 68.3 | 85.2 | 92.8 |
| AP3D | 66.9 | 86.5 | 95.6 |
| **TriPro-ReID** | **69.3** | **88.6** | **94.3** |

**On MARS\* dataset (V+E modalities):**
- TriPro-ReID: mAP 88.4, Rank-1 91.1

**Ablation (EvReID):**

### Ablation Study

| Configuration | mAP | Rank-1 |
|---------------|-----|--------|
| Base only | 49.2 | 73.0 |
| +PNAP | 62.3 | 81.1 |
| +CMP | 50.2 | 75.2 |
| +PNAP+CMP | **69.3** | **88.6** |

### Key Findings

- PNAP yields the largest improvement (+13.1 mAP); using positive prompts alone achieves 54.4 mAP, confirming the significant benefit of combining positive and negative attribute prompts.

## Highlights & Insights
- **First large-scale real-world RGB-Event ReID dataset**: 36× more identities than its predecessor, with multi-season and multi-illumination coverage.
- **Elegant positive-negative attribute prompt design**: Leverages not only "what attributes are present" but also "what attributes are absent" as discriminative cues; ablation studies validate the critical role of negative attributes.
- **Progressive three-stage training strategy**: Proceeds from text alignment → multimodal fusion → attribute tuning, introducing information incrementally for stable training.
- **Systematic evaluation of 15 SOTA baselines**: Provides the community with a comprehensive benchmark.

## Limitations & Future Work
- EvReID's low resolution ($346 \times 260$) limits fine-grained feature learning.
- Pedestrian attribute quality depends on the predictions of the external pretrained model VTFPAR++.
- Event-only modality performance is substantially lower than RGB (e.g., AP3D: 40.6 vs. 65.4 mAP), indicating considerable room for improvement in event feature utilization.
- Open-set and cross-domain evaluation scenarios remain unexplored.

## Related Work & Insights
- vs. SDCL (CVPR2023): Both target RGB-Event fusion ReID, but SDCL lacks attribute semantic guidance, resulting in 15.1 lower mAP.
- vs. CLIP-ReID (AAAI2023): TriPro-ReID extends it by introducing CMP and PNAP, improving V+E modality mAP from 49.2 to 69.3.
- vs. CLIMB-ReID (AAAI2025): Uses a Mamba architecture; TriPro-ReID outperforms it on both mAP and Rank-1.

## Related Work & Insights
- The positive-negative attribute prompt paradigm is generalizable to other fine-grained recognition tasks (e.g., vehicle ReID, animal identification).
- Event cameras have unique value in privacy-sensitive scenarios (e.g., intelligent surveillance); RGB-Event fusion is a growing research direction.
- The three-stage prompt learning strategy can serve as a reference for other multimodal CLIP-based downstream tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (significant dataset contribution + novel positive-negative attribute prompts)
- Experimental Thoroughness: ⭐⭐⭐⭐ (15 baselines + complete ablation + dual-dataset evaluation)
- Writing Quality: ⭐⭐⭐⭐ (well-structured with thorough dataset description)
- Value: ⭐⭐⭐⭐ (benchmark dataset provides long-term value to the community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[AAAI 2026\] Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification](debiased_dual-invariant_defense_for_adversarially_robust_person_re-identificatio.md)
- [\[CVPR 2026\] FedBPrompt: Federated Domain Generalization Person Re-Identification via Body Distribution Aware Visual Prompts](../../CVPR2026/autonomous_driving/fedbprompt_federated_domain_generalization_person_re-identification_via_body_dis.md)
- [\[NeurIPS 2025\] GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification](../../NeurIPS2025/autonomous_driving/gsalign_geometric_and_semantic_alignment_network_for_aerial-ground_person_re-ide.md)
- [\[AAAI 2026\] TSBOW: Traffic Surveillance Benchmark for Occluded Vehicles Under Various Weather Conditions](tsbow_traffic_surveillance_benchmark_for_occluded_vehicles_under_various_weather.md)

</div>

<!-- RELATED:END -->
