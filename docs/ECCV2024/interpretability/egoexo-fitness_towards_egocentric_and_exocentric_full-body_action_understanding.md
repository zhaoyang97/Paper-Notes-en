---
title: >-
  [Paper Note] EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding
description: >-
  [ECCV 2024][Interpretability][Egocentric Vision] This paper proposes the EgoExo-Fitness dataset, which contains synchronized egocentric and exocentric fitness videos. It provides two-level temporal boundary annotations and innovative explainable action assessment labels (technical keypoint verification, natural language commentary, and quality scoring) and establishes five benchmark tasks.
tags:
  - "ECCV 2024"
  - "Interpretability"
  - "Egocentric Vision"
  - "Full-Body Action Understanding"
  - "Fitness Dataset"
  - "Cross-view"
  - "Explainable Action Assessment"
date: 2026-05-08
content_hash: a22d6d2b4917b0cf
---

# EgoExo-Fitness: Towards Egocentric and Exocentric Full-Body Action Understanding

**Conference**: ECCV 2024  
**arXiv**: [2406.08877](https://arxiv.org/abs/2406.08877)  
**Code**: [GitHub](https://github.com/iSEE-Laboratory/EgoExo-Fitness)  
**Area**: Human Understanding  
**Keywords**: Egocentric Vision, Full-Body Action Understanding, Fitness Dataset, Cross-view, Explainable Action Assessment

## TL;DR

This paper proposes the EgoExo-Fitness dataset, which contains synchronized egocentric and exocentric fitness videos. It provides two-level temporal boundary annotations and innovative explainable action assessment labels (technical keypoint verification, natural language commentary, and quality scoring) and establishes five benchmark tasks.

## Background & Motivation

Imagine wearing smart glasses while working out, where a virtual coach can tell you what you did, when you did it, and how well you performed. Achieving this vision requires **Egocentric Full-Body Action Understanding (EgoFBAU)** capabilities, but existing research has three main gaps:

**Single-View Datasets**: Existing full-body action datasets (e.g., NTU-RGB+D, FineGym, FineDiving) are predominantly captured from exocentric fixed cameras, which limits their application in flexible scenarios.

**Limited Scenarios in Egocentric Datasets**: Existing egocentric datasets (e.g., Ego4D, EPIC-KITCHENS) mainly focus on tabletop manipulation and daily interactions rather than full-body action understanding.

**Lack of Explainable Assessment Annotations**: Existing action quality assessment datasets only provide scores or rankings, making it impossible to directly explore the interpretability of the evaluation (i.e., why a certain score was given).

EgoExo-Fitness fills these gaps by simultaneously providing ego/exo videos and rich annotations.

## Method

### Overall Architecture

EgoExo-Fitness is a **dataset + benchmark** contribution that corely consists of three parts:
1. Multi-view recording system design and data collection.
2. Multi-level rich annotation system.
3. Five benchmark tasks and experimental analysis.

### Key Designs

#### 1. Recording System

**Egocentric (Ego)**: A headset with three action cameras is designed:
- Ego-M: GoPro capturing the straight-ahead view.
- Ego-L / Ego-R: Two Insta-Go3 cameras capturing the bottom-left and bottom-right views, respectively, to capture more body details.

**Exocentric (Exo)**: Three fixed cameras placed in front (Exo-M), front-left (Exo-L), and front-right (Exo-R) of the participant.

All cameras are manually synchronized via visible timing events.

#### 2. Data Collection

- **12 fitness action categories**: Covering chest, abdomen, waist, hips, and full-body driver muscles (e.g., kneeling push-ups, push-ups, sit-ups, high knees, jumping jacks, etc.).
- **86 action sequences**: Randomly combining 3-6 different actions to enrich temporal diversity.
- **Natural collection**: Participants only received text instructions and completed actions naturally, with each action repeated at least 4 times.
- Scale: **1276 cross-view sequence videos, 6131 single actions, totaling approximately 32 hours**.

#### 3. Annotation System (Core Innovation)

**Two-Level Temporal Boundary Annotations**:
- Level 1: Locating the start and end times of each individual action from the action sequence videos.
- Level 2: Dividing a single action video into three substeps—Getting ready, Executing, and Relaxing.

**Explainable Action Assessment Annotations** (three progressive layers):

**(1) Technical Keypoint Verification**:
- Text guidance is provided for each type of action.
- LLM is used to split the guidance into several technical keypoints.
- Annotators verify whether the actions satisfy each keypoint one by one (True/False).

**(2) Natural Language Commentary**:
- Annotators write paragraph-style reviews based on keypoint verification results.
- Including well-performed aspects and suggestions for improvement.

**(3) Action Quality Scoring**: A subjective score of 1-5, with each action annotated by at least 2 database experts.

#### 4. Five Benchmark Tasks

1. **Action Classification**: Predicting fitness action categories from single action videos.
2. **Action Localization**: Temporal action detection.
3. **Cross-View Sequence Verification (CVSV)**: Verifying whether two videos from different views perform the same action sequence (newly proposed).
4. **Cross-View Skill Assessment**: Action quality assessment across views.
5. **Guidance-based Execution Verification (GEV)**: Determining whether the action execution satisfies the given technical keypoints (newly proposed).

### Loss & Training

As a dataset paper, the focus is on experimental benchmarks rather than specific model designs. Baseline models include pre-trained models such as I3D, TimeSformer, and EgoVLP, as well as a specialized sequence verification model, CAT.

## Key Experimental Results

### Main Results — Action Classification

| Training Data | Model | Pre-training | Exo Test↑ | Ego Test↑ |
|----------|------|--------|----------|----------|
| Exo | I3D | K400 | 0.9194 | 0.0927 |
| Ego | I3D | K400 | 0.1025 | 0.7469 |
| Ego&Exo | I3D | K400 | 0.8963 | 0.7266 |
| Exo | TSF | K600 | 0.9274 | 0.0836 |
| Ego | EgoVLP | Ego4D | 0.0887 | 0.7977 |
| Ego | TSF | EE4D | 0.1601 | 0.8000 |

### Cross-View Sequence Verification

| Training Data | Ego-Ego AUC↑ | Exo-Exo AUC↑ | Exo-Ego AUC↑ |
|----------|-------------|-------------|-------------|
| Exo-Exo | 0.532 | 0.800 | 0.577 |
| Ego-Ego | 0.803 | 0.487 | 0.480 |
| Exo-Ego | 0.761 | 0.813 | 0.744 |
| All | 0.751 | 0.814 | 0.743 |

Cross-view retrieval performance: Ego $\rightarrow$ Exo Rank-1 is only 0.296, and mAP is only 0.228, which is far lower than same-view retrieval.

### Ablation Study

**Impact of Pre-training on Views**: Kinetics pre-training achieves the best performance on Exo (0.9274), while Ego-Exo4D pre-training performs best on Ego (0.8000), aligning with the view of the pre-training data.

**Ego Training Data Proportion Experiment**: Gradually reducing the ego training data ($100\% \rightarrow 70\% \rightarrow 30\% \rightarrow 0\%$) leads to a continuous decline in all metrics, indicating that cross-view learning under limited ego data is a significant challenge.

### Key Findings

1. **Huge Viewpoint Gap**: Models trained solely on ego data fail almost completely on exo (<0.1), and vice versa.
2. **Mixed Training is Not Always Effective**: Training with mixed ego and exo data does not guarantee improvements and may even degrade performance on specific views.
3. **Ego is More Challenging than Exo**: Models consistently achieve lower accuracy on ego videos than on exo videos. This is because action patterns in egocentric views are more similar and offer fewer discriminative cues.
4. **Cross-View Sequence Verification is Highly Challenging**: The AUC for ego-exo pairs (0.744) is significantly lower than for same-view pairs (ego-ego 0.803, exo-exo 0.814).
5. **Dilemma of Limited Ego Data**: Reducing the proportion of ego data leads to continuous performance degradation, whereas collecting egocentric data is much harder than exocentric data in practice.

## Highlights & Insights

1. **First Systematic Annotation of Explainable Action Assessment**: The three-level annotation system (technical keypoint verification + natural language commentary + quality scoring) opens up a new direction for explainable action assessment.
2. **First Proposal of Cross-View Sequence Verification**: Extending traditional SV to cross-view scenarios closely aligns with the practical demands of smart wearable devices.
3. **Unique Downward-Looking Ego Camera Design**: In addition to the straight-ahead view, two cameras pointing bottom-left and bottom-right are used to capture more body details, compensating for the lack of body visibility in the standard front-facing egocentric camera.
4. **Action Sequence Design**: Each video segment contains 3-6 different actions, naturally supporting action localization and sequence verification tasks.

## Limitations & Future Work

- The dataset scale (32 hours) is relatively small compared to Ego-Exo4D (hundreds of hours).
- It only covers 12 types of fitness actions, offering limited movement diversity.
- Manual camera synchronization might introduce slight temporal deviations.
- Benchmark experiments mainly utilize existing models, without proposing methods specifically tailored for cross-view challenges.
- Multimodal evaluation frameworks combining technical keypoint verification with large language models can be explored.

## Related Work & Insights

- **Ego-Exo4D**: A concurrent large-scale ego-exo dataset, but EgoExo-Fitness focuses specifically on fitness scenarios and offers unique keypoint verification annotations.
- **FLAG3D, FineDiving**: Exocentric-only full-body action datasets lacking egocentric perspectives.
- **Ego4D**: A large-scale egocentric dataset, but it rarely involves full-body action understanding.
- **CAT (Sequence Verification Model)**: Exhibits a significant performance drop in cross-view settings, indicating a need for new cross-view temporal modeling methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first ego-exo fitness dataset with explainable action assessment annotations.
- Technical Depth: ⭐⭐⭐ — Primarily a dataset contribution with limited methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 tasks with multi-dimensional analysis and exhaustive view impact studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich statistical visualizations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT](../../AAAI2026/interpretability/attention_gathers_mlps_compose_a_causal_analysis_of_an_action-outcome_circuit_in.md)
- [\[CVPR 2025\] Geometry-Guided Camera Motion Understanding in VideoLLMs](../../CVPR2025/interpretability/geometry-guided_camera_motion_understanding_in_videollms.md)
- [\[ICML 2025\] Position: We Need An Algorithmic Understanding of Generative AI](../../ICML2025/interpretability/position_we_need_an_algorithmic_understanding_of_generative_ai.md)
- [\[ICLR 2026\] Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](../../ICLR2026/interpretability/towards_understanding_the_nature_of_attention_with_low-rank_sparse_decomposition.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](../../NeurIPS2025/interpretability/understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)

</div>

<!-- RELATED:END -->
