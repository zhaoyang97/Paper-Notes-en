---
title: >-
  [Paper Note] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection
description: >-
  [NeurIPS 2025][Human Understanding][Social Interaction Detection] This paper proposes a part-aware bottom-up group reasoning framework that enhances individual embeddings with pose-guided body part features and infers social groups via similarity-based association, achieving new state-of-the-art results on the NVI and Café datasets.
tags:
  - NeurIPS 2025
  - Human Understanding
  - Social Interaction Detection
  - Body Part Awareness
  - Bottom-Up Reasoning
  - Nonverbal Interaction
  - Pose Guidance
date: 2026-05-08
content_hash: af25ec3c7cc8799f
---

# Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection

**Conference**: NeurIPS 2025
**arXiv**: [2511.03666](https://arxiv.org/abs/2511.03666)
**Code**: N/A
**Area**: Human Understanding
**Keywords**: Social Interaction Detection, Body Part Awareness, Bottom-Up Reasoning, Nonverbal Interaction, Pose Guidance

## TL;DR

This paper proposes a part-aware bottom-up group reasoning framework that enhances individual embeddings with pose-guided body part features and infers social groups via similarity-based association, achieving new state-of-the-art results on the NVI and Café datasets.

## Background & Motivation

**Background**: Social interaction understanding encompasses tasks such as group activity recognition, pedestrian trajectory prediction, and group activity detection. The recent NVI-DET task requires detecting fine-grained nonverbal interactions (facial expressions, gestures, postures, gaze, touch) and outputting results as ⟨individual, group, interaction⟩ triplets.

**Limitations of Prior Work**: Existing methods (e.g., NVI-DEHR) exhibit two critical deficiencies: (1) they directly detect social groups without explicitly modeling inter-personal relationships, leading to ambiguous group predictions in scenarios such as gaze interactions across large spatial distances; and (2) they represent each individual as a holistic embedding, ignoring body part information, which makes it difficult to distinguish semantically similar yet distinct interactions (e.g., "mutual gaze" vs. "gaze following," "waving" vs. "pointing").

**Key Challenge**: Distinguishing fine-grained social interactions (e.g., gaze direction, gesture type) heavily relies on local body part cues, yet existing methods rely solely on global person representations. Group membership should emerge naturally from individual behaviors and inter-personal relationships rather than being predicted directly.

**Goal**: To design a framework that infers groups and interactions bottom-up from fine-grained body part cues through inter-personal relational reasoning.

**Key Insight**: Pose estimation is introduced as privileged information to guide part-aware learning; pose annotations are used only during training and are not required at inference.

**Core Idea**: Detect individuals → enhance individual embeddings with part features → infer groups via inter-individual similarity-based association → classify fine-grained interactions.

## Method

### Overall Architecture

The framework is built upon a DETR-based detection pipeline comprising four core modules: a feature extractor, an individual decoder, an individual embedding enhancer, and a group decoder with a similarity-based association module.

### Key Designs

1. **Individual Embedding Enhancer**: Each detected individual is decomposed into $P$ body parts. → $P$ learnable linear projections generate part queries: $\mathbf{Q}_P = \mathbf{E}_I \cdot [\mathbf{W}_1, \dots, \mathbf{W}_P] \in \mathbb{R}^{N_I \times P \times D}$. → Part queries are refined via self-attention (across parts) and cross-attention (with image feature maps). → Part embeddings are concatenated with individual embeddings and projected via fusion: $\mathbf{E}_A = [\mathbf{E}_I, \mathbf{E}_P^1, \dots, \mathbf{E}_P^P] \cdot \mathbf{W}_{\text{fuse}}$. → Unlike approaches that rely on external pose estimators, this method requires no additional inputs at inference.

2. **Pose-Guided Pseudo-Supervision**: ViTPose is used to extract keypoints as privileged information during training. → A square window is defined for each keypoint (sized proportionally to the individual bounding box: $s_i = \alpha \cdot \max(w_i, h_i)$), yielding binary masks $M_i^p$. → An MSE loss constrains part query attention maps to align with the corresponding masks: $\mathcal{L}_{\text{part}} = \frac{1}{N_I P} \sum_{i,p} \|A_i^p - M_i^p\|_2^2$. → 13 keypoints are used (excluding 4 facial keypoints to avoid spatial overlap).

3. **Bottom-Up Group Reasoning**: The group decoder uses learnable group queries that attend jointly to image features and part-aware individual embeddings. → It outputs group bounding box coordinates and multi-label interaction classification scores. → In contrast to prior methods that predict group boxes directly from group queries, the proposed approach has group queries aggregate information from relevant individuals to infer group composition.

4. **Similarity-Based Association**: A similarity matrix is computed between group and individual embeddings: $\mathbf{S} = \text{MLP}(\mathbf{E}_G) \cdot \text{MLP}(\mathbf{E}_I)^T$. → For each group, the individuals with the highest similarity scores are selected as members. → Association is trained with a BCE loss: $\mathcal{L}_{\text{assn}}$. → This design allows the number of individual and group queries to differ, unlike prior methods that require them to be equal.

### Loss & Training

The total loss is a weighted sum of five terms:
$$\mathcal{L} = \lambda_i \mathcal{L}_{\text{ind}} + \lambda_c \mathcal{L}_{\text{cls}} + \lambda_l \mathcal{L}_{\text{loc}} + \lambda_p \mathcal{L}_{\text{part}} + \lambda_a \mathcal{L}_{\text{assn}}$$

- $\mathcal{L}_{\text{ind}}$: Individual objectness, using Focal Loss
- $\mathcal{L}_{\text{cls}}$: Multi-label interaction classification, using Asymmetric Loss (ASL)
- $\mathcal{L}_{\text{loc}}$: Bounding box localization, $\ell_1$ + GIoU
- $\mathcal{L}_{\text{part}}$: Part attention supervision, MSE
- $\mathcal{L}_{\text{assn}}$: Group–individual association, BCE
- Hungarian matching is used for prediction-to-ground-truth assignment.

## Key Experimental Results

### Main Results (NVI Dataset)

| Method | val mR@25 | val mR@50 | val mR@100 | val AR | test AR |
|------|:-:|:-:|:-:|:-:|:-:|
| m-QPIC | 56.89 | 69.52 | 78.36 | 68.26 | 70.32 |
| m-CDN | 55.57 | 71.06 | 78.81 | 68.48 | 71.52 |
| m-GEN-VLKT | 50.59 | 70.87 | 80.08 | 67.18 | 71.72 |
| NVI-DEHR | 54.85 | 73.42 | 85.33 | 71.20 | 74.67 |
| **Ours** | **59.43** | **76.62** | **87.43** | **74.49** | **78.52** |

### Café Dataset

| Method | Group mAP₁.₀ (view) | Group mAP₀.₅ (view) | Outlier mIoU (view) |
|------|:-:|:-:|:-:|
| Café-base | 14.36 | 37.52 | 63.70 |
| **Ours** | **18.23** | **46.88** | **67.62** |

### Ablation Study

| Configuration | mR@25 | AR |
|------|:-:|:-:|
| Full model | 59.43 | 74.49 |
| w/o Enhancer | 55.20 | 72.17 |
| w/o Similarity Association | 55.95 | 72.75 |
| w/o Both | 56.29 | 70.86 |
| w/o $\mathcal{L}_{\text{assn}}$ | 30.38 | **48.32** (large drop) |
| w/o $\mathcal{L}_{\text{part}}$ | 54.32 | 73.58 |

### Comparison with MLLMs

| Method | mR@25 | AR |
|------|:-:|:-:|
| **Ours** | **63.59** | **78.52** |
| LLaVA (with GT group boxes) | 21.09 | 37.14 |
| LLaVA-LoRA fine-tuned | 17.40 | 33.81 |

### Key Findings

- **The embedding enhancer and similarity association each contribute approximately 2 AR points**, and their combined effect exceeds the sum of individual contributions.
- **The association loss $\mathcal{L}_{\text{assn}}$ is critical**: removing it causes AR to drop sharply from 74.49 to 48.32.
- **13 keypoints yields the best performance**: excluding the 4 redundant facial keypoints is beneficial; using all 17 keypoints slightly degrades results.
- **Pose guidance outperforms CLIP guidance**: the spatial precision provided by ViTPose is superior to VLM-based contrastive representations.
- **LLaVA falls far short of the proposed method even when given GT group boxes**, demonstrating a substantial gap between general-purpose multimodal models and specialized fine-grained social reasoning.
- **The proposed method surpasses prior work on Café without temporal modeling**, validating the generality of part-aware representations and bottom-up reasoning.

## Highlights & Insights

- Using pose estimation as "privileged information" exclusively during training is an elegant design choice—it provides fine-grained supervisory signals without incurring any additional inference cost.
- Bottom-up group reasoning aligns with the intrinsic nature of social interaction: groups emerge from inter-personal relationships rather than being independently predicted entities.
- Attention visualizations clearly demonstrate how the group decoder focuses on facial regions for expression-based interactions and on hand regions for handshake interactions.
- The similarity matrix association scheme is more flexible than guided embedding approaches, as it permits different numbers of individual and group queries.

## Limitations & Future Work

- Temporal information is not currently exploited; incorporating temporal modeling in video settings could yield further improvements.
- The number of parts $P=13$ is fixed; a dynamic part discovery mechanism warrants exploration.
- Evaluation is limited to the NVI and Café datasets; assessment on larger-scale or more diverse scenarios is lacking.
- Robustness analysis under heavy occlusion (where some body parts are not visible) is insufficient.
- The cascaded structure of the individual and group decoders may introduce error accumulation.

## Related Work & Insights

- NVI-DEHR models high-order relations via hypergraphs but neglects basic inter-personal relationships and body part information; the proposed method directly addresses both deficiencies.
- HOI-DET methods such as QPIC and CDN are adapted as baselines, but NVI-DET requires group-level reasoning, which HOI-DET does not address.
- The concept of privileged information learning originates from Vapnik and is elegantly applied here for pose supervision.
- Pose guidance has been explored in HOI-DET (e.g., Wu et al., Lei et al.), but this work is the first to apply it to social interaction detection in a training-only manner.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of part-aware features and bottom-up reasoning is novel in the context of social interaction detection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations, thorough comparisons with HOI/MLLM baselines, and rich visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, precise problem formulation, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Makes a substantive contribution to fine-grained social understanding with broadly applicable design principles.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](../../AAAI2026/human_understanding/clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)
- [\[NeurIPS 2025\] Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness](some_optimizers_are_more_equal_understanding_the_role_of_optimizers_in_group_fai.md)
- [\[NeurIPS 2025\] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning](k-decore_facilitating_knowledge_transfer_in_continual_structured_knowledge_reaso.md)
- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](../../ICCV2025/human_understanding/dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[NeurIPS 2025\] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion](hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)

<!-- RELATED:END -->
