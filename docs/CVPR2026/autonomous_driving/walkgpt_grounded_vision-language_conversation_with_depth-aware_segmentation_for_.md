---
title: >-
  [Paper Note] WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation
description: >-
  [CVPR2026][Autonomous Driving][Vision-Language Models] WalkGPT is proposed as the first pixel-grounded large vision-language model for pedestrian accessibility navigation, unifying conversational reasoning, segmentation masks, and depth estimation within a single architecture, accompanied by the 41k-scale PAVE dataset.
tags:
  - CVPR2026
  - Autonomous Driving
  - Vision-Language Models
  - Pixel-Level Grounding
  - Depth-Aware Segmentation
  - Pedestrian Navigation
  - Accessibility Reasoning
  - VQA
date: 2026-05-08
content_hash: 686d27dd6699643f
---

# WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation

**Conference**: CVPR2026  
**arXiv**: [2603.10703](https://arxiv.org/abs/2603.10703)  
**Code**: [Project Page](https://arxiv.org/abs/2603.10703) (code and dataset released)  
**Area**: Autonomous Driving / Pedestrian Navigation / Accessibility Assistance  
**Keywords**: Vision-Language Models, Pixel-Level Grounding, Depth-Aware Segmentation, Pedestrian Navigation, Accessibility Reasoning, VQA

## TL;DR

WalkGPT is proposed as the first pixel-grounded large vision-language model for pedestrian accessibility navigation, unifying conversational reasoning, segmentation masks, and depth estimation within a single architecture, accompanied by the 41k-scale PAVE dataset.

## Background & Motivation

**Urgent safety needs in pedestrian navigation**: Urban pedestrian routes contain static and dynamic barriers such as stairs, uneven surfaces, parked vehicles, and temporary obstacles, posing serious risks to individuals with mobility impairments. Yet nearly all automated navigation systems target vehicles, leaving pedestrian-level navigation severely underserved.

**Existing LVLMs lack spatial reasoning**: While large vision-language models such as LLaVA and InstructBLIP can describe visual content, they lack explicit spatial reasoning capabilities and cannot infer geometric structure or depth relationships, rendering them unsuitable for pedestrian navigation.

**Spatial reasoning methods rely on user input**: Spatially aware models such as SpatialRGPT and DepthLM require user-provided visual anchors or markers to estimate depth, which is impractical in pedestrian navigation scenarios.

**Hallucination poses safety risks**: LVLMs are prone to describing objects absent from the scene; in pedestrian navigation, such hallucinations can lead to dangerous guidance.

**Grounded LVLMs lack depth information**: Grounding models such as GLAMM and LISA can generate 2D segmentation masks but lack depth information, preventing understanding of relative distances and spatial hierarchy, which limits accessibility navigation applications.

**Absence of large-scale pedestrian-perspective datasets**: No large-scale dataset simultaneously providing pedestrian-perspective QA annotations and spatial grounding annotations previously existed, constraining progress in this domain.

## Method

### Overall Architecture

WalkGPT adopts a unified architecture in which a single SAM ViT-H pixel encoder is shared for both text generation and mask prediction. Given a pedestrian-perspective image and a navigation query, the model produces structured responses comprising free-text reasoning, segmentation masks, and relative depth estimates. Four structured tokens are employed: `<p>` (object reference), `<assessment>` (accessibility assessment), `<SEG>` (segmentation prompt), and `<distance>` (distance description), explicitly coupling language generation with segmentation and depth reasoning.

### Key Design 1: Multi-Scale Query Projector (MSQP)

- Maps pixel encoder embeddings into image tokens within the language space for LLM input.
- Aggregates feature maps across multiple spatial scales (original resolution, 2× pooling, 4× pooling, global average).
- Extracts information at each scale using learnable query embeddings via two-layer cross-attention.
- Introduces a segmentation-aware gating function to emphasize structure- and edge-rich regions.
- Concatenates and projects multi-scale outputs into $Q=36$ fixed-length tokens $\mathbf{V}_{\text{proj}} \in \mathbb{R}^{B \times Q \times H}$.
- Advantage: Preserves local detail and global scene context compared to simple MLP projection.

### Key Design 2: Calibrated Text Projector (CTP) + Region Alignment Loss

- Maps the hidden states of LLM-generated `<SEG>` tokens from $H=4096$ dimensions to a $d_{\text{vis}}=256$-dimensional visual space.
- Each projected vector is expanded via an MLP with learnable biases into $K_{\text{bank}}$ calibration sub-embeddings, preserving fine-grained semantics.
- **Region Alignment Loss**: An InfoNCE contrastive learning loss that aligns each `<SEG>` token with its corresponding visual region while repelling unrelated regions.
    - Top-K attention selects the most salient image tokens as positive samples.
    - Tokens from other images serve as negative samples.
    - Prevents semantic information loss during the $H \to d_{\text{vis}}$ dimensionality reduction.

### Loss & Training

The total loss is a weighted sum of three terms:

$$\mathcal{L}_{\text{total}} = \alpha_1 \mathcal{L}_{\text{CE}} + \alpha_2 \mathcal{L}_{\text{seg}} + \alpha_3 \mathcal{L}_{\text{NCE}}$$

- $\mathcal{L}_{\text{CE}}$: Autoregressive cross-entropy (dialogue generation + depth text prediction).
- $\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}_{seg}}$ (segmentation mask prediction).
- $\mathcal{L}_{\text{NCE}}$: Region alignment contrastive loss.

Depth estimation is predicted in natural language form via the `<distance>` token within the autoregressive sequence, requiring neither a dedicated depth head nor dense depth supervision.

### PAVE Dataset

- 41k pedestrian-perspective image–question–answer triplets sourced from the SANPO real-world subset.
- Each frame includes RGB, semantic/instance segmentation masks, depth maps, and accessibility labels.
- Structured VQA pairs are generated using GPT-5-nano, combined with automated verification and human review.
- Training set: 85 video sessions (~8.5k frames); validation set: 6 sessions (~600 frames).

## Key Experimental Results

### Main Results: Grounded Navigation Conversation (Table 1)

| Model | CIDEr↑ | METEOR↑ | mIoU↑ | Depth Acc.↑ | AbsRel↓ |
|------|--------|---------|-------|-------------|---------|
| GLAMM-FT | 37.96 | 39.12 | 18.23 | 38.95 | 77.05 |
| PixelLM-FT | 37.49 | 38.02 | 18.10 | 39.00 | 74.61 |
| Sa2VA-FT | 38.82 | 39.66 | 16.10 | 40.54 | 73.82 |
| **WalkGPT (7B)** | **41.97** | 42.36 | 19.95 | 41.97 | 67.88 |
| **WalkGPT (13B)** | 41.17 | **43.01** | **20.16** | **48.95** | **70.66** |

- All zero-shot baselines fail entirely and produce no depth estimates.
- Fine-tuned baselines show improvement but remain insufficient.
- WalkGPT 13B surpasses PixelLM-FT by more than 10% on mIoU (20.16 vs. 18.10) and by more than 25% on depth accuracy (48.95 vs. 39.00).

### RES Benchmark (Table 2)

| Model | refCOCO val | refCOCO+ val | refCOCOg val |
|------|------------|-------------|-------------|
| LISA | 74.1 | 62.4 | 66.4 |
| PixelLM | 73.0 | 66.3 | 69.3 |
| **WalkGPT** | **76.2** | **70.0** | **72.6** |

WalkGPT outperforms LISA and PixelLM by 3–4% on the RES task, for which it was not specifically designed.

### Ablation Study (Table 5)

| Variant | METEOR↑ | mIoU↑ | Depth Acc.↑ |
|------|---------|-------|-------------|
| WalkGPT (Full) | 43.01 | 20.16 | 48.95 |
| w/o MSQP → MLP | 39.50 | 17.40 | 43.39 |
| w/o multi-scale aggregation | 41.60 | 19.30 | 44.70 |
| CTP → Linear | 40.70 | 18.60 | 47.98 |
| w/o $\mathcal{L}_{\text{NCE}}$ | 41.00 | 18.90 | 47.00 |
| w/o `<distance>` | 41.22 | 20.01 | 38.77 |

### Key Findings

- MSQP is the core component driving performance gains; replacing it with an MLP causes substantial drops across all three metrics.
- Multi-scale aggregation is critical for capturing spatial structure and depth cues.
- The `<distance>` token contributes most to depth prediction (removing it reduces Depth Acc. from 48.95 to 38.77) while having negligible impact on segmentation.
- The hallucination rate CHAIRi is only 18.49 (vs. 22.16 for LLaVA-1.5), and object recall is 83.66% (vs. 33.04% for LLaVA-1.5), demonstrating that pixel-level grounding effectively suppresses hallucinations.

## Highlights & Insights

- **Pioneering task definition**: WalkGPT is the first pixel-grounded LVLM for pedestrian accessibility navigation, introducing the novel Grounded Navigation Guide task.
- **Elegant unified architecture**: Conversational reasoning, segmentation, and depth estimation are accomplished within a single autoregressive process via structured tokens, without any dedicated depth head.
- **MSQP design**: Multi-scale gated cross-attention compresses high-resolution detail and global context into compact tokens, substantially outperforming MLP projection.
- **Effective region alignment loss**: Contrastive learning prevents semantic loss during dimensionality reduction, strengthening language–visual correspondence.
- **Dataset contribution**: PAVE fills the gap in pedestrian-perspective VQA datasets with spatial grounding annotations.

## Limitations & Future Work

- The model misclassifies strong reflective surfaces (e.g., road reflections on building curtain walls) as physical obstacles, as a single viewpoint cannot distinguish real geometry from visual artifacts.
- Motion blur, noisy surfaces, and severe class imbalance all degrade segmentation quality.
- The dataset is sourced exclusively from the SANPO real-world subset, leaving cross-domain generalization unverified.
- The absolute segmentation mIoU (~20%) is low, reflecting the inherent difficulty of PAVE scenes while also limiting practical utility.
- Depth estimates are output in discrete text form, with accuracy constrained by the granularity of language expressions.
- Training data is sampled at 100 frames per session from 91 sessions, potentially missing important scene variations.

## Related Work & Insights

- **Grounded LVLMs**: GLAMM, LISA, PixelLM, GSVA, OMG-LLaVA, and Sa2VA unify semantic reasoning with pixel-level grounding but do not address pedestrian navigation.
- **Spatial reasoning LVLMs**: SpatialRGPT, DepthLM, and SpatialVLM use depth maps or visual anchors but depend on user input.
- **Pedestrian navigation**: WalkNet, StreetViewAI, and related work focus on static detection or metadata-dependent approaches, lacking fine-grained grounding.
- **Visual segmentation**: U-Net, nnU-Net, and Swin-UNETR achieve mIoU below 21% on PAVE, confirming the inherent difficulty of the scenes.
- **RES**: Standard methods including MCN, VLT, CRIS, LAVT, and ReLA are surpassed by WalkGPT without task-specific training.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (pioneering task definition + unified architecture + new dataset)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-dimensional evaluation + comprehensive ablations, but limited dataset scale and cross-domain experiments)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, rigorous formulations, rich figures and tables)
- Value: ⭐⭐⭐⭐ (fills an important gap with significant societal impact, though practical utility is constrained by segmentation accuracy)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation](../../NeurIPS2025/autonomous_driving/leveraging_depth_and_language_for_open-vocabulary_domain-generalized_semantic_se.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[CVPR 2026\] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](nord_a_data-efficient_vision-language-action_model_that_drives_without_reasoning.md)
- [\[CVPR 2026\] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)
- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)

<!-- RELATED:END -->
