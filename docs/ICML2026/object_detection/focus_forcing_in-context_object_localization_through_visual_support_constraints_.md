---
title: >-
  [Paper Note] FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization
description: >-
  [ICML 2026][Object Detection][In-Context Object Localization] FOCUS forces VLMs to perform in-context object localization based on visual support examples—rather than semantic priors—through a two-stage training process…
tags:
  - "ICML 2026"
  - "Object Detection"
  - "In-Context Object Localization"
  - "Visual Support Constraints"
  - "Attention Optimization"
  - "GRPO"
  - "Category-Agnostic"
date: 2026-05-08
content_hash: ef7697ffb3edbf3c
---

# FOCUS: Forcing In-Context Object Localization through Visual Support Constraints and Policy Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.31145](https://arxiv.org/abs/2605.31145)  
**Code**: To be confirmed  
**Area**: Object Detection / Multimodal VLM / In-Context Learning  
**Keywords**: In-Context Object Localization, Visual Support Constraints, Attention Optimization, GRPO, Category-Agnostic

## TL;DR
FOCUS forces VLMs to perform in-context object localization based on visual support examples—rather than semantic priors—through a two-stage training process involving "complete removal of category names + attention mask optimization + GRPO IoU rewards." A 7B parameter model outperforms 72B models, proving that task-aligned inductive bias is more critical than simple scaling.

## Background & Motivation

**Background**: In-context Object Localization (ICOL) enables models to locate a target in a query image using only 1-K support examples (with bounding boxes) without retraining or predefined category vocabularies. This is essential for user-driven applications such as personalized visual search, image editing, and interactive tracking.

**Limitations of Prior Work**: (1) Current VLMs rely heavily on **category name bias** in ICOL; when query text contains category words, models often skip visual support to rely on semantic priors, leading to errors in multi-instance scenarios. (2) While IPLoc attempts to mitigate category dependence using pseudo-labels, it still utilizes real category names during inference, reintroducing bias via train-test mismatch. (3) Even with pseudo-label training, models **ignore fine-grained cues** such as bbox geometry and spatial relative positions, relying instead on coarse visual similarity or residual category correlations.

**Key Challenge**: To achieve true instance-specific localization (distinguishing multiple instances of the same class), models must be forced to utilize the geometry of visual supports over semantic priors. However, current training objectives only optimize final bbox accuracy without explicit constraints on attention allocation—the model's "path of least resistance" is to identify the category and randomly select an instance.

**Goal**: (1) Completely remove category names to make visual support the sole indicator. (2) Use attention loss to force query tokens to attend to the query image and bbox tokens. (3) Refine bbox alignment using GRPO and IoU rewards. (4) Demonstrate that task-aligned objectives outweigh model scaling.

**Key Insight**: Empirical findings (Section 3) reveal that vanilla VLMs allocate excessive attention to category tokens while providing insufficient attention to query images and bbox tokens. Even after removing categories (vanilla wo/c), attention remains diffuse rather than concentrated on the support region. This suggests the issue is an attention mismatch rather than a lack of information—supervising attention directly is more effective than structural modifications.

**Core Idea**: A two-stage approach—Stage 1: SFT with bounding box attention optimization (forcing high attention from query tokens to bbox tokens); Stage 2: GRPO with IoU reward fine-tuning (directly minimizing localization error).

## Method

### Overall Architecture

**Prompt Design**: Completely category-free, consisting only of task descriptions and interleaved (image, bbox) pairs, with the final image as the query. The model outputs `<answer>[xmin,ymin,xmax,ymax]</answer>`.

Input sequence: $\mathcal{C} = \langle \text{prompt}, (I_1, b_1), \dots, (I_{T-1}, b_{T-1}), I_T \rangle$ —— Note the absence of category information.

**Training Stages**:
1. Stage 1: SFT + bbox attention mask loss (forcing query tokens to pay high attention to bbox tokens).
2. Stage 2: GRPO + IoU reward fine-tuning.

### Key Designs

1.  **Category-free Prompt**:
    - **Function**: Eliminates semantic prior channels, making visual support the sole target indicator.
    - **Mechanism**: The prompt only describes the task ("locate the same object across frames") and the output format, containing no category names (e.g., "dog", "bowl"). Support images are paired with bboxes, while the model predicts the bbox for the query image.
    - **Design Motivation**: Previous methods like IPLoc suffer from train-test mismatch by using real categories during inference. This method removes category names from both training and testing to fundamentally break the semantic shortcut.

2.  **Bounding Box Attention Optimization**:
    - **Function**: Explicitly supervises the attention distribution of query image tokens, forcing focus on bbox tokens and the query image.
    - **Mechanism**: Aggregates attention across all layers and heads $A = \tfrac{1}{LH}\sum_{\ell, h} A^{(\ell, h)}$. Extracts the row $P \in \mathbb{R}^{N \times T}$ corresponding to query image tokens. Uses a binary mask $m$ to label bbox token positions. The attention loss encourages $P$ to have high attention values where $m=1$ (bbox tokens) and low values elsewhere.
    - **Design Motivation**: Since vanilla models exhibit diffuse attention even without category names, explicit supervision is required to make the model "look at support geometry." This is an attention engineering approach rather than an architectural change, yet yields significant benefits.

3.  **GRPO + IoU Reward Refinement**:
    - **Function**: Directly optimizes localization error (IoU) using reinforcement learning after SFT convergence.
    - **Mechanism**: Group Relative Policy Optimization (GRPO) samples multiple rollouts (bbox outputs) per prompt, ranking them to derive group advantages. The reward is defined as $R = \text{IoU}(\text{predicted}, \text{ground-truth})$. No critic network is required.
    - **Design Motivation**: Standard SFT loss is token-level cross-entropy, which is only indirectly related to bbox geometric alignment. GRPO directly optimizes IoU, providing task-aligned reinforcement learning that is more stable than PPO for structured outputs like coordinates.

## Key Experimental Results

### Main Results

| Model | Parameters | RefCOCO+ AP@50 | RefCOCO++ | Visual-Genome AP |
| :--- | :--- | :--- | :--- | :--- |
| IPLoc (Qwen2-VL-7B) | 7B | 38.4 | 32.7 | 41.2 |
| Qwen2-VL-72B | 72B | 42.1 | 36.5 | 45.6 |
| LLaVA-OV-72B | 72B | 41.7 | 36.0 | 44.9 |
| **FOCUS (Qwen2-VL-7B)** | **7B** | **48.6** | **43.2** | **52.7** |

The 7B FOCUS model significantly outperforms 72B general-purpose VLMs, proving that task-aligned objectives are more effective than 10x scaling.

### Ablation Study (RefCOCO+ AP@50)

| Configuration | AP@50 |
| :--- | :--- |
| Full FOCUS | 48.6 |
| − GRPO (Stage 1 SFT only) | 45.3 |
| − Attention Mask Loss | 40.8 |
| − Category Removal (Keep category) | 39.5 |
| Vanilla SFT baseline | 38.4 |

Attention mask loss is the largest contributor (+4.5). Category removal and GRPO add 1-3 points each, providing synergistic improvements.

### Key Findings
- **Task-aligned Objective > Scaling**: 7B + FOCUS outperforms 72B general VLMs by over 10 points.
- **Attention mask loss is the critical component**: Its +4.5 AP contribution is larger than GRPO alone (+3.3), confirming that the primary issue is attention mismatch.
- **Complete Category Removal vs. Pseudo-labeling**: Radical removal outperforms partial (pseudo-label) strategies due to improved train-test consistency.
- **GRPO Refinement**: Directly optimizing IoU after SFT provides a 3.3-point boost, addressing the discrepancy between cross-entropy and geometric alignment.

## Highlights & Insights
- **Inductive Bias vs. Scale**: This work serves as a demonstrative case where 7B beats 72B, encouraging research into task-specific inductive biases.
- **Radical Category Removal**: While previous works tried to "weaken" category dependence, this work "severs" it. Radical ablation often yields surprisingly effective results.
- **Attention Loss as a General Tool**: Supervising attention to guide "where the model should look" is a transferable concept for tasks like VQA and image-text matching.
- **GRPO for Structured Output**: Using GRPO for continuous coordinate outputs proves stable even with small batches, offering a robust paradigm for VLM localization RL.

## Limitations & Future Work
- Complete category removal may lose important priors in specific scenarios where category is a key differentiator.
- The attention mask is manually designed based on bbox token positions; it may be less accurate for long prompts or complex layouts.
- Evaluation is limited to single-object localization; extensions to multi-object or panoptic segmentation were not tested.
- Testing on novel domains (e.g., medical, satellite) is needed to verify the generalizability of visual support.
- Limited by the 7B base model; further improvements likely require larger bases or finer attention supervision.

## Related Work & Insights
- **vs. IPLoc (Doveh 2025)**: IPLoc suffers from train-test mismatch; FOCUS ensures consistency by removing categories throughout.
- **vs. GLIP / GroundingDINO**: Those are text-conditioned; FOCUS uses pure visual conditioning, offering stronger open-set capabilities.
- **Insight**: For any task requiring specific focus, explicit attention supervision and the complete removal of competing shortcuts (semantic bias) are highly effective debiasing strategies.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Long-Context Generalization with Sparse Attention](../../ICLR2026/object_detection/long-context_generalization_with_sparse_attention.md)
- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](../../CVPR2026/object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](../../CVPR2026/object_detection/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](../../ICCV2025/object_detection/visual-rft_visual_reinforcement_fine-tuning.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)

</div>

<!-- RELATED:END -->
