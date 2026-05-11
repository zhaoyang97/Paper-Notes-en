---
title: >-
  [Paper Note] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection
description: >-
  [NeurIPS 2025 (Workshop: Foundation Models for the Brain and Body)][LLM Pretraining][Semi-supervised learning] This paper proposes a semi-supervised framework that uses mouse trajectories as weak supervision signals to p…
tags:
  - "NeurIPS 2025 (Workshop: Foundation Models for the Brain and Body)"
  - "LLM Pretraining"
  - "Semi-supervised learning"
  - "eye tracking"
  - "screen magnification"
  - "reading behavior classification"
  - "accessibility"
date: 2026-05-08
content_hash: 660aeb531f1c383b
---

# Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection

**Conference**: NeurIPS 2025 (Workshop: Foundation Models for the Brain and Body)  
**arXiv**: [2509.19574](https://arxiv.org/abs/2509.19574)  
**Code**: None  
**Area**: LLM Pretraining  
**Keywords**: Semi-supervised learning, eye tracking, screen magnification, reading behavior classification, accessibility

## TL;DR

This paper proposes a semi-supervised framework that uses mouse trajectories as weak supervision signals to pretrain gaze representations, followed by fine-tuning on labeled data to distinguish reading from scanning behavior. At inference time, only gaze signals are used, enabling hands-free assistive reading detection.

## Background & Motivation

**Background**: Screen magnification is a critical assistive tool for users with low vision, but the magnified viewport can display only a few words or lines at a time, requiring users to frequently scroll by dragging the mouse.

**Limitations of Prior Work**: Most gaze-based automatic scrolling control systems rely on hand-crafted heuristic rules with poor generalizability; gaze trajectories under magnification are fragmented and noisy, making it difficult to distinguish reading from scanning based on gaze alone.

**Key Challenge**: Accurate intent inference requires high-quality behavioral annotations, which are costly to obtain; the mouse is unavailable at inference time (since the goal is hands-free operation), yet mouse trajectories carry rich behavioral semantic information.

**Goal**: To robustly distinguish reading from scanning behavior at inference time using only gaze signals.

**Key Insight**: Mouse movement is used as a weak supervision target during pretraining to learn intention-aware gaze representations; raw gaze and compensated gaze are jointly modeled as two complementary views.

**Core Idea**: Mouse-guided semi-supervised pretraining combined with dual-stream cross-attention fusion of raw and compensated gaze.

## Method

### Overall Architecture

The framework consists of two stages:
- **Pretraining Stage (Pretext)**: The model predicts mouse velocity (2D regression) from unlabeled gaze data to learn behavior-aware gaze representations.
- **Fine-tuning Stage (Downstream)**: The pretrained encoder is transferred with the regression head replaced by a classification head, and fine-tuned on labeled data for binary reading/scanning classification.

Both stages share the same backbone: dual-stream cross-attention fusion followed by a Transformer encoder.

### Key Designs

1. **Dual-Stream Gaze Input (Raw + Compensated Gaze)**

    - **Function**: Both raw gaze coordinates (in the magnified viewport coordinate system) and compensated gaze coordinates (mapped back to the original screen coordinate system) are used as inputs.
    - **Design Motivation**: Raw gaze preserves fine-grained local eye movement dynamics, while compensated gaze restores global spatial continuity (line and paragraph alignment); the two views are complementary.
    - **Mechanism**: Each gaze stream is encoded by a three-layer 1D CNN (kernel=3, 64-dim), then fused via two cross-attention blocks (Q=g, K/V=c and Q=c, K/V=g) to capture complementary information.
    - **Novelty**: This is the first work to fuse raw and compensated gaze streams for reading behavior classification.

2. **Mouse-Guided Semi-Supervised Pretraining**

    - **Function**: During pretraining, the model learns to predict 2D mouse velocity from unlabeled gaze sequences.
    - **Design Motivation**: Mouse activity increases significantly during scanning (as users need to reposition the viewport), and mouse movement reflects active intentional decisions. This provides weak supervision without manual annotation.
    - **Mechanism**: A linear regression head with MSE loss is used to predict 2D mouse velocity; replaced by a classification head with cross-entropy loss during fine-tuning.
    - **Novelty**: Mouse signals are used only during training; inference relies entirely on gaze signals, enabling hands-free operation.

3. **Transformer Temporal Modeling**

    - The fused representations are passed through a three-layer Transformer encoder (64-dim, 4-head attention) to model temporal dependencies.
    - The input window is 0.2 seconds (24 steps at 120 Hz), with a sliding overlap window; labels correspond to the annotation at the last time point of each window.

### Loss & Training

- **Pretraining**: MSE loss for predicting 2D mouse velocity
- **Fine-tuning**: Weighted cross-entropy loss (to address class imbalance between reading and scanning)
- **Fine-tuning Strategy**: Partial fine-tuning (updating only the last three Transformer layers) vs. full fine-tuning (updating all parameters)
- Optimizer: Adam (lr=3e-4, weight_decay=0.01)
- Evaluation: Leave-one-subject-out cross-validation

## Key Experimental Results

### Dataset

Based on the dataset of Tang et al., containing synchronized gaze and mouse recordings from low-vision participants reading text documents and web pages under full-screen magnification. Gaze sampled at 120 Hz (Tobii Spectrum); mouse at 10 Hz.

### Main Results

#### Supervised Learning Results Under Different Input Configurations (Text Dataset)

| Input Type | Overall F1 | Reading F1 | Scanning F1 |
|---------|-----------|-----------|------------|
| Random baseline | 40.91 | 56.04 | 25.79 |
| Compensated only | 67.22 | 87.69 | 46.75 |
| Gaze only | 75.06 | 89.31 | 60.81 |
| **Gaze + Comp. (ours)** | **80.02** | **91.27** | **68.78** |
| Mouse only | 52.85 | 70.29 | 35.41 |
| Mouse + Gaze + Comp. | 83.64 | 91.17 | 76.10 |

#### Semi-Supervised vs. Supervised Learning Results

| Method | Text Overall | Text Reading | Text Scanning | Web Overall | Web Reading | Web Scanning |
|-----|-------------|-------------|--------------|------------|------------|-------------|
| Supervised | 80.02 | 91.27 | 68.78 | 62.49 | 60.27 | 64.59 |
| Semi-supervised (Partial) | 81.93 | 91.56 | 72.29 | 64.51 | 62.59 | 66.42 |
| **Semi-supervised (Full)** | **85.97** | **93.13** | **78.80** | **70.01** | **68.39** | **71.62** |

### Ablation Study

| Ablation Dimension | Result |
|---------|------|
| Raw gaze vs. compensated gaze vs. fusion | Fusion (80.02) > Raw (75.06) > Compensated (67.22) |
| Partial fine-tuning vs. full fine-tuning | Full (85.97) > Partial (81.93) > Supervised (80.02) |
| Contribution of mouse signal | Mouse alone is weak (52.85), but as a supervision signal improves performance to 85.97 |

### Key Findings

- Semi-supervised pretraining improves F1 by **6.0%** on the text dataset (80.02→85.97) and by **7.5%** on the more challenging web dataset (62.49→70.01).
- Even partial fine-tuning surpasses the fully supervised baseline, indicating that pretraining learns high-quality behavioral representations.
- The scanning class (minority class) benefits most: fully supervised 68.78→semi-supervised 78.80 (+10.02 F1).

## Highlights & Insights

- **Clever Use of Mouse as Weak Supervision**: The mouse is used only during training to guide representation learning and is completely absent at inference time—perfectly aligned with the hands-free requirement of assistive technology scenarios.
- **Dual-Stream Complementary Modeling**: The fusion of raw and compensated gaze is a well-motivated engineering design; the former preserves local detail while the latter restores global structure.
- **Clear Accessibility-Oriented Application**: The work directly targets screen magnification for low-vision users with clear practical value.

## Limitations & Future Work

- Experiments are conducted only under full-lens magnification; other magnification modes (e.g., partial magnification) are not evaluated.
- The dataset is limited in scale (leave-one-subject-out evaluation implies a small number of participants), and generalizability remains to be validated.
- The optimality of the 0.2-second window length is not explicitly discussed.
- No direct comparison with recent Transformer-based reading behavior classification methods (e.g., Yang et al. 2025) on the same data.
- Future work could extend the intent classifier into a real-time automatic scrolling controller.

## Related Work & Insights

- **Self-supervised Gaze Representation Learning**: Prior work has focused primarily on gaze estimation from eye images or coarse-grained behavior recognition from EOG signals; this paper is the first to apply semi-supervised learning to frame-level reading behavior classification.
- **Mouse–Gaze Alignment**: Previous work has mainly addressed attention analysis and reading depth prediction; this paper innovatively repurposes the mouse as a pretraining target.
- **Implications for Other Assistive Technologies**: Similar weakly supervised pretraining strategies could be generalized to other multimodal human–computer interaction scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The mouse-guided pretraining idea is concise and effective; the dual-stream fusion is a first application in this context
- Experimental Thoroughness: ⭐⭐⭐ Reasonably complete for a workshop paper, but limited in dataset scale and baseline comparisons
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated motivation
- Value: ⭐⭐⭐⭐ Practical accessibility-oriented scenario; semi-supervised strategy is broadly applicable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](../../AAAI2026/llm_pretraining/learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)
- [\[NeurIPS 2025\] Optimal Online Change Detection via Random Fourier Features](optimal_online_change_detection_via_random_fourier_features.md)
- [\[NeurIPS 2025\] Understanding and Enhancing Mask-Based Pretraining towards Universal Representations](understanding_and_enhancing_mask-based_pretraining_towards_universal_representat.md)
- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](../../ICLR2026/llm_pretraining/pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[NeurIPS 2025\] Efficient Pre-Training of LLMs via Topology-Aware Communication Alignment on More Than 9600 GPUs](efficient_pre-training_of_llms_via_topology-aware_communication_alignment_on_mor.md)

</div>

<!-- RELATED:END -->
