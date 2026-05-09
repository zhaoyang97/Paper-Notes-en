---
title: >-
  [Paper Note] DC-Merge: Improving Model Merging with Directional Consistency
description: >-
  [CVPR 2026 (Main Track)][Multimodal VLM][model merging] DC-Merge identifies that the key to effective model merging lies in maintaining **directional consistency in singular space** between the merged multi-task vector and the original single-task vectors. By combining singular value smoothing with shared orthogonal subspace projection, DC-Merge achieves state-of-the-art merging performance on both Vision and Vision-Language tasks.
tags:
  - CVPR 2026 (Main Track)
  - Multimodal VLM
  - model merging
  - task vector
  - singular value decomposition
  - directional consistency
  - LoRA
date: 2026-05-08
content_hash: 2d4cc6dffc4b8516
---

# DC-Merge: Improving Model Merging with Directional Consistency

**Conference**: CVPR 2026 (Main Track)  
**arXiv**: [2603.06242](https://arxiv.org/abs/2603.06242)  
**Code**: [https://github.com/Tobeginwith/DC-Merge](https://github.com/Tobeginwith/DC-Merge)  
**Area**: Model Merging / Multi-Task Learning  
**Keywords**: model merging, task vector, singular value decomposition, directional consistency, LoRA

## TL;DR
DC-Merge identifies that the key to effective model merging lies in maintaining **directional consistency in singular space** between the merged multi-task vector and the original single-task vectors. By combining singular value smoothing with shared orthogonal subspace projection, DC-Merge achieves state-of-the-art merging performance on both Vision and Vision-Language tasks.

## Background & Motivation

### State of the Field
Model merging aims to consolidate multiple task-specific fine-tuned models into a unified model that inherits knowledge from all tasks. Existing methods such as Task Arithmetic, TIES, and DARE achieve merging by performing weighted averaging or pruning on task vectors (fine-tuned parameters minus pre-trained parameters).

### Limitations of Prior Work

**Unbalanced energy distribution**: In the SVD decomposition of task vectors, a small number of large singular values dominate the total energy (e.g., the top 5% of singular values may account for over 90% of the energy), causing semantically important but low-energy components to be neglected during merging.

**Geometric directional inconsistency**: Task vectors from different tasks conflict with each other in the parameter space geometry, and naive merging distorts the directional structure of individual task vectors.

### Root Cause
Simple weighted averaging or pruning is too coarse for handling directional information in high-dimensional parameter spaces — these approaches cannot guarantee that the merged result maintains directional consistency with individual task vectors in the singular space.

### Core Idea
**Directional Consistency** is achieved through two steps: first, smooth the singular values of each task vector to balance the energy distribution; then project the energy-balanced task vectors onto a shared orthogonal subspace to align their geometric directions.

## Method

### Overall Architecture
Given $N$ task vectors $\{\boldsymbol{\tau}_i\}_{i=1}^N$, where $\boldsymbol{\tau}_i = \boldsymbol{\theta}_i - \boldsymbol{\theta}_0$, the pipeline consists of three stages:
1. **Singular Value Smoothing**: Apply SVD to each $\boldsymbol{\tau}_i$ and smooth the singular value distribution.
2. **Shared Orthogonal Subspace Projection**: Project the smoothed task vectors onto a shared subspace to align their directions.
3. **Aggregation & Back-projection**: Aggregate within the subspace and project back to the original parameter space.

### Key Designs

#### 1. Singular Value Smoothing
- **Function**: Decompose each task vector via SVD as $\boldsymbol{\tau}_i = \mathbf{U}_i \boldsymbol{\Sigma}_i \mathbf{V}_i^\top$, then apply a smoothing transformation to the singular value vector $\boldsymbol{\sigma}_i$.
- **Mechanism**: Apply a power transformation $\sigma_j \leftarrow \sigma_j^\alpha$ (with $\alpha < 1$, e.g., $\alpha = 0.5$) to compress large singular values and amplify small ones, resulting in a more uniform energy distribution.
- **Design Motivation**: Prevents a few dominant singular values from overwhelming the merging outcome, ensuring that semantically important but low-energy knowledge components are adequately represented.
- **Distinction from Prior Work**: Task Arithmetic performs direct averaging without considering singular value distribution; TIES applies only vector-level pruning without operating in the singular space.

#### 2. Shared Orthogonal Subspace Projection
- **Function**: Identify a shared orthogonal basis $\mathbf{Q}$ such that projecting all energy-smoothed task vectors onto this subspace minimizes reconstruction error relative to their original directions.
- **Mechanism**: Solve the optimization problem $\min_{\mathbf{Q}} \sum_{i=1}^N \|\tilde{\boldsymbol{\tau}}_i - \mathbf{Q}\mathbf{Q}^\top \tilde{\boldsymbol{\tau}}_i\|_F^2$, where $\tilde{\boldsymbol{\tau}}_i$ denotes the smoothed task vector. $\mathbf{Q}$ is obtained by performing SVD on the concatenated matrix and retaining the top $k$ singular vectors.
- **Design Motivation**: The singular bases $\mathbf{U}_i, \mathbf{V}_i$ of different task vectors point in different directions; direct averaging causes directional distortion. The shared subspace provides a unified coordinate system that aligns the geometric directions of all task vectors while minimizing reconstruction error.

#### 3. Aggregation & Back-projection
- **Function**: Perform weighted aggregation of the aligned task vectors within the subspace: $\hat{\boldsymbol{\tau}} = \lambda \sum_{i=1}^N \mathbf{Q}^\top \tilde{\boldsymbol{\tau}}_i$, then back-project to the original parameter space: $\boldsymbol{\theta}_{merge} = \boldsymbol{\theta}_0 + \mathbf{Q}\hat{\boldsymbol{\tau}}$.
- **Design Motivation**: Aggregating within the shared subspace inherently ensures directional consistency between the merged result and the individual task vectors.

### Loss & Training
DC-Merge is a **completely training-free** post-processing method — no additional data or fine-tuning is required. The only hyperparameters are the smoothing exponent $\alpha$ and the subspace dimensionality $k$, both selected via a small-scale validation set.

## Key Experimental Results

### Main Results: Vision Tasks (8-Task Merging, ViT-B/32)

| Method | Avg. Accuracy (%) | vs. Pretrained |
|------|---------------|-----------------|
| Pretrained | 48.3 | — |
| Task Arithmetic | 55.4 | +7.1 |
| TIES | 56.3 | +8.0 |
| DARE | 57.0 | +8.7 |
| Consensus | 57.8 | +9.5 |
| **DC-Merge** | **59.6** | **+11.3** |

### Ablation Study

| Configuration | Avg. Accuracy (%) | Note |
|------|---------------|------|
| Full DC-Merge | 59.6 | Complete method |
| w/o SVD Smoothing | 57.8 | −1.8% without smoothing |
| w/o Subspace Projection | 56.5 | −3.1% without projection |
| w/o Both (baseline) | 55.4 | Equivalent to Task Arithmetic |

### LoRA Merging Results

| Method | Vision-Language Avg (%) |
|------|------------------------|
| LoRA Arithmetic | 72.1 |
| DARE-LoRA | 73.5 |
| **DC-Merge-LoRA** | **75.8** |

### Key Findings
- Subspace projection is the most critical module (contributing +3.1%), while SVD smoothing contributes +1.8%; the two components are complementary.
- Performance is stable for $\alpha \in [0.3, 0.6]$, indicating low sensitivity to this hyperparameter.
- DC-Merge consistently outperforms baselines under both LoRA and full fine-tuning settings.
- Gains are more pronounced in Vision-Language scenarios (e.g., CLIP fine-tuning).

## Highlights & Insights
- **Understanding model merging from the perspective of singular-space directional consistency** — this viewpoint is more principled than "pruning conflicting parameters" or "weight averaging," and provides a theoretical foundation.
- **Elegance of SVD smoothing** — a single power transformation effectively balances the energy distribution with negligible computational overhead.
- **Generalizability of shared subspace projection** — the approach is not limited to task vector merging; it is applicable to any scenario requiring "merging multiple high-dimensional vectors while preserving their directions."
- **Training-free** — requires no additional data or computation; purely post-hoc and plug-and-play.

## Limitations & Future Work
- The subspace dimensionality $k$ requires selection via a validation set, which is inconvenient in scenarios without validation data.
- SVD decomposition may incur substantial computational cost for very large models (e.g., 70B+ LLMs).
- Validation is limited to ViT and CLIP families; extension to decoder-only LLMs has not been explored.
- When the number of tasks is large (>20), the shared subspace may be insufficient to simultaneously satisfy the directional constraints of all task vectors.
- Robustness to large quality disparities among task vectors (e.g., poorly fine-tuned tasks) is not discussed.

## Related Work & Insights
- **vs. Task Arithmetic**: Task Arithmetic naively averages task vectors without considering singular space structure. DC-Merge introduces directional constraints on top of this, yielding improvements of 4+ percentage points.
- **vs. TIES/DARE**: TIES and DARE handle conflicts via pruning or random dropping — a "subtractive" strategy. DC-Merge adopts a "transformative" strategy: rather than discarding parameters, it transforms them into a directionally consistent space.
- **vs. RegMean/Fisher Merging**: These methods require additional data for regularization, whereas DC-Merge does not.
- **Insight**: The singular value smoothing idea could be transferred to LoRA initialization — constraining singular value distributions to be more uniform during LoRA training may inherently improve compatibility with subsequent merging.

## Rating
- Novelty: ⭐⭐⭐⭐ Framing model merging through the lens of directional consistency is novel, though the two technical components (SVD smoothing, subspace projection) are individually not new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers Vision and VL scenarios, both full fine-tuning and LoRA settings, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; the progression from analysis to method to experiments is logically coherent.
- Value: ⭐⭐⭐⭐ Addresses a core challenge in model merging; the training-free property offers high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/multimodal_vlm/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[NeurIPS 2025\] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](../../NeurIPS2025/multimodal_vlm/robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)
- [\[CVPR 2026\] Self-Consistency for LLM-Based Motion Trajectory Generation and Verification](self-consistency_for_llm-based_motion_trajectory_generation_and_verification.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)

<!-- RELATED:END -->
