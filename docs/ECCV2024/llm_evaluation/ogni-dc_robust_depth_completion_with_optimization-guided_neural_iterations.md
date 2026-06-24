---
title: >-
  [Paper Note] OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations
description: >-
  [ECCV 2024][LLM Evaluation][Depth Completion] This paper proposes OGNI-DC, which achieves both SOTA accuracy and strong generalization in depth completion through an "Optimization-Guided Neural Iteration" (OGNI) framework, combining a ConvGRU for iterative depth gradient field refinement with a Differentiable Depth Integrator (DDI).
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Depth Completion"
  - "Differentiable Optimization"
  - "Iterative Refinement"
  - "Generalization"
  - "Depth Gradients"
date: 2026-05-08
content_hash: 7fa0ce066dd2f377
---

# OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations

**Conference**: ECCV 2024  
**arXiv**: [2406.11711](https://arxiv.org/abs/2406.11711)  
**Code**: [https://github.com/princeton-vl/OGNI-DC](https://github.com/princeton-vl/OGNI-DC)  
**Area**: LLM Evaluation  
**Keywords**: Depth Completion, Differentiable Optimization, Iterative Refinement, Generalization, Depth Gradients

## TL;DR

This paper proposes OGNI-DC, which achieves both SOTA accuracy and strong generalization in depth completion through an "Optimization-Guided Neural Iteration" (OGNI) framework, combining a ConvGRU for iterative depth gradient field refinement with a Differentiable Depth Integrator (DDI).

## Background & Motivation

The Depth Completion task aims to generate a dense depth map from an RGB image and a sparse depth map, which is widely used in autonomous driving, robotics, and augmented reality. Existing methods face a **dilemma between accuracy and robustness**:

**Traditional Optimization Methods** (e.g., Zhang et al.): Formulate a global optimization problem with hand-crafted energy terms. These exhibit good generalization but suffer from insufficient accuracy.

**Deep Learning Methods** (e.g., NLSPN, CFormer): Directly regress depth values. They offer high accuracy but often fail catastrophically under domain shifts or variations in sparsity.

Key Insight: Depth gradients (depth differences between adjacent pixels) are easier to infer from local windows than absolute depths, making them more generalizable. Meanwhile, enforcing explicit optimization constraints to align the predicted depth with sparse observations naturally adapts the model to different sparsity patterns.

## Method

### Overall Architecture

The pipeline of OGNI-DC consists of three main components:
1. **Backbone Feature Extraction**: Based on CompletionFormer, extracting 1/4 and full-resolution features from the concatenated RGB and sparse depth images.
2. **OGNI Intermediate Depth Prediction**: Refining the depth gradient field iteratively via ConvGRU + integrating it into a depth map with DDI, for a total of $T=5$ iterations.
3. **Upsampling and Enhancement**: Convex upsampling + DySPN spatial propagation network to output the full-resolution depth.

### Key Designs

1. **Predicting Depth Gradients instead of Depth Values**: The network predicts $\hat{\mathbf{G}} = \{\hat{\mathbf{G}}^x, \hat{\mathbf{G}}^y\} \in \mathbb{R}^{2 \times H/4 \times W/4}$, representing depth gradients in the x and y directions. Depth gradients can be inferred from local windows, which are easier to learn and generalize compared to global depth.

2. **Differentiable Depth Integrator (DDI)**: The core component that integrates depth gradients into a depth map. Formulated as a linear least-squares problem:

$$\hat{\mathbf{D}} = \arg\min_{\mathbf{D}} \left( E_G(\mathbf{D}, \hat{\mathbf{G}}) + \alpha \cdot E_O(\mathbf{D}, \mathbf{O}, \mathbf{M}) \right)$$

where $E_G$ constrains the depth differences to align with the predicted gradients, and $E_O$ constrains the depth values to be consistent with the sparse observations ($\alpha=5.0$). This is efficiently solved using the conjugate gradient method without explicitly storing the massive system matrix $\mathbf{A}^\top\mathbf{A}$.

The differentiability of DDI is achieved through the chain rule: during backpropagation, the same conjugate gradient solver is reused to compute $\partial\hat{\mathbf{D}}/\partial\hat{\mathbf{G}}$, which requires significantly lower memory overhead than directly tracing the entire solving process.

3. **ConvGRU Iterative Refinement**: Utilizing a RAFT-style ConvGRU to iteratively update depth gradients:

$$\Delta\hat{\mathbf{G}}, \mathbf{h}_t = \text{ConvGRU}(\hat{\mathbf{F}}^{1/4}, \mathbf{h}_{t-1}, \hat{\mathbf{D}}^{1/4}_{t-1}, \hat{\mathbf{G}}_{t-1})$$

Key point: Refinement and integration are **tightly coupled**—the ConvGRU receives the integrated depth from the previous DDI iteration as input, allowing it to perceive the consequences of its gradient outputs, thereby providing stronger guidance and regularization.

4. **Initialization Acceleration**: Since depth gradients change only minutely between iterations, DDI can use the solution from the previous iteration as the initial guess, reducing latency by up to 62.1%.

5. **Random Masking Data Augmentation**: During training, 0-100% of the known depth values are randomly dropped for 50% of the samples to enhance generalization across different sparsities.

### Loss & Training

Supervision is applied to the outputs of all $T$ iterations using a decay factor $\gamma=0.9$:

$$\mathcal{L}_\mathbf{D} = \sum_{t=1}^T \gamma^{T-t} \left( \|\hat{\mathbf{D}}_t - \mathbf{D}\|_2^2 + \|\hat{\mathbf{D}}_t - \mathbf{D}\|_1 + \|\hat{\mathbf{D}}^{up}_t - \mathbf{D}\|_2^2 + \|\hat{\mathbf{D}}^{up}_t - \mathbf{D}\|_1 \right)$$

$$\mathcal{L}_\mathbf{G} = \sum_{t=1}^T \gamma^{T-t} \|\hat{\mathbf{G}}_t - \mathbf{G}\|_1, \quad \mathcal{L} = \mathcal{L}_\mathbf{D} + \lambda \cdot \mathcal{L}_\mathbf{G}$$

A joint $L_1 + L_2$ loss is used for depth, and $L_1$ loss is used for gradients, with $\lambda=1.0$. The model is trained on a single RTX 3090 for 36 epochs (~3 days) on NYUv2, and on 8×L40 GPUs for 100 epochs (~1 week) on KITTI.

## Key Experimental Results

### Main Results — Zero-Shot Generalization

| Test Dataset | Metric | OGNI-DC | NLSPN | CFormer | VPP4DC | Gain |
|-----------|------|---------|-------|---------|--------|------|
| VOID1500 | MAE↓ | **0.175** | 0.298 | 0.261 | 0.253 | -30.8% vs VPP4DC |
| VOID500 | MAE↓ | **0.198** | 0.381 | 0.385 | 0.307 | -35.5% vs VPP4DC |
| VOID150 | MAE↓ | **0.261** | 0.492 | 0.487 | 0.397 | -34.2% vs VPP4DC |
| DDAD | RMSE↓ | **6.876** | 11.646 | 9.606 | 10.247 | -25.0% vs LRRU |

### In-Domain Performance (NYUv2 / KITTI)

| Dataset | Metric | OGNI-DC | CFormer | LRRU | BEV@DC |
|--------|------|---------|---------|------|--------|
| NYUv2 | RMSE↓ | **0.087m** | 0.090 | 0.091 | 0.089 |
| NYUv2 | REL↓ | **0.011** | 0.012 | 0.011 | 0.012 |
| KITTI | MAE↓ | **182.29mm** | 183.88 | 189.96 | 189.44 |
| KITTI | iRMSE↓ | **1.81** | 1.89 | 1.87 | 1.83 |

### Ablation Study

| Configuration | NYUv2 RMSE↓ | NYUv2 MAE↓ | KITTI RMSE↓ | Description |
|------|-------------|-----------|-------------|------|
| CFormer+DySPN baseline | 123.6 | 43.2 | 825.1 | Baseline |
| No DDI (直接预测深度) | 128.6 | 44.9 | 824.0 | No optimization layer |
| **OGNI-DC (Ours)** | **112.2** | **38.0** | **813.7** | Full model |
| 1 GRU iteration | 114.0 | 39.9 | 820.1 | Single iteration |
| 3 GRU iterations | 112.4 | 38.2 | 818.6 | 3 iterations |
| ConvRNN (替换 GRU) | 112.7 | 38.1 | 817.7 | No gating mechanism |
| DDI zeros init | - | - | - | 62.1% higher latency |
| DDI pre-filled init | - | - | - | 56.3% higher latency |

### Key Findings

- **DDI is the Core of Generalization**: Removing DDI degrades the NYUv2 MAE from 38.0mm to 44.9mm.
- **5 Iterations is the Accuracy-Speed Sweet Spot**: From 1 to 5 iterations, MAE drops from 39.9 to 38.0, with no further improvement observed at 7 iterations.
- **Single Model Generalization Across Sparsity**: On KITTI 16-Lines, MAE is reduced by 25.5% compared to SpAgNet (451.9 vs 606.9).
- The inference speed is slightly slower than the baseline (FPS drops by ~38%), but the improvements in accuracy and generalization make this trade-off worthwhile.

## Highlights & Insights

1. **The Combination of Depth Gradients and Optimization Layer is an Elegant Design**: Predicting local quantities (gradients) rather than global ones (depth), combined with explicit constraints (DDI), achieves both the learning accuracy of deep models and the robustness of traditional optimization methods.
2. **The Differentiable Implementation of DDI is Highly Ingenious**: The backward pass reuses the conjugate gradient solver, while the solution from the previous iteration serves as a warm-start to accelerate convergence.
3. **First to Introduce the DROID-SLAM-style Coupled Optimization-Iterative Refinement Paradigm into Single-View Tasks.**
4. The random masking data augmentation is simple yet effective, enabling the model to work robustly under unseen densities.

## Limitations & Future Work

1. **Extreme Sparsity Scenarios** (e.g., only 5 observed points) show slightly inferior performance compared to SpAgNet, as errors in depth gradient integration accumulate over large unobserved areas.
2. Inference speed incurs a certain overhead (DDI solving is time-consuming); faster solvers or approximation methods can be explored in the future.
3. The iterative refinement operates at 1/4 resolution, and full-resolution details rely on SPN, which may result in a loss of some edge information.
4. $\alpha=5.0$ in DDI is fixed for all scenes; introducing adaptive weights could potentially further improve performance.

## Related Work & Insights

- **DROID-SLAM / DPVO**: The source of the high-level idea of coupled optimization and iterative refinement, but OGNI-DC is the first to apply it to single-view tasks.
- **RAFT**: The source of inspiration for the ConvGRU iterative refinement architecture.
- **CompletionFormer**: Utilized as the backbone.
- **SPN Series (NLSPN, DySPN)**: Spatial propagation networks used for depth enhancement.
- Insight: The paradigm of local prediction + explicit optimization constraints can be extended to tasks such as normal estimation and optical flow.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Unique design concept combining depth gradient prediction with a differentiable integrator, elegantly uniting optimization and learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely thorough evaluation across four datasets, cross-domain/cross-sparsity generalizations, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear technical details and complete mathematical derivation of DDI, though the overall text is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐⭐ — Simultaneously addressing both accuracy and robustness is what practical systems need the most, with open-source code available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Deep Cost Ray Fusion for Sparse Depth Video Completion](deep_cost_ray_fusion_for_sparse_depth_video_completion.md)
- [\[ECCV 2024\] SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)
- [\[ECCV 2024\] VisFocus: Prompt-Guided Vision Encoders for OCR-Free Dense Document Understanding](visfocus_prompt-guided_vision_encoders_for_ocr-free_dense_document_understanding.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](../../NeurIPS2025/llm_evaluation/leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)
- [\[ICML 2026\] AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning](../../ICML2026/llm_evaluation/agzo_activation-guided_zeroth-order_optimization_for_llm_fine-tuning.md)

</div>

<!-- RELATED:END -->
