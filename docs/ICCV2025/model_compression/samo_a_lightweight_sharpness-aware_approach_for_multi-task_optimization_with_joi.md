---
title: >-
  [Paper Note] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation
description: >-
  [ICCV 2025][Model Compression][Multi-task learning] This paper proposes SAMO, a lightweight sharpness-aware multi-task optimization method that mitigates task gradient conflicts via joint global-local perturbation…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Multi-task learning"
  - "sharpness-aware minimization"
  - "gradient conflict"
  - "zeroth-order gradient estimation"
  - "layer-wise normalization"
date: 2026-05-08
content_hash: e1a566a6f75839c8
---

# SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation

**Conference**: ICCV 2025
**arXiv**: [2507.07883](https://arxiv.org/abs/2507.07883)
**Code**: [GitHub](https://github.com/OptMN-Lab/SAMO)
**Area**: Model Compression
**Keywords**: Multi-task learning, sharpness-aware minimization, gradient conflict, zeroth-order gradient estimation, layer-wise normalization

## TL;DR

This paper proposes SAMO, a lightweight sharpness-aware multi-task optimization method that mitigates task gradient conflicts via joint global-local perturbation, while substantially reducing computational overhead through zeroth-order gradient approximation and layer-wise normalization.

## Background & Motivation

Multi-task learning (MTL) aims to train a single model to simultaneously learn multiple tasks, leveraging shared knowledge to improve data efficiency and generalization. However, the central challenge in MTL is **task conflict**: gradients from different tasks may contradict each other in direction or magnitude, causing naive gradient averaging to be dominated by a single task and thereby degrading overall performance.

Existing approaches fall into two broad categories: (1) **gradient manipulation methods** (e.g., MGDA, CAGrad, FairGrad), which adjust gradient directions or weights to find a compromise update direction; and (2) **architectural design methods** (e.g., MoE, soft parameter sharing), which reduce conflict through model structure. Most of these methods, however, overlook the role of loss landscape geometry.

**Sharpness-Aware Minimization (SAM)** is widely applied in single-task settings, improving generalization by simultaneously minimizing the loss value and the "sharpness" of the loss landscape. Through empirical analysis, the authors identify a key insight: **SAM effectively alleviates task conflicts in MTL**. Specifically:

- SAM drives the model toward flatter regions, where changes in one task objective do not significantly affect others.
- SAM substantially increases the cosine similarity between task gradients (from negative to positive values).
- Sharpness metrics $\lambda_{\max}$ and $\lambda_{\max}/\lambda_5$ of the loss landscape both decrease markedly.

Nevertheless, integrating SAM into MTL faces two major challenges: (1) **both global information (averaged gradients) and local information (per-task gradients) are beneficial for SAM**, yet how to effectively combine them remains unclear—G-SAM and L-SAM each exhibit advantages depending on the method and dataset; (2) computing per-task gradients for local perturbation requires $K$ additional backward passes, incurring **substantial computational cost**. The sole prior work, F-MTL, applies SAM to each individual task but introduces a $K$-fold increase in backward passes and doubles memory requirements.

## Method

### Overall Architecture

SAMO augments standard MTL gradient manipulation methods with a lightweight sharpness-aware module. The core idea is to first compute a joint global-local perturbation, evaluate gradients at the perturbed parameters, and then feed the perturbed gradients into any existing gradient manipulation method (e.g., FairGrad). The overall pipeline proceeds as: compute averaged loss gradient → approximate per-task local gradients → construct joint global-local perturbation → compute new gradients at perturbed parameters → pass into MTL method to obtain the update direction.

### Key Designs

1. **Joint Global-Local Perturbation**: SAMO defines the perturbation for each task $i$ as a weighted average of the global gradient and the local gradient:

$$\hat{\epsilon}_i(\theta) = \rho \frac{\alpha \nabla_\theta l_0(\theta) + (1-\alpha) \nabla_\theta l_i(\theta)}{\|\alpha \nabla_\theta l_0(\theta) + (1-\alpha) \nabla_\theta l_i(\theta)\|}$$

where $\alpha \in [0,1]$ balances global and local information. The global perturbation captures positive transfer across tasks, while the local perturbation retains task-specific characteristics. Each task's perturbation direction thus accounts for both shared patterns and its own loss landscape. The design motivation stems from the empirical observation that G-SAM and L-SAM each have advantages, and their joint use achieves a better balance.

2. **Zeroth-Order Gradient Approximation (SPSA)**: To avoid computing $K$ backward passes for each task, SAMO employs **Simultaneous Perturbation Stochastic Approximation (SPSA)**, which estimates local gradients using only forward passes:

$$\hat{\nabla}_\theta l_i(\theta) \approx \frac{l_i(\theta + \mu z_i) - l_i(\theta - \mu z_i)}{2\mu} z_i$$

where $z_i$ is a random vector sampled from a standard Gaussian distribution and $\mu$ is a small perturbation factor. This approximation requires only $2K$ forward passes (in place of $K$ backward passes), and the forward pass cost $C_f$ is substantially lower than the backward pass cost $C_b$. The design is inspired by parameter-efficient fine-tuning (PEFT): the global perturbation serves as the "backbone" and the local perturbation acts as the "adapter."

3. **Layer-wise Normalization Strategy**: The variance of zeroth-order gradient estimates can be large, leading to training instability. SAMO proposes layer-wise normalization, which rescales the estimated local gradient at each layer to match the magnitude of the global gradient:

$$\hat{\nabla}_\theta l_i(\theta^d) \leftarrow \hat{\nabla}_\theta l_i(\theta^d) \frac{\|\nabla_\theta l_0(\theta^d)\|}{\|\hat{\nabla}_\theta l_i(\theta^d)\|}$$

where $\theta^d$ denotes the parameters of layer $d$. This preserves the directional information of the zeroth-order estimate while aligning its magnitude with that of the exact gradient, preventing optimization instability caused by variance fluctuations.

### Loss & Training

SAMO introduces no additional loss functions; instead, it operates as a **plug-and-play module** that can be integrated into any existing MTL method, reusing the original training configuration entirely. Compared to F-MTL, SAMO's additional computational overhead is only $C_b + 2KC_f$ (one backward pass plus $2K$ forward passes), whereas F-MTL requires $KC_b + C_{gm}$ ($K$ backward passes plus the cost of gradient manipulation). Since $C_f \ll C_b$, SAMO's runtime overhead is comparable to using global SAM alone (G-SAM).

For hyperparameters: the perturbation radius $\rho$ is searched over $\{0.01, 0.05, 0.1, 0.5\}$; the global-local weight $\alpha$ is selected from $\{0.3, 0.5, 0.7\}$; and the zeroth-order perturbation factor $\mu$ is set to $0.01$.

## Key Experimental Results

### Main Results

**Cityscapes (2 tasks: semantic segmentation + depth estimation)**

| Method | mIoU ↑ | Pix Acc ↑ | Abs Err ↓ | Rel Err ↓ | Δm% ↓ |
|---|---|---|---|---|---|
| STL | 74.01 | 93.16 | 0.0125 | 27.77 | — |
| FairGrad | 74.10 | 93.03 | 0.0135 | 29.92 | 3.90 |
| F-MTL (best) | 73.77 | 93.12 | 0.0129 | 27.44 | 0.67 |
| **SAMO-FairGrad** | **74.37** | **93.14** | **0.0129** | **26.30** | **-0.62** |

**NYU-v2 (3 tasks: segmentation + depth + surface normals)**

| Method | mIoU ↑ | Abs Err ↓ | Angle Dist ↓ | Δm% ↓ |
|---|---|---|---|---|
| FairGrad | 38.80 | 0.5572 | 24.55 | -4.96 |
| F-MTL (best) | 40.42 | 0.5389 | 25.03 | -4.77 |
| **SAMO-FairGrad** | **39.05** | **0.5359** | **24.43** | **-6.55** |

### Ablation Study

| Configuration | Cityscapes Δm% | NYU-v2 Δm% | CelebA Δm% | Note |
|---|---|---|---|---|
| G-SAM-MGDA | 7.51 | -0.23 | 11.78 | Global information only |
| L-SAM-MGDA | 11.94 | 0.01 | 8.47 | Local information only |
| SAMO-MGDA | 4.30 | -2.19 | 9.59 | Joint global-local (Ours) |
| G-SAM-FairGrad | 0.93 | -5.70 | 0.41 | Global information only |
| L-SAM-FairGrad | 1.01 | -5.42 | -0.42 | Local information only |
| SAMO-FairGrad | **-0.62** | **-6.55** | **-0.74** | Joint global-local (Ours) |

### Key Findings

- SAMO consistently improves all three baseline methods (LS, MGDA, FairGrad) across all datasets.
- The joint global-local perturbation consistently outperforms using either global or local perturbation alone.
- Consistent improvements are maintained on CelebA (40 tasks) and QM9 (11 tasks), demonstrating scalability.
- SAMO remains effective in the multi-input setting of Office-Home, confirming that the method is not restricted to shared-input configurations.
- Runtime comparison: SAMO incurs only 2–6% additional time over baselines on Cityscapes/NYU-v2, far below the 80%+ overhead of F-MTL.

## Highlights & Insights

- The **mechanistic discovery that SAM mitigates task conflicts** is a significant contribution: through Hessian spectral analysis and cosine similarity visualization, the paper quantitatively verifies that SAM's guidance toward flat regions is equivalent to alleviating task conflict.
- The combination of **zeroth-order gradient approximation and layer-wise normalization** is elegant—directional information is preserved while variance is controlled.
- The **plug-and-play nature** of the method enables integration with any existing gradient manipulation approach, offering strong practical value.
- The visualization analysis on a two-dimensional synthetic problem intuitively illustrates how SAM alters optimization trajectories.

## Limitations & Future Work

- Zeroth-order gradient estimation is inherently approximate, with limited directional accuracy in high-dimensional spaces; single-sample estimation may introduce substantial noise.
- $\alpha$ is treated as a fixed hyperparameter requiring manual tuning; adaptive adjustment strategies merit exploration.
- Experiments do not cover large-scale pretrained models (e.g., ViT-Large); the effectiveness of zeroth-order estimation at larger parameter scales requires further validation.
- Combinations with advanced SAM variants (e.g., ASAM, LookSAM) remain unexplored.

## Related Work & Insights

- **F-MTL** (Phan et al.) is the most closely related prior work, applying SAM directly to each individual task, but at prohibitive computational cost.
- **SPSA gradient estimation** originates from the stochastic optimization literature and is applied here for the first time to approximate local perturbations in an MTL setting.
- Insight: Zeroth-order optimization has found applications beyond MTL (e.g., prompt tuning), and SAMO's methodology may generalize to other scenarios requiring multi-objective optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The joint global-local perturbation paradigm for integrating SAM into MTL is novel, though the zeroth-order approximation itself is an existing technique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets covering segmentation, classification, regression, and multi-input scenarios, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical flow, thorough analysis, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Strong practical utility as a plug-and-play module; the lightweight design offers considerable engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/model_compression/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)
- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](local_dense_logit_relations_for_enhanced_knowledge_distillation.md)
- [\[ICCV 2025\] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)
- [\[CVPR 2026\] Towards Source-Aware Object Swapping with Initial Noise Perturbation](../../CVPR2026/model_compression/towards_source-aware_object_swapping_with_initial_noise_perturbation.md)

</div>

<!-- RELATED:END -->
