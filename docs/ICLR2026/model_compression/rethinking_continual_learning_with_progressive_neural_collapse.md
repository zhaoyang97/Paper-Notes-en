---
title: >-
  [Paper Note] Rethinking Continual Learning with Progressive Neural Collapse
description: >-
  [Model Compression] This paper proposes the ProNC framework, which replaces fixed pre-defined ETFs with a progressively expanding Equiangular Tight Frame (ETF) target to achieve a balance between maximal inter-class separation and minimal forgetting in continual learning.
tags:
  - Model Compression
date: 2026-05-08
content_hash: 06791aa00f11b9f5
---

# Rethinking Continual Learning with Progressive Neural Collapse

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2505.24254](https://arxiv.org/abs/2505.24254)
- **Code**: [GitHub](https://github.com/Continue-Edge-AI-Lab/ProNC)
- **Area**: Continual Learning / Model Compression
- **Keywords**: Continual Learning, Neural Collapse, ETF, Class-Incremental Learning, Knowledge Distillation

## TL;DR

This paper proposes the ProNC framework, which replaces fixed pre-defined ETFs with a progressively expanding Equiangular Tight Frame (ETF) target to achieve a balance between maximal inter-class separation and minimal forgetting in continual learning.

## Background & Motivation

### State of the Field
Continual Learning (CL) aims to enable models to learn new tasks without forgetting previously acquired knowledge, with catastrophic forgetting being the central challenge. Recent studies have identified the **Neural Collapse (NC)** phenomenon at the terminal phase of deep network training, wherein class feature prototypes geometrically converge to a Simplex ETF (Equiangular Tight Frame), achieving maximal equiangular inter-class separation.

### Limitations of Prior Work
Existing approaches (e.g., NCT) attempt to pre-define a globally fixed ETF as the training target in CL, but suffer from three key issues:

**Impracticality**: A pre-defined ETF requires prior knowledge of the total number of classes across all tasks, which is infeasible in real-world settings.

**Performance bottleneck**: When the total number of classes is large, the angular margin between ETF vertices diminishes, impairing discriminability in early training stages (as shown in Figure 1, accuracy degrades as $k$ increases).

**Violation of NC dynamics**: NC is an emergent phenomenon that evolves during training; randomly initializing an ETF tends to cause geometric misalignment.

### Root Cause
The number of ETF vertices should always equal the number of classes seen so far to preserve maximal inter-class separation. This necessitates a dynamic, progressive ETF expansion mechanism.

## Method

### Overall Architecture

ProNC (**Progressive Neural Collapse**) consists of two core steps — ETF initialization and ETF expansion — combined with an alignment loss and a distillation loss for CL training.

### 1. ETF Initialization (After Task 1)

After training on the first task, the ETF closest to the learned class feature means $\tilde{M}_{K_1}$ is constructed as:

$$
\mathbf{E}^* = \sqrt{\frac{K_1}{K_1-1}} \mathbf{W}\mathbf{V}^\top \left(\mathbf{I}_{K_1} - \frac{1}{K_1}\mathbf{1}_{K_1}\mathbf{1}_{K_1}^\top\right)
$$

where $\mathbf{W}\mathbf{\Sigma}\mathbf{V}^\top$ is the SVD decomposition of $\mathbf{U}'$. This ensures the initial ETF is aligned with the actually learned features, avoiding geometric misalignment from random initialization.

### 2. Progressive ETF Expansion (Upon New Tasks)

When task $t$ introduces $K_t - K_{t-1}$ new classes:

- **Step a**: The existing orthonormal basis $\mathbf{U}_{t-1} \in \mathbb{R}^{d \times K_{t-1}}$ is extended to $\mathbf{U}_t \in \mathbb{R}^{d \times K_t}$ via Gram-Schmidt orthogonalization, with newly added vectors orthogonal to existing basis vectors.
- **Step b**: $\mathbf{U}_t$ and $K_t$ are substituted into the ETF construction formula (Equation 1) to obtain the updated ETF target $\mathbf{E}_t$ with $K_t$ vertices.

Key property: the original orthonormal basis is preserved, minimizing drift of vertices corresponding to old classes.

### 3. Loss Function Design

For task $t \geq 2$, model training employs a weighted combination of three losses:

$$
\mathcal{L} = \mathcal{L}_{\text{ce}} + \lambda_1 \cdot \mathcal{L}_{\text{align}} + \lambda_2 \cdot \mathcal{L}_{\text{distill}}
$$

**(1) Alignment Loss**: Pushes learned features toward the corresponding ETF target vertices:

$$
\mathcal{L}_{\text{align}}(\boldsymbol{\mu}_{k,i}^t, \mathbf{e}_{k,t}) = \frac{1}{2}(\mathbf{e}_{k,t}^\top \boldsymbol{\mu}_{k,i}^t - 1)^2
$$

**(2) Distillation Loss**: Mitigates drift of old-class features caused by ETF expansion:

$$
\mathcal{L}_{\text{distill}}(\boldsymbol{\mu}_{k,i}^{(t-1)}, \boldsymbol{\mu}_{k,i}^{(t)}) = \frac{1}{2}((\boldsymbol{\mu}_{k,i}^{(t-1)})^\top \boldsymbol{\mu}_{k,i}^{(t)} - 1)^2
$$

### 4. Inference

A cosine-similarity-based nearest ETF vertex classifier replaces the linear classifier:

$$
\hat{y} = \arg\max_k \boldsymbol{\mu}_j^\top \mathbf{e}_k
$$

## Experiments

### Main Results

| Buffer | Method | Seq-CIFAR-10 (Class-IL) | Seq-CIFAR-100 (Class-IL) | Seq-TinyImageNet (Class-IL) |
|--------|--------|------------------------|--------------------------|----------------------------|
| 200 | ER | 44.79 | 21.78 | 8.49 |
| 200 | DER++ | 64.88 | 28.13 | 11.34 |
| 200 | STAR | 65.94 | 38.15 | 13.64 |
| 200 | NCT (Fixed ETF) | 51.59 | 26.38 | 10.95 |
| 200 | **ProNC (Ours)** | **72.70** | **44.32** | **20.11** |
| 500 | DER++ | 72.25 | 41.67 | 19.69 |
| 500 | STAR | 73.42 | 49.72 | 22.18 |
| 500 | **ProNC (Ours)** | **79.42** | **52.49** | **28.27** |

### Ablation Study

| Component | Seq-CIFAR-10 (FAA) | Seq-CIFAR-100 (FAA) | Seq-TinyImageNet (FAA) |
|-----------|-------------------|--------------------|-----------------------|
| Full ProNC | **72.70** | **44.32** | **20.11** |
| w/o Alignment Loss | 65.94 | 38.15 | 13.64 |
| w/o Distillation Loss | 69.82 | 41.76 | 17.53 |
| Fixed Global ETF (NCT) | 51.59 | 26.38 | 10.95 |

### Key Findings

1. **ProNC substantially outperforms all baselines across all datasets**, with Class-IL accuracy on TinyImageNet improved by over 6 percentage points.
2. **The alignment loss is the most critical component**; removing it degrades performance to the level of STAR.
3. **Progressive ETF expansion is far superior to a fixed ETF**; NCT's fixed global ETF is severely limited when the number of classes is large.
4. **Forgetting rate is significantly reduced**; ProNC's average forgetting rate is substantially lower than that of DER++ and STAR.

## Highlights & Insights

- No pre-defined global ETF is required; the initial ETF is adaptively extracted from the first task and progressively expanded.
- Theoretical guarantees (Theorem 1) ensure optimal alignment of the initial ETF.
- The ETF expansion strategy is based on orthonormal basis preservation, minimizing drift of old-class vertices.
- The framework is concise and flexible, serving as a plug-in feature regularizer compatible with any replay-based CL method.

## Limitations & Future Work

- The feature dimension $d$ must satisfy $d \geq K-1$; ETF expansion becomes constrained when the total number of classes approaches the feature dimensionality.
- The method still relies on a replay buffer; its effectiveness in purely replay-free settings remains unverified.
- SVD computation in ETF construction may incur additional overhead when the number of classes is very large.
- Validation is limited to ResNet-18; larger-scale models and real-world deployment scenarios are not explored.

## Related Work & Insights

- **Neural Collapse**: Papyan et al. (2020) observed that terminal-phase features converge to a Simplex ETF.
- **ETF-based CL**: NCT (Yang et al., 2023b) pre-defines a fixed global ETF; MNC3L (Dang et al., 2025) integrates contrastive learning.
- **Replay-based CL**: DER/DER++ (Buzzega et al., 2020), STAR (Eskandar et al., 2025).
- **Knowledge distillation-based CL**: iCaRL (Rebuffi et al., 2017), LODE (Liang & Li, 2023).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The progressive ETF expansion idea is original and theoretically grounded.
- **Technical Depth**: ⭐⭐⭐⭐ — Complete from theory to implementation; Theorem 1 provides rigorous mathematical guarantees.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 3 datasets and 2 CL scenarios with comprehensive ablations.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play feature regularization compatible with multiple CL frameworks.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)

<!-- RELATED:END -->
