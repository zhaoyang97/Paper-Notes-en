---
title: >-
  [Paper Note] Rethinking Residual Distribution in Locate-then-Edit Model Editing
description: >-
  [NeurIPS 2025][Knowledge Editing][model editing] This paper reveals that the residual distribution mechanism in locate-then-edit model editing introduces weight deviation errors that grow with distribution distance, batch size, and sequential edit length. It proposes BLUE (Boundary Layer UpdatE), a strategy that updates only the first and last critical layers, achieving an average improvement of 35.59%.
tags:
  - NeurIPS 2025
  - Knowledge Editing
  - model editing
  - locate-then-edit
  - residual distribution
  - MEMIT
date: 2026-05-08
content_hash: 76078f0123fe7638
---

# Rethinking Residual Distribution in Locate-then-Edit Model Editing

**Conference**: NeurIPS 2025
**arXiv**: [2502.03748](https://arxiv.org/abs/2502.03748)
**Code**: [GitHub](https://github.com/xpq-tech/BLUE)
**Area**: Knowledge Editing
**Keywords**: model editing, locate-then-edit, residual distribution, knowledge editing, MEMIT

## TL;DR

This paper reveals that the residual distribution mechanism in locate-then-edit model editing introduces weight deviation errors that grow with distribution distance, batch size, and sequential edit length. It proposes BLUE (Boundary Layer UpdatE), a strategy that updates only the first and last critical layers, achieving an average improvement of 35.59%.

## Background & Motivation

- Model editing aims to efficiently update outdated or incorrect knowledge in LLMs without full retraining.
- **Locate-then-edit** is the dominant paradigm: first locate critical layers, then compute weight updates via least squares.
- Methods such as MEMIT compute residuals $\delta_i^L$ at the last critical layer and **uniformly distribute** them across all critical layers.
- This paper identifies a **counterintuitive failure mode**: the residual distribution mechanism itself **introduces weight deviation errors** that degrade editing precision.
- This is the first systematic theoretical and empirical analysis of the residual distribution mechanism in locate-then-edit methods.

## Method

### Overall Architecture

**Locate-then-Edit Review**:

1. FFN is treated as key-value memory: $\mathbf{m}^l = \mathbf{W}_{\text{out}}^l \sigma(\mathbf{W}_{\text{in}}^l \gamma(\mathbf{h}^{l-1} + \mathbf{a}^l))$
2. Knowledge update objective: $\mathbf{W}_1^l = \arg\min_{\mathbf{W}} \|\mathbf{W}\mathbf{K}_0^l - \mathbf{M}_0^l\|^2 + \|\mathbf{W}\mathbf{K}_1^l - \mathbf{M}_1^l\|^2$
3. Closed-form solution: $\Delta^l = \mathbf{R}^l {\mathbf{K}_1^l}^T (\mathbf{K}_0^l{\mathbf{K}_0^l}^T + \mathbf{K}_1^l{\mathbf{K}_1^l}^T)^{-1}$
4. Residual distribution: $\mathbf{R}^l = \frac{\mathbf{R}^L}{L - l + 1}$ (uniform distribution from the last critical layer)

**Problem Analysis**:

Residual distribution introduces three core issues:
- The contribution of distributed residuals to the editing objective decreases sharply as distribution distance increases.
- Distributed residuals are not optimal for individual layers.
- Weight deviation errors grow with batch size, sequential edit length, and distribution distance.

### Key Designs

**Empirical Analysis 1: Contribution of Distributed Residuals**

Contribution score is defined as: $s = \mathbb{P}_{\theta^*}(o^*|p) - \mathbb{P}_\theta(o^*|p)$

Experimental findings:
- Distributed residuals yield a contribution close to 1.0 only at the last critical layer.
- Contributions at all other layers fall below 0.7 and decrease progressively.
- Contributions at the first critical layer remain below 0.1 across all three LLMs.
- However, when residuals are **computed directly** per layer, contributions approach 1.0 at every layer.

**Empirical Analysis 2: Similarity Comparison**

The cosine similarity between distributed residuals and directly computed residuals also exhibits a monotonic decrease across layers. In single-layer editing experiments, directly computed residuals outperform distributed residuals in Efficacy by more than 3× on average.

**Theoretical Analysis (Theorem 4.1)**:

$$\|\Delta^{l^*} - \Delta^l\|_2 \leq \left(\|\mathbf{R}^{l^*} - \mathbf{R}^L\|_2 + (L-l)\|\mathbf{R}^L\|_2\right)\|\mathbf{Q}\|_2$$

The error upper bound grows with three factors:
1. $\|\mathbf{R}^{l^*} - \mathbf{R}^L\|_2$: residual deviation (increases with distribution distance)
2. $(L-l)$: distribution distance itself
3. $\|\mathbf{Q}\|_2$: grows with batch size

**Lemma 4.3** further establishes that in sequential batch editing, the error upper bound also grows with the sequential edit length $\|\mathbf{K}_p^l{\mathbf{K}_p^l}^T\|_2$.

### BLUE Strategy

**Core Finding**: Only two layers need to be updated to achieve the editing objective. Experimental observations show:
- After updating the first critical layer, the number of optimization steps required by subsequent layers drops sharply (by 84% on GPT-J and 55.6% on Llama3).
- After updating the first two layers, the third layer requires virtually no optimization (steps < 2.0).

**BLUE Design**:
- **Updates only the first and last critical layers**: the first critical layer (most affected by residual distribution) and the last critical layer (where residuals are computed).
- **Directly computes** residuals for each layer rather than distributing them.
- Retains the original residual computation mechanism at the last critical layer from the locate-then-edit framework.
- Applicable to all locate-then-edit methods using uniform residual distribution: MEMIT, RECT, PRUNE, and AlphaEdit.

### Loss & Training

Per-layer residual optimization: $\mathbf{m}_i^L = \mathbf{h}_i^L + \arg\min_{\delta_i^L} \frac{1}{P}\sum_{j=1}^P -\log\mathbb{P}_{\theta(\mathbf{h}_i^L += \delta_i^L)}[o^*|x_j \oplus p]$

Weight updates employ the standard least-squares closed-form solution; the only difference lies in the residual source (directly computed vs. distributed).

## Key Experimental Results

### Main Results: Sequential Batch Editing (Llama3-8B, CounterFact)

| Method | Efficacy↑ | Generalization↑ | Specificity↑ | Fluency↑ | Consistency↑ |
|--------|----------|----------------|-------------|---------|-------------|
| MEMIT | 65.65 | 64.65 | 51.56 | 437.43 | 6.58 |
| AlphaEdit | 98.90 | 94.22 | 67.88 | 622.49 | 32.40 |
| MEMIT_BLUE | **99.57** | **94.13** | **83.77** | **626.26** | 32.29 |
| AlphaEdit_BLUE | **99.93** | **97.25** | 75.24 | 624.90 | **33.79** |

### Main Results: Sequential Batch Editing (GPT-J 6B, CounterFact)

| Method | Efficacy↑ | Generalization↑ | Specificity↑ | Fluency↑ |
|--------|----------|----------------|-------------|---------|
| MEMIT | 98.55 | 95.50 | 63.64 | 546.28 |
| AlphaEdit | 99.75 | 96.38 | 75.48 | 618.50 |
| MEMIT_BLUE | **99.70** | **96.90** | **74.61** | **620.89** |
| AlphaEdit_BLUE | **99.77** | **97.13** | 75.23 | **621.07** |

### ZsRE Dataset Results (Llama3-8B)

| Method | Efficacy↑ | Generalization↑ | Specificity↑ |
|--------|----------|----------------|-------------|
| MEMIT | 34.62 | 31.28 | 18.49 |
| AlphaEdit | 94.47 | 91.13 | 32.55 |
| MEMIT_BLUE | **95.94** | **90.98** | 32.41 |
| AlphaEdit_BLUE | **95.77** | **91.73** | 31.96 |

### Ablation Study

**Optimization Steps per Layer**:

| Model | Optimization steps per layer [first to last critical layer] |
|-------|------|
| GPT2-XL [13–17] | [16.37, 8.43, 1.71, 0.32, 0.10] |
| GPT-J [3–8] | [10.47, 1.68, 0.11, 0.0, 0.0, 0.0] |
| Llama3 [4–8] | [25.0, 11.10, 0.63, 0.0, 0.0] |

The pattern is consistent: after updating the first layer, subsequent layers require virtually no additional optimization.

**Residual Deviation Across Layers**: $\|\mathbf{R}^{l^*} - \mathbf{R}^L\|_2$ increases monotonically with residual distribution distance, corroborating the theoretical analysis.

### Key Findings

- BLUE achieves an average improvement of 35.59% and consistently outperforms the original methods across all 12 experimental settings.
- BLUE not only improves editing effectiveness but also better preserves the general capabilities of LLMs, as evidenced by downstream task evaluation and representation shift analysis.
- BLUE reduces the number of updated layers, thereby improving editing efficiency.
- BLUE remains effective in long-text model editing scenarios.

## Highlights & Insights

1. **Identifying a Core Problem**: This is the first work to reveal that residual distribution — a widely adopted mechanism — is in fact detrimental, a counterintuitive yet significant finding.
2. **Dual Theoretical and Empirical Validation**: Theorem 4.1 and Lemma 4.3 provide theoretical error upper bounds that are fully consistent with experimental observations.
3. **Remarkably Simple Solution**: BLUE requires only replacing "multi-layer update + residual distribution" with "two-layer update + direct computation" — a minimal modification with substantial gains.
4. **Broad Applicability**: BLUE is a general enhancement strategy directly applicable to four methods: MEMIT, RECT, PRUNE, and AlphaEdit.
5. **Empirical Evidence that Two Layers Suffice**: The optimization step analysis provides a solid empirical basis for the design choice of updating only two layers.

## Limitations & Future Work

- The theoretical analysis derives error upper bounds rather than tight bounds; growth in the upper bound does not necessarily imply growth in the actual error, though experimental results support the trend.
- Directly computing residuals for each layer incurs slightly higher computational overhead than distribution, though in practice BLUE is more efficient overall due to updating only two layers.
- Critical layer selection still relies on causal tracing analysis and may not generalize to all model architectures.
- Evaluation is limited to the CounterFact and ZsRE datasets.
- The applicability of BLUE to variants employing non-uniform residual distribution requires further investigation.
- Whether the two updated layers should be selected dynamically rather than fixed as the boundary layers remains an open question.

## Related Work & Insights

- This work exposes a critical overlooked flaw in the locate-then-edit paradigm, directly improving upon classical methods such as MEMIT and AlphaEdit.
- The principle of "updating less yields better results" aligns with the minimal intervention philosophy in knowledge editing.
- The findings provide important guidance for the scalability of model editing under large-batch and long-sequence settings.
- The proposed strategy is compatible with complementary ideas from other editing methods such as GRACE and MEND.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic analysis of residual distribution flaws; the finding is profound, though the method itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across 3 LLMs, 2 datasets, 12 experimental settings, and 4 baseline methods.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is well-structured and progressive; theoretical derivations are clear; figures and tables are informative.
- Value: ⭐⭐⭐⭐ Directly advances the model editing field with significant and consistent improvements; BLUE is plug-and-play.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](../../ICLR2026/knowledge_editing/got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)
- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](../../ICLR2026/knowledge_editing/fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](../../ICLR2026/knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)

<!-- RELATED:END -->
