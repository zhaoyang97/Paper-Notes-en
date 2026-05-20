---
title: >-
  [Paper Note] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate
description: >-
  [ICLR 2026][Optimization][Scaling laws] Grounded in convex optimization theory, this paper proves that training loss in deep learning converges at a rate of $O(1/\sqrt{T})$ and that the optimal learning rate scales as $1…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Scaling laws"
  - "learning rate schedule"
  - "convex optimization"
  - "loss convergence"
  - "training planning"
date: 2026-05-08
content_hash: 53a75150a1ea4023
---

# Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate

**Conference**: ICLR 2026
**arXiv**: [2602.07145](https://arxiv.org/abs/2602.07145)  
**Code**: Not released  
**Area**: Optimization
**Keywords**: Scaling laws, learning rate schedule, convex optimization, loss convergence, training planning

## TL;DR
Grounded in convex optimization theory, this paper proves that training loss in deep learning converges at a rate of $O(1/\sqrt{T})$ and that the optimal learning rate scales as $1/\sqrt{T}$. The resulting scaling law is validated across models ranging from GPT-2 to 12.5B parameters ($R^2 \geq 0.978$), enabling learning rate extrapolation across an 80× range of training steps.

## Background & Motivation

**Background**: Scaling laws such as Chinchilla characterize the relationship between loss and data volume/model size, but the coupled dependence of loss on training steps and learning rate lacks a theoretical foundation. In practice, learning rate schedule choices (cosine, linear decay, WSD) are largely empirical.

**Limitations of Prior Work**: When the training budget (total steps $T$) changes, the optimal learning rate must be re-searched — a process that is both costly and uncertain. Existing empirical scaling laws lack theoretical grounding and cannot reliably extrapolate to new training configurations.

**Key Challenge**: Although deep learning involves non-convex optimization, the convergence behavior observed in practice resembles that of convex optimization. This implicit convexity has not yet been systematically theorized.

**Goal**: (a) Establish a unified scaling law relating loss, learning rate, and training steps; (b) enable learning rate transfer across training budgets.

**Key Insight**: The paper hypothesizes that deep learning exhibits weak convexity at a macroscopic level (convex dominance), and that convergence bounds derived from convex analysis can describe actual training behavior.

**Core Idea**: Training loss in deep learning follows $L(T) \sim L_\infty + C/\sqrt{T}$, with optimal learning rate $\eta^* = \eta_{\text{ref}}/\sqrt{T}$, where $\eta_{\text{ref}}$ can be determined from small-scale experiments and transferred directly.

## Method

### Overall Architecture
Starting from the convergence theorem for convex SGD, the paper derives conditions for "qualified learning rate schedules," then generalizes the theoretical form to deep learning by replacing theoretical constants with data-fitted parameters, ultimately yielding a predictable and transferable scaling law.

### Key Designs

1. **Convex SGD Convergence Bound**:

    - Function: Derive a last-iterate loss upper bound for SGD on convex objectives.
    - Mechanism: $\mathbb{E}[L(w_T)] \leq L^* + \frac{D^2}{2\sum \eta_t} + \frac{G^2 \sum \eta_t^2}{2\sum \eta_t} + \text{residual}$, where $D$ is the distance from initialization to the optimum and $G$ is a gradient bound.
    - Design Motivation: Unlike average-iterate bounds, the last-iterate bound better reflects practical training, where the final model checkpoint is used rather than a model average.

2. **Qualified Learning Rate Schedules**:

    - Function: Define which schedules achieve $O(1/\sqrt{T})$ convergence.
    - Mechanism: Linear decay, cosine decay, and WSD are all qualified. For each schedule, the optimal peak learning rate is derived as $\eta_{\text{peak}}^*(T) = D/(G\sqrt{cT})$, where $c$ is a constant factor.
    - Design Motivation: This unifies the theoretical analysis of different schedules — they are equivalent in the $O(\cdot)$ sense, differing only by constant factors.

3. **Generalization to Deep Learning**:

    - Function: Extend theoretical results to non-convex practical deep learning.
    - Mechanism: $\mathbb{E}[L(w_T)] \sim L_\infty + q_1^2/(T \cdot \eta_{\text{peak}}) + \eta_{\text{peak}} \cdot q_2^2$. The form mirrors the convex bound, but $L_\infty$, $q_1$, $q_2$ are determined by data fitting rather than theory. The optimal learning rate is $\eta_{\text{peak}}^* = q_1/(q_2\sqrt{T})$.
    - Design Motivation: The "Convex Dominance" hypothesis — although deep learning is non-convex, convexity dominates the optimization dynamics at the macroscopic level of the loss landscape.

4. **Learning Rate Transfer Rule**:

    - Function: Determine a reference optimal learning rate $\eta_{\text{ref}}$ from small-scale experiments and transfer it to large-scale training.
    - Mechanism: $\eta_{\text{peak}}^*(T) = \eta_{\text{ref}}/\sqrt{T}$, where $\eta_{\text{ref}} = \eta_{\text{peak}}^*(T_{\text{small}}) \cdot \sqrt{T_{\text{small}}}$. $\eta_{\text{ref}}$ is searched once at small scale, then directly computed for any $T$.
    - Design Motivation: Enables effective extrapolation over an 80× range of training steps (100 steps → 5000 steps).

## Key Experimental Results

### Model Fit Quality

| Model | Dataset | $R^2$ |
|-------|---------|-------|
| ResNet18 | ImageNet | $\geq 0.95$ |
| GPT2-124M | OpenWebText | $\geq 0.95$ |
| GPT2-0.1B (AdamW) | OpenWebText | $\geq 0.99$ |
| GPT2-0.1B (Muon-NSGD) | OpenWebText | $\geq 0.99$ |

### Cross-Model-Size Validation (Chinchilla Dataset)

| Model Size | $L_\infty$ | $R^2$ |
|-----------|-----------|-------|
| 0.074B | 2.825 | 0.991 |
| 0.632B | 2.367 | 0.998 |
| 2.004B | 2.178 | 0.999 |
| 9.290B | 2.046 | 0.988 |
| 12.56B | 2.053 | 1.000 |

### Key Findings
- $L(T) \sim L_\infty + C/\sqrt{T}$ achieves $R^2 \geq 0.978$ across all tested settings.
- Learning rate transfer enables 80× extrapolation (100 → 5000 steps) with minimal error.
- Different learning rate schedules (linear, cosine, WSD) exhibit highly consistent behavior after normalization.
- Different optimizers (SGD, AdamW, Muon) all conform to the same scaling law.
- The scaling law form is consistent across model sizes ranging from 0.074B to 12.56B.

## Highlights & Insights
- **Bridging Theory and Practice**: Bounds derived from convex optimization theory can precisely characterize actual deep learning training behavior, supporting the bold hypothesis that deep learning is macroscopically convex.
- **Practical Extrapolation Tool**: Once $\eta_{\text{ref}}$ is determined from small-scale experiments, the optimal learning rate for any training budget can be computed directly, eliminating large-scale hyperparameter search.
- **Unified Schedule Analysis**: Linear decay, cosine decay, and WSD are equivalent within the scaling law framework up to constant factors, explaining why multiple schedules tend to perform similarly in practice.

## Limitations & Future Work
- "Convex Dominance" remains an empirical hypothesis; the loss landscape of deep learning is theoretically non-convex.
- The fitted parameters ($L_\infty$, $q_1$, $q_2$) require multiple training runs to estimate, incurring non-trivial initial cost.
- The theoretical analysis does not account for the warmup phase, which is important in practice.
- Validation is limited to the pre-training setting; scaling laws for fine-tuning may differ.

## Related Work & Insights
- **vs. Chinchilla Scaling Law**: Chinchilla characterizes the data/model size–loss relationship, while this paper characterizes the steps/learning rate–loss relationship; the two are complementary.
- **vs. Kaplan et al. (2020)**: An early LLM scaling law that does not address learning rate scaling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical connection from convex optimization to deep learning scaling laws is deeply insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage spans from ResNet to 12.5B GPT, a 170× range in model size.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and experimental presentation is clear.
- Value: ⭐⭐⭐⭐⭐ Offers direct guidance for LLM training planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](deepafl_deep_analytic_federated_learning.md)
- [\[ICLR 2026\] Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[ICLR 2026\] Weak-SIGReg: Covariance Regularization for Stable Deep Learning](weak-sigreg_covariance_regularization_for_stable_deep_learning.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](../../NeurIPS2025/optimization/emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)

</div>

<!-- RELATED:END -->
