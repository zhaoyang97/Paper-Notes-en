---
title: >-
  [Paper Note] Generalizing while Preserving Monotonicity in Comparison-based Preference Learning Models
description: >-
  [NeurIPS 2025][LLM Alignment][preference learning] This paper proposes **Linear GBT with Diffusion Prior**, a class of preference learning models that simultaneously guarantee **monotonicity** (the score of the preferred item never paradoxically decreases after a comparison) and **generalization to uncompared items**, thereby affirmatively answering the central question of whether generalization and monotonicity can coexist.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - preference learning
  - monotonicity
  - Bradley-Terry
  - diffusion prior
  - alignment
date: 2026-05-08
content_hash: a330f27df6684c11
---

# Generalizing while Preserving Monotonicity in Comparison-based Preference Learning Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.08616](https://arxiv.org/abs/2506.08616)
**Code**: [github.com/pevab/gbtlab2](https://github.com/pevab/gbtlab2)
**Area**: LLM Alignment
**Keywords**: preference learning, monotonicity, Bradley-Terry, diffusion prior, alignment

## TL;DR

This paper proposes **Linear GBT with Diffusion Prior**, a class of preference learning models that simultaneously guarantee **monotonicity** (the score of the preferred item never paradoxically decreases after a comparison) and **generalization to uncompared items**, thereby affirmatively answering the central question of whether generalization and monotonicity can coexist.

## Background & Motivation

Preference learning (e.g., RLHF, DPO) is fundamental to LLM alignment, yet these algorithms exhibit a counterintuitive bug: **when the model is told "A is preferred over B," the score of A may actually decrease.**

A minimal example illustrates this: given a linear model $\beta \in \mathbb{R}^2$ and item embeddings $x_a=(1,0)$, $x_b=(2,0)$, the comparison $a \succ b$ pushes $\beta_1$ downward (since $x_{a1} < x_{b1}$), which in turn lowers $a$'s score $\beta^T x_a = \beta_1$. More critically, the score of a third item $c$ with embedding $(0,1)$ is entirely unaffected.

- **Background**: Classical **(Generalized) Bradley-Terry (GBT)** models guarantee monotonicity but cannot generalize—items never compared always receive a score of zero. Linear/nonlinear models with embeddings (including RLHF/DPO) can generalize but **cannot guarantee monotonicity**.
- **Key Challenge**: The tension between expressivity through embeddings and the structural constraints required for monotonicity.
- **Goal**: Design a preference learning algorithm that achieves both generalization and monotonicity.

## Method

### Overall Architecture

**Linear GBT with Diffusion Prior** extends the classical GBT model along two dimensions:

1. **Embeddings**: Each alternative $a$ is associated with an embedding $x_a \in \mathbb{R}^D$, and its score is modeled as the linear function $\theta_a(\beta) = x_a^T \beta$.
2. **Diffusion Prior**: A regularization term introduces a Laplacian matrix $L$ encoding prior similarity between alternatives.

### Key Designs

**Loss Function**:

$$\mathcal{L}(\beta|\mathbf{D}) = \underbrace{\frac{1}{2\sigma^2}\sum_d \beta_d^2 + \frac{1}{2}\sum_{ab}\theta_a(\beta)L_{ab}\theta_b(\beta)}_{\text{Regularization } \mathcal{R}(\beta)} + \sum_{(a,b,r)\in\mathbf{D}} \Phi_f(x_{a\ominus b}^T\beta) - r \cdot x_{a\ominus b}^T\beta$$

where $\Phi_f(\theta) = \log\int_{\mathfrak{R}} e^{r\theta} df(r)$ is the cumulant generating function of the root law $f$. The Laplacian regularization term $\sum_{ab} \theta_a L_{ab} \theta_b = \frac{1}{2}\sum_{a \neq b}|L_{ab}|(\theta_a - \theta_b)^2$ encourages similar alternatives to have similar scores.

**Core Theorem Chain for Monotonicity Guarantees**:

1. **Good Embedding**: An embedding $x$ is *good* if and only if for all Laplacian matrices $Y$ and all pairs $(a,b)$: $e_a^T(I + XY)^{-1}X e_{a\ominus b} \geq 0$.

2. **Diffusion Embedding**: An embedding $x$ is a *diffusion embedding* if and only if the inverse of the Gram matrix $X_\lambda = x^T x + \lambda I$ is a super-Laplacian matrix for all $\lambda > 0$.

3. **Derivation Chain**: Diffusion Embedding → Good Embedding → Monotonicity.

**Theorem 1 (Main Theorem)**: For any root law $f$, $\sigma > 0$, diffusion embedding $x$, and Laplacian $L$, the model $\text{GBT}_{f,\sigma,x,L}$ is monotone.

### Loss & Training

**Differential Analysis Framework**: The paper introduces a Smoothed Loss $\mathcal{L}_\lambda$ and employs the integral representation:

$$\theta^*(o(\mathbf{D})) - \theta^*(\mathbf{D}) = \int_0^1 \frac{d\theta_\lambda^*}{d\lambda}(\mathbf{D}, o) d\lambda$$

For the update operation, the derivative is given by:

$$\frac{d\theta_{\lambda a}^*}{d\lambda}\bigg|_{\lambda=\mu} = (r-s) \cdot e_a^T(I + X(L+H))^{-1}X e_{a\ominus b}$$

An analogous formula applies to the append operation, with an additional $\Phi_f''$ term. Monotonicity reduces to verifying that the matrix $(I + X(L+\tilde{H}))^{-1}X$ is nonnegative in the direction $e_a^T \cdot e_{a\ominus b}$.

**One-Hot Encoding Special Case (Theorem 2)**: Categorical one-hot encodings are diffusion embeddings, and scores decompose as $\theta_a = \gamma_{d(a)} + s^2 \cdot \alpha_a$ (category score + residual).

## Key Experimental Results

### Main Results

**Goodness Probability for Random Embeddings**:
- Gaussian i.i.d. embeddings $x$: probability is high when $D/A$ is large, and drops sharply when $A/D$ is large.
- After concatenating $[I, x]^T$: goodness probability improves substantially.

**nMSE Comparison** ($A=25$, $N=500$, uniform root law, 100 seeds):

| Embedding | nMSE Performance |
|-----------|-----------------|
| $[I, x]^T$ (full embedding) | **Best**—integrates both categorical and feature information |
| $I$ (classical GBT) | Better at small $D$, worse at large $D$ |
| $x$ (features only) | Worse at small $D$, better at large $D$ |

### Ablation Study

**Data Efficiency** ($A=20$, $D=10$, one-hot encoding, 1000 seeds):
- GBT with one-hot encoding achieves the same nMSE as classical GBT using roughly $1/2$ to $1/3$ of the comparisons.
- This demonstrates that structured embeddings can **substantially reduce data requirements**.

### Key Findings

1. Random embeddings are highly unlikely to satisfy the goodness condition; monotonicity is far from automatic.
2. Concatenating the identity matrix is a simple and theoretically grounded fix (Proposition 9).
3. The full embedding model $[I,x]^T$ consistently performs best across varying $D/A$ ratios by leveraging both GBT and feature-learning advantages.
4. Structured one-hot embeddings significantly reduce estimation error in data-scarce settings.

## Highlights & Insights

- **First affirmative answer** to the question of whether generalization and monotonicity can coexist.
- The **diffusion embedding** concept is elegant—treating comparisons as "heat pumps" where scores propagate analogously to heat diffusion.
- The work connects algebraic conditions with graph theory (Laplacians) and physics (diffusion dynamics), demonstrating notable theoretical depth.
- **Practical relevance**: directly applicable to collaborative content rating platforms (e.g., Tournesol) and recommender systems.
- Explicitly identifying the lack of monotonicity guarantees in mainstream methods such as RLHF/DPO is likely to raise community awareness of this issue.
- A simple practical fix: concatenating embeddings with the identity matrix substantially improves the probability of satisfying goodness.

## Limitations & Future Work

- Theoretical guarantees are **restricted to diffusion embeddings**; more general embedding classes (e.g., those produced by neural networks) are not covered.
- Only **linear models** are considered, whereas practical RLHF/DPO relies on nonlinear models (Transformers).
- Experiments are primarily conducted on synthetic data at modest scale; validation on real-world large-scale preference datasets is limited.
- The goodness condition is not sufficiently intuitive for practitioners.
- Extension of the theoretical framework to gradient-based nonlinear optimization pipelines such as DPO/RLHF is not discussed.
- The computational complexity of determining goodness—and whether it is NP-hard or efficiently decidable—is left open.

## Related Work & Insights

- Classical GBT (Fageot et al., AAAI 2024) established monotonicity without generalization; the present work is its natural extension.
- Bareilles et al. (2025) showed that nonlinear models satisfy only weak monotonicity (local pairwise), reinforcing the motivation for the linear + diffusion approach.
- Chen et al. (NeurIPS 2024) empirically identified ranking inconsistencies in DPO, providing empirical corroboration of the theoretical findings here.
- The framework has direct applicability to collaborative rating platforms such as Tournesol, ensuring that user preferences are never penalized by "negative feedback."
- The mathematical tools of super-Laplacian matrices and graph diffusion have potential to extend to more complex social choice problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to resolve the generalization + monotonicity compatibility problem.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — A complete proof chain from diffusion embeddings to monotonicity.
- **Experimental Thoroughness**: ⭐⭐⭐ — Primarily synthetic experiments; real-data validation is limited.
- **Practical Value**: ⭐⭐⭐⭐ — Direct implications for social choice and recommender systems.
- **Overall**: 8.0/10

## Related Papers & Insights

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Preference Learning with Lie Detectors can Induce Honesty or Evasion](preference_learning_with_lie_detectors_can_induce_honesty_or_evasion.md)
- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)
- [\[NeurIPS 2025\] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

</div>

<!-- RELATED:END -->
