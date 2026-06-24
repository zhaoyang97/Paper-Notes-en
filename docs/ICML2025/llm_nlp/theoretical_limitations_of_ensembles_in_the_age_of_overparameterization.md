---
title: >-
  [Paper Note] Theoretical Limitations of Ensembles in the Age of Overparameterization
description: >-
  [ICML2025][LLM (Other)][Overparameterized ensembles] Under overparameterization conditions, infinite ensembles are pointwise equivalent to a single infinitely wide model. The ensemble variance no longer reflects traditional Bayesian uncertainty but instead measures the expected effect of increasing model capacity, providing a theoretical explanation for empirical observations that deep ensembles offer no fundamental generalization advantage over large models.
tags:
  - "ICML2025"
  - "LLM (Other)"
  - "Overparameterized ensembles"
  - "random feature models"
  - "uncertainty quantification"
  - "kernel regression"
  - "generalization theory"
date: 2026-05-08
content_hash: 09752c9ec0ca2321
---

# Theoretical Limitations of Ensembles in the Age of Overparameterization

**Conference**: ICML2025  
**arXiv**: [2410.16201](https://arxiv.org/abs/2410.16201)  
**Code**: Not open-sourced  
**Area**: LLM/NLP Theory · Ensemble Learning · Overparameterization Theory  
**Keywords**: Overparameterized ensembles, random feature models, uncertainty quantification, kernel regression, generalization theory

## TL;DR

Under overparameterization conditions, infinite ensembles are pointwise equivalent to a single infinitely wide model. The ensemble variance no longer reflects traditional Bayesian uncertainty but instead measures the expected effect of increasing model capacity, providing a theoretical explanation for empirical observations that deep ensembles offer no fundamental generalization advantage over large models.

## Background & Motivation

Classical ensemble learning (bagging, boosting, random forests, etc.) performs exceptionally well on **underparameterized** models, obtaining better generalization and robustness by aggregating multiple weak learners. However, in recent years, empirical studies have found that these intuitions no longer hold on **overparameterized** neural networks:

**No Generalization Advantage**: Abe et al. (2022b) found that the predictions of deep ensembles (e.g., 4 ResNet-18s) are highly consistent with a single larger model (e.g., a WideResNet-18 with 4x width), across both in-distribution and out-of-distribution data.

**Failure of Classical Diversity Strategies**: Methods that increase member diversity, such as bagging, can even be detrimental to overparameterized ensembles (Nixon et al., 2020; Jeffares et al., 2024).

**Questionable Reliability of Uncertainty Quantification**: Multiple studies have questioned the reliability of uncertainty estimation in deep ensembles (Abe et al., 2022b; Theisen et al., 2024).

The core questions of this paper are:

- **Q1**: Do overparameterized ensembles have any generalization or robustness advantages over a single large model?
- **Q2**: Does the predictive variance of overparameterized ensembles correspond to traditional frequentist or Bayesian uncertainty?

## Method

### Theoretical Framework: Random Feature (RF) Regression Model

The random feature linear regressor is used as an analyzable proxy for neural networks. The RF model is formulated as follows:

$$h_{\mathcal{W}}(x) = \frac{1}{\sqrt{D}} \sum_{i=1}^{D} \phi(\omega_i, x) \theta_i$$

where $\omega_i \sim \pi(\cdot)$ represents the randomly sampled feature parameters, $\theta_i$ represents the learnable weights, and $D$ represents the number of features (width). Overparameterization implies $D > N$ (the number of features is greater than the number of training samples).

### Key Designs 1: Weakest Assumption (Assumption 1)

Unlike prior works that assume Gaussian random features, this paper only requires a **sub-exponential** condition:

- The whitened features $w_i w_{\perp i}$ follow a sub-exponential distribution (light-tailed).
- The feature matrix $\Phi$ is almost surely of full rank.

This assumption is significantly weaker than Gaussianity and can cover practical activation functions like ReLU and softplus (via smooth approximation), ensuring the conclusions do not depend on specific feature distributions.

### Key Designs 2: Infinite Ensemble = Infinite-Width Single Model (Theorem 1)

**Core Theorem**: Under Assumption 1, the overparameterized RF regressor of an infinite ensemble $\bar{h}_\infty^{(LN)}$ and a single infinitely wide RF regressor $h_\infty^{(LN)}$ are pointwise almost surely equivalent:

$$\bar{h}_\infty^{(LN)}(x^*) = h_\infty^{(LN)}(x^*) = k_N(x^*)^\top K^{-1} y$$

That is, the ensemble mean is exactly equal to the ridgeless kernel regressor. The key lemma for the proof is:

$$\mathbb{E}_{W, w_\perp}[w_\perp^\top W^\top (WW^\top)^{-1}] = 0$$

This result holds even when $w_\perp$ and $W$ **have dependencies**, significantly generalizing prior analyses that required independence/Gaussianity.

### Key Designs 3: Non-Asymptotic Analysis Under Finite Parameter Budget (Theorem 2)

Given a total of $MD$ random features, an ensemble of $M$ members each with $D$ features is compared against a single model using all $MD$ features:

$$\|h_{\mathcal{W}^*}^{(LN)}(\cdot) - \bar{h}_{\mathcal{W}_{1:M}}^{(LN)}(\cdot)\|_2^2 \leq O(\sqrt{\log(1/\delta)}) + O(1/D)$$

That is, under the same parameter budget, the difference between the ensemble and the single large model vanishes as the single-member width $D$ increases.

### Key Designs 4: The Essence of Ensemble Variance (Sec 3.3)

The ensemble predictive variance equals the expected squared difference between the finite-width and infinite-width model predictions:

$$\mathbb{V}_{\mathcal{W}}[h_{\mathcal{W}}^{(LN)}(x^*)] = \mathbb{E}_{\mathcal{W}}[(h_{\mathcal{W}}^{(LN)}(x^*) - h_\infty^{(LN)}(x^*))^2]$$

- **Under Gaussian features**: Variance $= r_\perp^2 \cdot \frac{\|h_\infty^{(LN)}\|_k^2}{D-N-1}$, where $r_\perp^2$ is exactly the GP posterior variance, which admits a Bayesian interpretation.
- **Under general features**: The variance depends on the complex joint expectation of $W$ and $w_\perp$, which **is not equal to** a scalar multiple of the GP posterior variance—meaning the ensemble variance **does not** possess the traditional meaning of uncertainty.

### Key Designs 5: Smooth Transition of Small Ridge Regularization (Theorem 3)

When a small ridge parameter $\lambda > 0$ is added, the difference between the infinite ensemble and the infinite-width single model is Lipschitz continuous with respect to $\lambda$:

$$|\bar{h}_{\infty,\lambda}^{(RR)}(x^*) - h_{\infty,\lambda}^{(RR)}(x^*)| \leq C \cdot \lambda$$

The constant $C$ is independent of the test point $x^*$ (on a compact space), ensuring the conclusion still approximately holds when using small regularization in practice.

## Key Experimental Results

### Main Results

| Experimental Setup | Data | Results |
|---------|------|------|
| RF Ensemble vs Infinite-Width RF (Fig 1) | Synthetic data, N=6 | No perceptible difference between predictions of M=10000 ensemble and the infinite-width model |
| Overparameterization vs Underparameterization (Fig 2 Left) | California Housing, N=12, softplus | When D>N, the difference between ensemble and large model drops sharply (hockey-stick shape) |
| Neural Network Ensemble (Fig 2 Right) | California Housing, N=12000, ReLU | Neural networks also exhibit a similar hockey-stick pattern |
| Comparison Under Equal Parameter Budget (Fig 3 Left) | RF, N=12, ReLU, D=200/member | Generalization errors of ensemble and single model are almost identical as total parameters vary |
| NN Under Equal Parameter Budget (Fig 3 Right)| 3-layer MLP, N=12000, width=256 | Generalization performance of deep ensembles and larger single models are almost identical |

### Uncertainty Quantification Experiments (Fig 4)

| Variance Type | N=6, D=200, ReLU | Conclusion |
|---------|-------------------|------|
| RF Ensemble Variance | Spatially non-uniform distribution | Correlated with model capacity sensitivity |
| GP Posterior Variance | Classically larger away from data points | Reflects uncertainty of data coverage |
| Difference Between Both | **Significantly different** | Ensemble variance ≠ Bayesian uncertainty |

### Smoothness Experiment of Small Ridge (Fig 5)

California Housing, N=12, D=200, ReLU: The ensemble-to-single-model difference grows linearly with $\lambda$, and all 500 test points exhibit good Lipschitz continuity, validating Theorem 3.

## Highlights & Insights

1. **Strong Equivalence under the Weakest Assumption**: Proving the ensemble-large model equivalence requires only sub-exponential conditions, relying neither on Gaussian assumptions nor asymptotic analysis, greatly expanding the theoretical scope.
2. **Precisely Revealing the Nature of Ensemble Variance**: The variance measures the "expected effect of increasing capacity" rather than uncertainty—providing the first theoretical explanation for the empirical findings in Abe et al. (2022b).
3. **The Double-Edged Sword of Gaussianity**: Under Gaussian features, the ensemble variance is exactly equal to the GP posterior variance, creating the illusion of "effective" uncertainty quantification; this connection breaks once the Gaussian assumption is discarded.
4. **Dual Guarantees with Non-Asymptotic and Asymptotic Analysis**: Theorem 2 provides high-probability bounds under finite parameters, while Theorem 1 covers the asymptotic limit.
5. **Clear Contrast between Underparameterization and Overparameterization**: The width controls the implicit ridge parameter in the underparameterized regime, but has no effect on ensemble predictions in the overparameterized regime—this transition is intuitively illustrated by the hockey-stick shape.

## Limitations & Future Work

1. **RF Models vs. Real Neural Networks**: RFs only train the last layer without feature learning, failing to fully explain NN behaviors (although experiments show similar trends).
2. **Regression-Only Tasks**: The theoretical analysis is limited to regression settings; generalization to classification tasks remains unclear.
3. **Technical Limitations of ReLU**: ReLU does not satisfy the full-rank assumption (requiring smooth approximations like softplus), so rigorous conclusions must be obtained indirectly through limit arguments.
4. **The Practical Value of Ensembles Is Not Entirely Refuted**: The paper does not deny that ensembles are useful in practice (e.g., parallel training, hyperparameter search); it merely points out that their advantages can be replicated by a larger single model.
5. **Precise Bounds for Ridge Regularization**: Theorem 3 provides Lipschitz continuity but does not give an explicit constant; how large a $\lambda$ causes the equivalence to break down in practice still requires a case-by-case analysis.

## Related Work & Insights

- **Deep Ensembles Work, But Are They Necessary?** (Abe et al., NeurIPS 2022): The main empirical motivation for this study, which found that ensemble variance is strongly correlated with capacity sensitivity.
- **Jacot et al. (2020)**: Analyzed ensemble variance under Gaussian RFs; this work significantly relaxes its assumptions.
- **Adlam & Pennington (2020)**: High-dimensional asymptotic analysis of overparameterized RFs, relying on Gaussian universality.
- **Ruben et al. (2024)**: Concurrent work that also found no advantage for RF ensembles, though using optimal ridge tuning and Gaussian assumptions.
- **Neal (1996), Williams (1996)**: Classical connection between infinitely wide NNs and GPs; this paper distinguishes ensembles from GPs based on this foundation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Unifies the ensemble-single model equivalence theory under the weakest assumptions, revealing the non-uncertainty nature of ensemble variance.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Mutually validating RF and NN experiments across multiple activation functions and datasets, though lacking large-scale NN experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Precise theoretical formulations, clear intuitive explanations, beautiful illustrations, and the hockey-stick plot is highly intuitive.
- Value: ⭐⭐⭐⭐ — Provides a solid theoretical foundation for whether ensembles are still needed in the era of large models, offering an important warning to the practice of uncertainty quantification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding](on_expressive_power_of_looped_transformers_theoretical_analysis_and_enhancement_.md)
- [\[ACL 2025\] Argument Mining in the Age of Large Language Models](../../ACL2025/llm_nlp/argument_mining_in_the_age_of_large_language_models.md)
- [\[ACL 2025\] Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers](../../ACL2025/llm_nlp/can_llms_identify_critical_limitations_within_scientific_research_a_systematic_e.md)
- [\[ACL 2026\] Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification](../../ACL2026/llm_nlp/nürnberg_nlp_at_psydefdetect_multi-axis_voter_ensembles_for_psychological_defenc.md)
- [\[ICLR 2026\] Is the Reversal Curse a Binding Problem? Uncovering Limitations of Transformers from a Basic Generalization Failure](../../ICLR2026/llm_nlp/is_the_reversal_curse_a_binding_problem_uncovering_limitations_of_transformers_f.md)

</div>

<!-- RELATED:END -->
