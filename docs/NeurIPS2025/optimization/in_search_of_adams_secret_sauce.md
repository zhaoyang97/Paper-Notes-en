---
title: >-
  [Paper Note] In Search of Adam's Secret Sauce
description: >-
  [NeurIPS 2025][Optimization][Adam] Through large-scale experiments training 1500+ language models, this paper establishes: (1) Signum closes 96% of the SGD–Adam gap yet remains 25% slower than Adam…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Adam"
  - "Signum"
  - "implicit bias"
  - "variational inference"
  - "signal-to-noise ratio"
  - "language modeling"
date: 2026-05-08
content_hash: cc6f83f9120f8c7c
---

# In Search of Adam's Secret Sauce

**Conference**: NeurIPS 2025
**arXiv**: [2505.21829](https://arxiv.org/abs/2505.21829)
**Code**: [GitHub](https://github.com/aorvieto/SecretSauce)
**Area**: Optimization
**Keywords**: Adam, Signum, implicit bias, variational inference, signal-to-noise ratio, language modeling

## TL;DR
Through large-scale experiments training 1500+ language models, this paper establishes: (1) Signum closes 96% of the SGD–Adam gap yet remains 25% slower than Adam; (2) setting $\beta_1 = \beta_2$ is a near-optimal simplification of Adam; (3) under $\beta_1 = \beta_2 = \beta$, Adam can be reinterpreted as a signal-to-noise ratio–adaptive Signum that estimates the gradient mean and variance via online Gaussian variational inference.

## Background & Motivation
**The indispensability of Adam**: Although recent optimizers such as Muon, Scion, and SOAP outperform Adam in certain settings, they still rely on Adam to update embedding layers, LM heads, and normalization parameters. The core advantage of Adam remains incompletely understood.

**Adam ≈ Signum?**: Recent work has highlighted a close connection between Adam and SignSGD with momentum (Signum). Nevertheless, at the 160M parameter scale, a carefully tuned Signum still incurs a **25% effective slowdown**—requiring 25% more training budget to reach the same perplexity.

**Core Problem**: What is the "secret sauce" that distinguishes Adam from simplified variants such as Signum, RMSprop, and SGD?

**Methodology**: Approximately 10,000 A100 GPU hours are invested to systematically ablate all hyperparameters—including independently tuning momentum parameters for each learning rate—providing a comprehensive and reproducible benchmark.

## Method

### Large-Scale Benchmark Experiments (§3)

**Basic setup**: 160M-parameter Transformer LM, SlimPajama dataset, Chinchilla-optimal training.

**Run counts**:
- SGD: 131 runs (sweeping weight decay, gradient clipping, momentum, lr)
- RMSprop: 48 runs
- Signum: 70 runs (sweeping clipping, momentum, lr)
- Adam: 200 runs (sweeping $\beta_1$, $\beta_2$, lr)

### Takeaway 1: Signum Closes the Gap but Falls Short

| Optimizer | Validation Perplexity |
|-----------|----------------------|
| Adam | **21.86** ± 0.21 |
| Signum | 23.23 ± 0.16 |
| RMSprop | 27.04 ± 0.34 |
| SGD+Cclip | 33.40 ± 0.39 |
| SignSGD | 36.78 ± 0.57 |
| SGD+Gclip | 37.76 ± 0.61 |
| SGD | 53.62 ± 5.14 |

Signum closes 96% of the SGD–Adam gap, yet a 1.37 perplexity-point difference remains, corresponding to a 25% training efficiency loss.

### Takeaway 2: $\beta_1 = \beta_2$ Is a Near-Optimal Choice
Among 200 Adam configurations, the optimal $\beta_2$ is consistently close to $\beta_1$ for every value of $\beta_1$. Restricting to $\beta_1 = \beta_2$ incurs a performance degradation of no more than 0.3 perplexity points—far smaller than the 1.37-point gap with Signum.

### Theoretical Interpretation (§4): A Variational Inference Perspective

Setting $\beta_1 = \beta_2 = \beta$ and $\epsilon = 0$ (experiments confirm that $\epsilon$ has no significant effect in the range $[10^{-6}, 10^{-15}]$):

**Proposition 1**: The Adam update direction can be rewritten as:

$$d_k = \frac{m_k}{\sqrt{m_k^2 + \beta \cdot \text{EMA}_\beta[(m_{k-1} - g_k)^2]}}$$

The term $\beta \cdot \text{EMA}_\beta[(m_{k-1} - g_k)^2]$ in the denominator is precisely an online estimate of the gradient variance.

**Theorem 4.1 (Variational Inference Interpretation)**: The two momentum buffers of Adam correspond to the closed-form solution of the following KL-regularized maximum likelihood problem:

$$\min_{m, \sigma^2 \geq 0} -\log p(g_{k+1} | m, \sigma^2) + \frac{1}{\lambda} \text{KL}(\mathcal{N}(m_k, \sigma_k^2) \| \mathcal{N}(m, \sigma^2))$$

where $\beta = 1/(1+\lambda)$, with solutions:
- $m_{k+1} = \beta m_k + (1-\beta) g_{k+1} = \text{EMA}_\beta[g_{k+1}]$
- $\sigma_{k+1}^2 = \beta \sigma_k^2 + \beta(1-\beta)(m_k - g_{k+1})^2 = \beta \cdot \text{EMA}_\beta[(m_k - g_{k+1})^2]$

**Signal-to-Noise Ratio (SNR) Adaptive Trust Region**: Adam can be viewed as Signum with a step size that adapts to the SNR:

$$d_k = \frac{\text{sign}(m_k)}{\sqrt{1 + \sigma_k^2 / m_k^2}}$$

- High SNR ($\sigma_k^2 / m_k^2$ small) → step size ≈ 1 → approaches Signum
- Low SNR → step size is shrunk → conservative update

**Uniqueness (Proposition, §C.2)**: The Adam update direction admits the representation $m_k / \sqrt{m_k^2 + \gamma \cdot \text{EMA}_\tau[(a m_{k-1} - b g_k)^2]}$ **if and only if** $\beta_1 = \beta_2$. That is, only when the two beta parameters are equal does the denominator admit an exact variance-estimation interpretation.

### Ruling Out Confounders (§3.3)

| Confounder | Conclusion |
|------------|-----------|
| $\epsilon$ value | No significant difference in $[10^{-6}, 10^{-15}]$; adding an $\epsilon$ mollifier to Signum yields no improvement |
| Momentum initialization | Zero initialization vs. gradient initialization makes no meaningful difference |
| Bias correction | Final validation perplexity is nearly unchanged |

## Key Experimental Results

### Robustness of $\beta_1 = \beta_2$ Across Settings

| Ablation Dimension | Performance of $\beta_1 = \beta_2$ |
|--------------------|-----------------------------------|
| Different batch sizes (128, 256, 512) | Consistently near-optimal |
| Different sequence lengths (512, 2048) | Consistently near-optimal |
| Different data (SlimPajama, Fineweb) | Consistently near-optimal |
| No weight decay | Consistently near-optimal |
| 2× token budget | Consistently near-optimal |
| 410M parameter model (44 runs) | Consistently near-optimal |

### Standard $(\beta_1, \beta_2) = (0.9, 0.95)$ vs. Equal Beta

| Model Scale | $(0.9, 0.95)$ | Equal Beta Optimum |
|-------------|---------------|--------------------|
| 160M | Competitive | Comparable or better |
| 410M | Suboptimal (Figure 5) | Better |

### Quadratic Function Validation (§5)

| Setting | SGD | Signum | Adam ($\beta_1=\beta_2$) |
|---------|-----|--------|--------------------------|
| Homogeneous Hessian | Slow | Fast | Fast (≈ Signum) |
| Heterogeneous Hessian (Transformer-like) | Very slow | Moderate | **Fastest** |

Key insight: On heterogeneous loss landscapes, the variance term $\sigma_k^2$ takes on different magnitudes across parameter blocks, enabling block-level adaptivity that a fixed mollifier cannot replicate.

## Highlights & Insights
- **Adam = SNR-Adaptive Signum**: This result precisely refines the observation of Balles & Hennig (2018)—who could not prove that the denominator estimates variance because they assumed $\beta_1 \neq \beta_2$. The $\beta_1 = \beta_2$ simplification makes this connection exact.
- **Unexpected Elegance of the Variational Inference Perspective**: Adam's two EMA buffers are exactly the online solution to Gaussian variational inference, and the regularization parameter $\lambda$ corresponds precisely to the EMA coefficient $\beta$.
- **A Fixed $\epsilon$ Cannot Replace Adaptive Variance**: Both experiments and the quadratic example demonstrate that Signum with a constant mollifier cannot match Adam's performance; a data-driven adaptive variance term is necessary.
- **$\beta_1 = \beta_2 = 0.95$ as a Default for LM Training**: This setting has been independently adopted by Zhao et al. and Shah et al., among others.

## Limitations & Future Work
- **Limited to 160M and 410M scales**: Despite the remarkable number of runs (1500+), the findings are not validated on models with 1B+ parameters.
- **Fixed hyperparameter grids**: Although results reside at grid interiors rather than boundaries, different grid choices could in principle yield different conclusions.
- **$\beta_1 = \beta_2$ may shift slightly at small batch sizes**: Figure 3 hints at this for batch size 128.
- **Theorem 4.1 explains Adam's structure but not why that structure works**: Why should the mean and variance be arranged as a ratio?
- **The effect of weight decay on the theory is not addressed**: The interaction between weight decay and the Adam update in AdamW is not covered theoretically.

## Rating
- Novelty: ⭐⭐⭐⭐ The $\beta_1 = \beta_2$ finding appears simple yet carries far-reaching implications; the variational inference interpretation is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 1500+ runs, ~10K A100 GPU hours, covering a wide range of ablation dimensions.
- Theoretical Depth: ⭐⭐⭐⭐ Proposition 1 and Theorem 4.1 are rigorously derived; the uniqueness proof is compelling.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative is clear (empirical findings → simplification → theoretical explanation → validation), and figures are carefully designed.
- Practical Value: ⭐⭐⭐⭐⭐ Directly actionable tuning advice: $\beta_1 = \beta_2$ reduces Adam to a single-momentum-parameter optimizer.

## Related Work & Insights

## Highlights & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[NeurIPS 2025\] MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions](mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)
- [\[NeurIPS 2025\] Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful](small_batch_size_training_for_language_models_when_vanilla_sgd_works_and_why_gra.md)

</div>

<!-- RELATED:END -->
