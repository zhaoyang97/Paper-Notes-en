---
title: >-
  [Paper Note] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies
description: >-
  [NeurIPS 2025][Optimization][Natural Gradient Descent] This work replaces standard SGD with the natural gradient descent optimizer iVON for optimizing BNN parameters under variational inference, achieving better uncertainty calibration in radio galaxy classification while maintaining predictive performance comparable to HMC and BBB-VI.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Natural Gradient Descent"
  - "Variational Inference"
  - "Bayesian Neural Networks"
  - "Uncertainty Calibration"
  - "Radio Galaxy Classification"
date: 2026-05-08
content_hash: 769323b10286d1af
---

# Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies

**Conference**: NeurIPS 2025
**arXiv**: [2511.13224](https://arxiv.org/abs/2511.13224)  
**Code**: [Available](https://github.com/devinamhn/RadioGalaxies-BNNs)  
**Area**: Bayesian Deep Learning / Radio Astronomy
**Keywords**: Natural Gradient Descent, Variational Inference, Bayesian Neural Networks, Uncertainty Calibration, Radio Galaxy Classification

## TL;DR

This work replaces standard SGD with the natural gradient descent optimizer iVON for optimizing BNN parameters under variational inference, achieving better uncertainty calibration in radio galaxy classification while maintaining predictive performance comparable to HMC and BBB-VI.

## Background & Motivation

Future radio astronomy surveys are expected to produce exabyte-scale data, demanding statistically robust ML models. **Bayesian Neural Networks (BNNs)** offer a principled approach to uncertainty modeling.

**Prior benchmark work** found that:
- HMC (Hamiltonian Monte Carlo) achieves overall best performance in predictive accuracy, calibration, and out-of-distribution detection, but at prohibitive computational cost (7 days)
- BBB-VI (Bayes by Backprop) performs adequately but suffers from initialization sensitivity, slow convergence, and cold posterior effects

**Core Problem**: Standard VI optimizes variational parameters via SGD, yet the VI parameter space is a **statistical manifold** (Riemannian manifold) where each point corresponds to a probability distribution. SGD assumes Euclidean geometry and may not identify the most efficient optimization direction in distribution space.

**Natural Gradient Descent (NGD)** preconditions gradients with the inverse Fisher Information Matrix (FIM), accounting for the geometry of the statistical manifold and providing a more direct path through distribution space.

## Method

### Overall Architecture

The **iVON (Improved Variational Online Newton)** algorithm replaces SGD for optimizing variational parameters of the BNN. iVON is grounded in the **Bayesian Learning Rule (BLR)** framework, which unifies variational inference and natural gradient descent.

### Key Designs

1. **Natural Gradient Updates under the BLR Framework**:

    - Variational distribution: multivariate Gaussian $q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}|\mathbf{m}, \mathbf{S}^{-1})$
    - Natural gradient update on natural parameters: $\boldsymbol{\lambda}_{t+1} \leftarrow \boldsymbol{\lambda}_t - \alpha \nabla_{\boldsymbol{\mu}}\{\mathbb{E}_{q}[l(\boldsymbol{\theta})] - \mathcal{H}(q)\}$
    - Equivalent to a Newton-like update requiring both gradient and Hessian information

2. **Scalable Approximations in iVON**:

    - **Diagonal Hessian approximation**: reduces complexity from $O(d^2)$ to $O(d)$
    - **Reparameterization trick** for second-order estimation: $\hat{\mathbf{h}} = \hat{\mathbf{g}} \cdot (\boldsymbol{\theta} - \mathbf{m}) / \boldsymbol{\sigma}^2$ — approximates curvature by measuring gradient response to random parameter perturbations
    - **Geometric correction term**: ensures positive definiteness of the precision matrix throughout training
    - Mean and standard deviation updates: $\mathbf{m} \leftarrow \mathbf{m} - \alpha \frac{(\hat{\mathbf{g}} + \delta\mathbf{m})}{(\mathbf{h} + \delta)}$, $\boldsymbol{\sigma} \leftarrow \frac{1}{\sqrt{\text{ess}(\mathbf{h} + \delta)}}$

3. **Key Differences from BBB-VI**:

    - BBB-VI computes separate Euclidean gradients for mean and variance: $\mathbf{m} \leftarrow \mathbf{m} - \alpha \nabla_\mathbf{m} \mathcal{L}$
    - iVON employs natural gradients, with the curvature estimate $(\mathbf{h} + \delta)$ in the denominator providing adaptive step sizes
    - iVON requires only 1 MC sample, consistent with BBB

### Loss & Training

- Dataset: MiraBest Confident (FRI/FRII radio galaxy binary classification; 584 train / 145 validation / 104 test)
- Architecture: LeNet-like network
- Learning rate 0.2 with cosine annealing and 5-epoch warmup
- Hessian initialization $h_0 = 0.5$ (selected from $\{0.01, 0.1, 0.5, 1, 5\}$)
- **Effective Sample Size (ESS)**: $\text{ess} = 10N$ yields best results (cold posterior)
- Weight decay $\delta = 10^{-4}$, trained for 1000 epochs across 10 random seeds

## Key Experimental Results

### Main Results — Predictive Performance and Calibration (Table 1)

| Inference Method | Test Error Rate ↓ | UCE ↓ | Training Time |
|---|---|---|---|
| HMC | 4.16 ± 0.45 | 14.76 ± 0.95 | **7 days** |
| BBB-VI | 3.94 ± 0.01 | 12.77 ± 6.11 | 40 min |
| **iVON (ess=10N)** | **3.07 ± 1.47** | **8.37 ± 4.12** | **25 min** |
| iVON (ess=100N) | 3.36 ± 1.23 | 12.19 ± 6.57 | 25 min |

### Out-of-Distribution Detection (Energy Score Analysis)

| Dataset | HMC | BBB-VI | iVON |
|---|---|---|---|
| MiraBest (iD) | Low energy ✓ | Low energy ✓ | Low energy ✓ |
| GalaxyMNIST (far OoD, optical) | High energy ✓ | High energy ✓ | High energy ✓ |
| MIGHTEE (near OoD, different radio telescope) | High energy ✓ | Moderate separation ✓ | Unreliable detection ✗ |

### Key Findings

- **iVON achieves best UCE across all methods** (8.37 vs. BBB-VI 12.77 vs. HMC 14.76), demonstrating substantially improved uncertainty calibration
- Predictive performance is competitive with the best methods (3.07%), with **37.5% faster training than BBB-VI** and over 400× faster than HMC
- **Cold posterior effect persists**: ess=10N (cold posterior) outperforms ess=N (standard posterior), indicating that even improved optimizers do not eliminate this effect
- **Near-OoD detection degrades**: iVON fails to reliably detect MIGHTEE data (radio galaxies from telescopes with different resolution/sensitivity), though far-OoD detection (optical galaxies) remains effective
- Different optimizers converge to **qualitatively distinct solutions** — better calibration but weaker near-OoD detection — highlighting that the inductive bias introduced by the optimizer is non-negligible

## Highlights & Insights

- **Optimizer as inductive bias**: The paper reveals an important observation — the choice of optimizer not only affects convergence speed but also determines the type of representation learned (distributed/redundant vs. compressed/local), with downstream consequences for different evaluation criteria
- Natural gradient descent exploits the Riemannian geometry of parameter space, providing a more "natural" optimization direction for VI
- Though small in scale, the experiments are **analytically thorough**, with each finding grounded in concrete implications for physical applications

## Limitations & Future Work

- Validation is limited to a small LeNet architecture and a small dataset; confirmation on larger architectures and datasets is needed
- Degraded near-OoD detection is a significant limitation, restricting applicability in scenarios requiring detection of distributional shift across different telescopes
- The cold posterior effect remains unresolved, pointing to deeper issues of model misspecification
- The ESS hyperparameter requires domain knowledge to set appropriately, with no automated selection strategy proposed
- The diagonal Hessian approximation may discard important inter-parameter correlations

## Related Work & Insights

- **iVON** (Lin et al.) and **BLR** (Khan & Rue) provide an elegant theoretical framework unifying diverse learning algorithms under the Bayesian Learning Rule
- **Noisy Natural Gradient** (Zhang et al.) represents an earlier attempt at natural gradient VI
- The findings offer direct guidance for researchers applying BNNs in astronomy and other scientific domains: optimizer choice must be considered in light of its impact across multiple evaluation dimensions
- This work motivates future research into formally characterizing the inductive biases of different optimizers and their effects on posterior approximation quality

## Rating

- Novelty: ⭐⭐⭐⭐ — Applies an existing optimizer to a new domain; theoretical contribution is limited, but the observations are valuable
- Experimental Thoroughness: ⭐⭐⭐⭐ — Small-scale but carefully analyzed, with faithful reproduction of prior benchmark results
- Writing Quality: ⭐⭐⭐⭐⭐ — Background is clearly explained and mathematical derivations are complete
- Value: ⭐⭐⭐⭐ — Directly useful to the radio astronomy BNN community; insights are broadly inspiring for the wider BNN community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Least Squares Variational Inference](least_squares_variational_inference.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

</div>

<!-- RELATED:END -->
