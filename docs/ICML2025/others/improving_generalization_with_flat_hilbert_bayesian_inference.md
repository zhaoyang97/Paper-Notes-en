---
title: >-
  [Paper Note] Improving Generalization with Flat Hilbert Bayesian Inference
description: >-
  [ICML2025][Bayesian Inference] Proposes Flat Hilbert Bayesian Inference (FHBI), which generalizes the sharpness-aware minimization (SAM) concept of flatness from finite-dimensional Euclidean space to infinite-dimensional Reproducing Kernel Hilbert Space (RKHS), and integrates it with particle-based Bayesian inference, outperforming nine baselines with an average Top-1 accuracy of 73.7% on the VTAB-1K benchmark.
tags:
  - "ICML2025"
  - "Bayesian Inference"
  - "Sharpness-Aware Minimization"
  - "RKHS"
  - "SVGD"
  - "LoRA"
  - "Generalization"
date: 2026-05-08
content_hash: d5a0977c5162c6fb
---

# Improving Generalization with Flat Hilbert Bayesian Inference

**Conference**: ICML2025  
**arXiv**: [2410.04196](https://arxiv.org/abs/2410.04196)  
**Code**: To be confirmed  
**Area**: LLM Evaluation  
**Keywords**: Bayesian Inference, Sharpness-Aware Minimization, RKHS, SVGD, LoRA, Generalization

## TL;DR

Proposes Flat Hilbert Bayesian Inference (FHBI), which generalizes the sharpness-aware minimization (SAM) concept of flatness from finite-dimensional Euclidean space to infinite-dimensional Reproducing Kernel Hilbert Space (RKHS), and integrates it with particle-based Bayesian inference, outperforming nine baselines with an average Top-1 accuracy of 73.7% on the VTAB-1K benchmark.

## Background & Motivation

Bayesian inference models uncertainty through the posterior distribution. However, existing methods (such as SVGD) only approximate the empirical posterior $p(\theta|\mathcal{S})$, which is prone to overfitting the training set. On the other hand, SAM improves generalization by finding flat minima through minimizing loss sharpness, but is restricted to single-model optimization in finite-dimensional Euclidean space.

**Core Motivation**: Can the flatness minimization concept of SAM be integrated with particle Bayesian methods to approximate the **population posterior** $p(\theta|\mathcal{D})$ instead of the empirical posterior in the function space (RKHS), thereby simultaneously achieving:

1. Uncertainty quantification capabilities of Bayesian methods
2. Generalization improvements brought by flatness minimization
3. Interactive diversity among particles

## Method

### Population Posterior vs. Empirical Posterior

Define the empirical posterior and population posterior:

$$p(\theta|\mathcal{S}) \propto \exp(-\mathcal{L}_\mathcal{S}(\theta))p(\theta), \quad p(\theta|\mathcal{D}) \propto \exp(-\mathcal{L}_\mathcal{D}(\theta))p(\theta)$$

Proposition 4.1 proves that the population posterior is the solution to the following optimization problem:

$$\mathbb{Q}^* = \min_{\mathbb{Q} \ll \mathbb{P}_\theta} \left\{ \mathbb{E}_{\theta \sim \mathbb{Q}}[\mathcal{L}_\mathcal{D}(\theta)] + D_{\text{KL}}(\mathbb{Q} \| \mathbb{P}_\theta) \right\}$$

That is, the particle ensemble sampled from $\mathbb{Q}^*$ optimally minimizes the population loss, preventing overfitting.

### Function Space Generalization Bound (Theorem 4.2)

Generalizes existing Euclidean space generalization bounds to RKHS $\mathcal{H}^d$:

$$\tilde{L}_\mathcal{D}(f) \leq \max_{\|f'-f\|_{\mathcal{H}^d} \leq \rho} \tilde{L}_\mathcal{S}(f') + \mathcal{O}\left(\sqrt{\frac{\log(1+1/\rho^2)+\log(n/\delta)}{n-1}}\right)$$

The key contribution lies in handling the **infinite-dimensional** property of RKHS (such as RBF kernels), which prevents the direct application of dimension-dependent existing results.

### Bayesian Inference Generalization Bound (Theorem 4.3)

Connects function space sharpness to Bayesian inference—the KL divergence of the population posterior can be upper-bounded by the worst-case KL divergence of the empirical posterior:

$$D_{\text{KL}}(q_{[I+f]} \| p(\theta|\mathcal{D})) \leq \max_{\|f'-f\|_{\mathcal{H}^d} \leq \rho} D_{\text{KL}}(q_{[I+f']} \| p(\theta|\mathcal{S})) + \mathcal{O}(\cdot)$$

### FHBI Algorithm

Based on the above theory, FHBI adopts a **two-step iterative** update:

**Step 1 - Adversarial Perturbation (Ascent Step)**: Find the worst perturbation in RKHS along the direction of the function gradient

$$\hat{f}_k^* = \rho \frac{\nabla_f D_{\text{KL}}(q_{[I+f]} \| p(\cdot|\mathcal{S}))|_{f=f_k}}{\|\nabla_f D_{\text{KL}}(q_{[I+f]} \| p(\cdot|\mathcal{S}))|_{f=f_k}\|_{\mathcal{H}^d}}$$

**Step 2 - Function Descent Step**: Compute the gradient at the perturbed position and update

$$f_{k+1} = f_k - \epsilon \nabla_f D_{\text{KL}}(q_{[I+f]} \| p(\cdot|\mathcal{S}))|_{f=f_k+\hat{f}_k^*}$$

In practice, $m$ particles $\{\theta_i\}_{i=1}^m$ are maintained. The update of each particle involves information interaction among all particles:

- **Sharpness Minimization**: Each particle searches for flat regions (similar to SAM)
- **Angle Repulsion Force**: Directs gradient diversity of particles (minimizing $\nabla_{\theta_j}\mathcal{L}(\theta_j) \cdot \nabla_{\theta_k}\mathcal{L}(\theta_k)$)
- **Spatial Repulsion Force**: Kernel gradient term $\nabla_\theta k(\theta, \theta_j)$ prevents particle collapse

**Unified Perspective**: FHBI is a generalization of SAM and SVGD—it degenerates to SVGD when $\rho=0$, and to SAM when $m=1$.

## Key Experimental Results

Evaluated on the VTAB-1K benchmark (19 datasets, spanning Natural, Specialized, and Structured categories) using ViT-B/16 + LoRA fine-tuning:

| Method | Natural (7) | Specialized (4) | Structured (8) | Average |
|------|------------|-----------------|----------------|------|
| AdamW | 79.1 | 84.3 | 59.0 | 72.0 |
| SAM | 80.1 | 83.2 | 56.0 | 70.5 |
| DeepEns | 79.3 | 83.9 | 42.8 | 67.0 |
| BayesTune | 80.5 | 84.9 | 59.3 | 72.2 |
| SVGD | 79.8 | 84.6 | 56.3 | 70.9 |
| SADA-JEM | 80.3 | 84.7 | 58.6 | 72.1 |
| **FHBI** | **82.4** | **86.9** | **61.6** | **73.7** |

- FHBI achieves the best performance across all three domains (Natural, Specialized, and Structured)
- Achieves an average Top-1 accuracy of 73.7%, outperforming the best baseline BayesTune by 1.5 percentage points
- Shows a more pronounced advantage on the highly challenging Structured datasets

**Expected Calibration Error (ECE)**: FHBI also achieves the lowest ECE across multiple datasets, indicating better model confidence calibration.

## Highlights & Insights

1. **Solid Theoretical Contribution**: Generalizes the generalization bound of SAM from finite dimensions to infinite-dimensional RKHS for the first time, which is non-trivial as it requires addressing infinite-dimensionality.
2. **Elegant Unified Perspective**: FHBI unifies SAM (single-particle flatness) and SVGD (multi-particle posterior approximation), revealing their intrinsic relationship.
3. **Triple Diversity Mechanism**: Sharpness + Angle Repulsion + Spatial Repulsion, a triple-force mechanism that promotes particle diversity.
4. **Comprehensive Experiments**: 19 datasets × 9 baselines, validating effectiveness on both Top-1 accuracy and ECE metrics.
5. **Natural Integration with LoRA**: Multi-particle inference is run only on lightweight LoRA parameters, keeping computational overhead manageable.

## Limitations & Future Work

1. **Computational Cost**: $m$ particles entail $m$ times forward/backward passes and kernel matrix computation, leading to non-negligible overhead when scaling to large-scale models.
2. **Evaluated Only on ViT-B/16 + LoRA**: Not verified on larger models (such as ViT-L or LLMs) or other PEFT methods.
3. **Sensitivity to Kernel Choice**: The paper employs the RBF kernel without fully exploring the impact of other kernel functions.
4. **Theory-to-Practice Gap**: Lemma 4.4 relies on the approximation of sufficiently small $\|f\|$, and the degree to which this condition is met in practice is not thoroughly discussed.
5. **Focus Only on Image Classification**: Generalization has not been validated on NLP, object detection, or other tasks.

## Related Work & Insights

- **SAM** (Foret et al., 2021): Euclidean space special case of FHBI ($m=1$)
- **SVGD** (Liu & Wang, 2016): Perturbation-free special case of FHBI ($\rho=0$)
- **BayesTune** (Kim et al., 2023): Bayesian fine-tuning baseline
- **SA-BNN** / **SADA-JEM**: Sharpness-aware Bayesian methods
- **Insight**: Systematically elevating optimization techniques (sharpness awareness) to the function space level is an important direction for Bayesian deep learning.

## Rating

- Novelty: ⭐⭐⭐⭐ (RKHS generalization bound + unified SAM/SVGD perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (19 datasets × 9 baselines, multiple metrics)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations and intuitive diagrams)
- Value: ⭐⭐⭐⭐ (Meaningful fusion of Bayesian inference and generalization theory)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ICML 2025\] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces](function_encoders_a_principled_approach_to_transfer_learning_in_hilbert_spaces.md)
- [\[ICML 2025\] IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [\[ICML 2025\] Latent Variable Estimation in Bayesian Black-Litterman Models](latent_variable_estimation_in_bayesian_black-litterman_models.md)
- [\[ICML 2025\] Set-Valued Predictions for Robust Domain Generalization](set_valued_predictions_for_robust_domain_generalization.md)

</div>

<!-- RELATED:END -->
