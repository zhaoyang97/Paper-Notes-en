---
title: >-
  [Paper Note] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit
description: >-
  [ICLR 2026][Optimization][Multi-Index model] This paper proves that under generic non-degenerate assumptions, standard two-layer neural networks trained via layer-wise gradient descent can learn generic Gaussian Multi-Index models $f(\bm{x})=g(\bm{U}\bm{x})$ using $\tilde{O}(d)$ samples and $\tilde{O}(d^2)$ time. Both sample and time complexities reach information-theoretic optimality, providing the first proof that neural networks can efficiently learn hierarchical functions…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Multi-Index model"
  - "Information-theoretic lower bound"
  - "Feature learning"
  - "Power iteration"
  - "Two-layer neural network"
date: 2026-05-08
content_hash: 2022b3af534e62fa
---

# Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit

**Conference**: ICLR 2026  
**arXiv**: [2511.15120](https://arxiv.org/abs/2511.15120)  
**Code**: None  
**Area**: Optimization  
**Keywords**: Multi-Index model, Information-theoretic lower bound, Feature learning, Power iteration, Two-layer neural network

## TL;DR
This paper proves that under generic non-degenerate assumptions, standard two-layer neural networks trained via layer-wise gradient descent can learn generic Gaussian Multi-Index models $f(\bm{x})=g(\bm{U}\bm{x})$ using $\tilde{O}(d)$ samples and $\tilde{O}(d^2)$ time. Both sample and time complexities reach information-theoretic optimality, providing the first proof that neural networks can efficiently learn hierarchical functions.

## Background & Motivation

**Background**: The Multi-Index model $f^*(\bm{x})=g^*(\bm{U}\bm{x})$ is a standard framework for studying representation learning, where the output depends only on the projection of high-dimensional inputs onto a low-dimensional subspace. Information-theoretic lower bounds require $\Theta(d)$ samples, and polynomial-time spectral algorithms can achieve this bound.

**Limitations of Prior Work**:
   - While Single-Index models ($r=1$) have established theories, learning guarantees for Multi-Index models ($r>1$) in neural networks remain extremely limited.
   - Damian et al. (2022) proved that neural networks can learn Multi-Index models but require a suboptimal $\tilde{\Theta}(d^2)$ samples.
   - Arnaboldi et al. focused only on a specific subclass with leapfrog structures, which is far from optimal.
   - Recently, Mousavi-Hosseini & Wu (2026) could only weakly recover partial feature directions, failing to guarantee full subspace recovery.

**Key Challenge**: Information theory suggests $\Theta(d)$ samples are sufficient and spectral methods achieve this, but can **neural networks** (trained via standard gradient descent) reach this bound? This remained unproven.

**Key Insight**: By analyzing the dynamics of gradient descent under small initialization, it is observed that the inner layer weights implicitly perform a power iteration process. This is similar to spectral initialization on a local Hessian; however, the number of training steps must be precise. Too few steps fail to eliminate noise, while too many steps cause all features to align with the direction of the largest eigenvalue, losing non-dominant directions.

**Core Idea**: Neural network gradient descent implicitly executes "power iteration with early stopping." A suitable number of training steps precisely eliminates finite-sample noise without losing subspace diversity, thereby approaching information-theoretic optimality.

## Method

### Overall Architecture
The goal is to enable a standard two-layer network $f_{\Theta}(\bm{x}) = \sum_j a_j \sigma(\bm{w}_j^T \bm{x} + b_j)$ to learn the $r$-dimensional hidden subspace $\bm{U}$ of a Multi-Index model $f^*(\bm{x})=g^*(\bm{U}\bm{x})$ using only gradient descent, with sample complexity approaching the information-theoretic lower bound $\tilde{O}(d)$. Instead of using exotic algorithms, this work decomposes training into two sequential phases for the inner and outer layers: **Phase 1** trains the inner weights $\bm{W}$ on dataset $\mathcal{D}_1$ (for $T_1$ steps) to align neuron directions with the hidden subspace—this phase implicitly performs "power iteration with early stopping" and is the core contribution; **Phase 2** re-initializes the biases $\bm{b}$ and trains the outer weights $\bm{a}$ on an independent dataset $\mathcal{D}_2$ (for $T_2$ steps), which is a convex problem where standard GD converges to the optimum. The following three key designs revolve around the inner layer dynamics of Phase 1.

### Key Designs

**1. Implicit Power Iteration Mechanism: Reducing Inner GD to the Power Method on Local Hessian**

Why can neural networks find the hidden subspace through pure gradient descent? This paper proves that the first $T_1$ steps of inner weight updates in Phase 1 essentially perform power iteration on the local Hessian $\hat{\Sigma}_\ell = \frac{1}{n}\sum_i \ell_i \bm{x}_i\bm{x}_i^T$. Crucially, under small initialization ($\epsilon_0 \to 0$), the activation function is approximated by a Taylor expansion as quadratic ($\sigma''(0)=1$), and the gradient updates of individual neurons become nearly decoupled and independent. Each step reduces to:

$$\bm{w}_j^{(t+1)} \approx \hat{\Sigma}_\ell\, \bm{w}_j^{(t)},$$

which is the standard iteration for the power method. The spectrum of $\hat{\Sigma}_\ell$ separates signal from noise: large eigenvalues correspond to true hidden subspace directions, while small eigenvalues correspond to sampling noise from finite $n$. Power iteration naturally amplifies large eigenvalues and suppresses small ones, causing signal directions to grow exponentially while noise decays—provided the iteration stops at the correct number of steps.

**2. Critical Stopping Condition: The Inner Training Steps $T_1$ Must Be Exact**

The amplification of signals by power iteration is a double-edged sword. Stopping too early or too late is problematic. If $T_1$ is too small, noise directions are not sufficiently suppressed (especially at the most difficult sample size $n \sim d$), leading to a subspace estimate contaminated by noise. If $T_1$ is too large, all neurons are pulled toward the direction of the maximum eigenvalue, resulting in only one direction being learned and the remaining $r-1$ non-dominant directions being lost. The optimal $T_1$ falls within a window that allows all $r$ signal directions to be covered by enough neurons while suppressing noise to negligible levels. A counter-intuitive corollary is that the optimal number of steps for the first layer is **not** $O(1)$ but divergent ($T_1 \to \infty$), which contradicts analysis frameworks based on DMFT or state evolution that only run for $O(1)$ steps.

**3. General Loss Function = Data Preprocessing: Tuning the Spectrum of $\Sigma_\ell$ via Loss Selection**

This work reinterprets the choice of loss function as a form of feature preprocessing. The rationale is that the spectral structure of the local Hessian is determined by the first derivative of the loss $\ell'(0, y)$. Changing the loss function alters the spectral structure of $\Sigma_\ell$, thereby changing which directions are amplified by the power iteration. For Multi-Index models with a generative index of 2, there exists a loss that makes $\Sigma_\ell$ non-degenerate (Assumption 5), ensuring all signal directions can be activated. This elevates the loss function from a mere "optimization objective" to a "feature extraction preprocessor," extending the applicability from squared loss to more general functions like $\ell^1$ and Huber loss.

### Loss & Training
- Phase 1: $\bm{W}^{(t)} \leftarrow \bm{W}^{(t-1)} - \eta_1[\nabla_W \hat{\mathcal{L}}_{\mathcal{D}_1} + \beta_1 \bm{W}^{(t-1)}]$ (GD with weight decay)
- Phase 2: Standard GD on outer weights $\bm{a}$ (convex optimization, converges to optimality)
- Symmetric Initialization: $a_j = -a_{m-j}$, $\bm{w}_j^{(0)} = \bm{w}_{m-j}^{(0)}$ (to eliminate mean bias)
- Critical Learning Rate: Learning rates that are too small cause the power iteration approximation to fail.

## Key Experimental Results

### Main Results (Key Metrics from Theorem 1)

| Metric | Complexity | Info-theoretic Optimal? |
|----|--------|-----------|
| Sample Complexity | $\tilde{O}(d)$ | Yes (Leading order) |
| Time Complexity | $\tilde{O}(d^2)$ | Yes (Leading order) |
| Test Error | $o_d(1)$ → 0 | Yes |
| Applicable Rank | Any constant $r$ | - |
| Applicable Loss | Squared, $\ell^1$, Huber, etc. | - |

### Comparison with Existing Results

| Work | Sample Complexity | End-to-End? | Generic Model? |
|------|----------|-----------|---------|
| Damian et al. 2022 | $\tilde{\Theta}(d^2)$ | Yes | Partial |
| Arnaboldi et al. 2023 | $\tilde{O}(d^L)$ ($L$=leapfrog steps) | Yes | Leapfrog structure |
| Mousavi-Hosseini & Wu 2026 | $O(d)$ | **No** (Weak recovery only) | Yes |
| **Ours** | **$\tilde{O}(d)$** | **Yes** | **Yes** (Non-leapfrog) |

### Key Findings
- **First End-to-End Optimal Guarantee**: NN + GD learns the full Multi-Index model with info-theoretically optimal $\tilde{O}(d)$ samples.
- **Training Steps Cannot Be Too Few**: Optimal results require the first layer training steps $T_1 \to \infty$ (ruling out DMFT/state evolution frameworks requiring $O(1)$ steps).
- **Learning Rates Cannot Be Too Small**: The power iteration approximation requires moderately large learning rates—dynamics degrade if too small.
- **No Spectral Initialization/Warm Start Required**: Standard random initialization suffices—the network learns correct features purely via GD.
- **Loss Function as Preprocessing**: Choosing a suitable loss is equivalent to data preprocessing, activating more signal directions.

## Highlights & Insights
- The discovery of the **"Implicit Power Iteration + Early Stopping" mechanism** is the core theoretical contribution—gradient descent is not just an optimizer; its dynamics encode spectral methods. This provides a new perspective on why deep learning is effective.
- The **counter-intuitive conclusion that "$O(1)$ steps are insufficient"** challenges the DMFT-based analysis framework, suggesting that the true mechanism of neural network learning may differ from what simplified theories predict.
- **Treating the loss function as a preprocessor** bridges the gap between loss selection and feature extraction. In practice, loss functions can be chosen for their signal extraction properties rather than just their optimization attributes.

## Limitations & Future Work
- The **non-leapfrog assumption** is a key limitation—target functions with a generative leapfrog structure are not covered.
- Layer-wise training (W then a) rather than standard joint GD—while theoretically clearer, it differs from common practice.
- Requirement for sample splitting (two independent datasets)—a cost of theoretical simplicity.
- Analysis is limited to two-layer networks—extension to deep networks is needed.
- Constant $r$ (subspace dimension) and $p$ (polynomial degree)—high-dimensional scaling is not covered.
- Leading-order optimality is achieved, but constants are not sharpened—the constant gap with optimal spectral methods remains unquantified.

## Related Work & Insights
- **vs. Damian et al. 2022**: First Multi-Index NN learning result but required $d^2$ samples; ours reduces this to $d$—a leading-order improvement.
- **vs. Mousavi-Hosseini & Wu 2026**: Concurrent work achieving $O(d)$ samples but only weak recovery of incomplete subspaces; ours proves full recovery and end-to-end learning.
- **vs. Spectral Methods (DLB25)**: Info-theoretically optimal but not neural networks; ours proves NNs can match the efficiency of spectral methods.
- **vs. DMFT/State Evolution**: Those require $O(1)$ training steps or warm starts; ours proves optimal learning requires divergent steps.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First proof of NNs reaching info-theoretic optimality for Multi-Index learning; high theoretical depth.
- Experimental Thoroughness: ⭐⭐ Purely theoretical work, no experimental validation (though theoretical completeness makes it unnecessary).
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear problem positioning, though notation density is high.
- Value: ⭐⭐⭐⭐⭐ A milestone contribution to neural network feature learning theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](../../NeurIPS2025/optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[ICLR 2026\] Unifying Formal Explanations: A Complexity-Theoretic Perspective](unifying_formal_explanations_a_complexity-theoretic_perspective.md)
- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] Multi-Action Self-Improvement for Neural Combinatorial Optimization](multi-action_self-improvement_for_neural_combinatorial_optimization.md)
- [\[ICLR 2026\] High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes](high-dimensional_limit_theorems_for_sgd_momentum_and_adaptive_step-sizes.md)

</div>

<!-- RELATED:END -->
