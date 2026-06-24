---
title: >-
  [Paper Note] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics
description: >-
  [CVPR2025][LLM Pretraining][Neural Tangent Kernel] Reveals through the NTK framework that linearized attention mechanisms do not converge to the infinite-width NTK limit (the spectral amplification effect cubes the condition number of the Gram matrix, requiring a width of $m = \Omega(\kappa^6)$), and introduces the concept of "influence malleability" to quantify the dual consequences of this non-convergence: an attention network's malleability…
tags:
  - "CVPR2025"
  - "LLM Pretraining"
  - "Neural Tangent Kernel"
  - "Attention Mechanism"
  - "Kernel Methods"
  - "Influence Function"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: 6a959a259e9f1748
---

# Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics

**Conference**: CVPR2025  
**arXiv**: [2603.13085](https://arxiv.org/abs/2603.13085)  
**Code**: To be confirmed  
**Area**: Deep Learning Theory  
**Keywords**: Neural Tangent Kernel, Attention Mechanism, Kernel Methods, Influence Function, Adversarial Robustness

## TL;DR

Reveals through the NTK framework that linearized attention mechanisms do not converge to the infinite-width NTK limit (the spectral amplification effect cubes the condition number of the Gram matrix, requiring a width of $m = \Omega(\kappa^6)$), and introduces the concept of "influence malleability" to quantify the dual consequences of this non-convergence: an attention network's malleability, which is 6-9 times higher than that of a ReLU network, both enhances task adaptability and exacerbates adversarial vulnerability.

## Background & Motivation

**Basic Prediction of NTK Theory**: Neural Tangent Kernel (NTK) theory predicts that sufficiently wide networks maintain an approximately constant kernel during training (lazy training), allowing precise analysis of learning dynamics using kernel methods.

**Lack of Theoretical Characterization for Attention**: The nonlinear dynamics of attention prevent it from fitting neatly into the NTK theoretical framework, leaving its flexibility during the learning process without rigorous theoretical characterization.

**Surprising Empirical Findings**: Standard ReLU networks exhibit a monotonically decreasing NTK distance as width increases (as expected), but attention-based networks show an increasing or non-monotonic NTK distance, indicating that attention never enters the kernel regime.

**Core Problem**: Why does attention not converge to the NTK limit? What does this non-convergence imply for training data dependence?

**Key Insight**: Designing a parameter-free linearized attention $f^{\text{att}}(\mathbf{X}) = \mathbf{X}\mathbf{X}^T\mathbf{X}$ and establishing a precise correspondence with the data-dependent Gram-induced kernel, thereby enabling rigorous theoretical analysis.

## Method

### Core Theoretical Framework

1. **Definition of Linearized Attention**: $f^{\text{att}}(\mathbf{X}) = \mathbf{X}\mathbf{X}^T\mathbf{X}$, which corresponds to standard attention with identity QKV projections and linearized softmax ($\exp(A_{ij}) \approx 1 + A_{ij}$). It retains the core quadratic interaction structure of attention, equivalent to an unnormalized Nadaraya-Watson estimator.
2. **MLP-Attn Architecture**: Linearized attention preprocessing (with $\ell_2$ normalized output) $\to$ two-layer ReLU MLP ($f = \frac{1}{\sqrt{m}} \sum_r a_r \sigma(\mathbf{w}_r^T \tilde{\mathbf{x}})$). The attention layer is parameter-free, and only the MLP weights $\mathbf{w}_r$ are trained, while $a_r \in \{-1, +1\}$ remain fixed.

### Key Theorems

1. **Theorem 4.1 (Data-Dependent Gram-Induced Kernel)**: The linearized attention-induced kernel is $K_{\text{LinAttn}}(\mathbf{x}_i, \mathbf{x}_j) = \sum_{k,\ell} (\mathbf{x}_i^T \mathbf{x}_k)(\mathbf{x}_k^T \mathbf{x}_\ell)(\mathbf{x}_\ell^T \mathbf{x}_j)$, or in matrix form $\mathbf{K} = \mathbf{G}^3$ ($\mathbf{G} = \mathbf{X}\mathbf{X}^T$). This exhibits a transitive similarity chain $i \to k \to \ell \to j$: influence propagates from $\mathbf{x}_i$ to $\mathbf{x}_j$ via intermediate points.
2. **Theorem 4.2 (Sequential Architecture NTK)**: The infinite-width limit NTK of MLP-Attn is $K_{\text{seq}}(\mathbf{x}, \mathbf{x}') = \mathbb{E}_{\mathbf{w}}[\sigma'(\mathbf{w}^T \tilde{\mathbf{x}}) \sigma'(\mathbf{w}^T \tilde{\mathbf{x}}')] \cdot \langle \tilde{\mathbf{x}}, \tilde{\mathbf{x}}' \rangle$. Since $f^{\text{att}}$ is parameter-free, only the gradients of $\mathbf{w}_r$ contribute.
3. **Theorem 4.7 (Spectral Amplification and NTK Non-Convergence)**: The attention transformation cubes the condition number $\kappa(\tilde{\mathbf{G}}) = \kappa(\mathbf{G})^3$. The width required for NTK convergence is $m = \Omega(\kappa(\mathbf{G})^6/\epsilon^2)$. On MNIST, $\kappa(\mathbf{G}) \approx 1.2 \times 10^3$ $\to$ requiring $m \gg 10^{18}$; on CIFAR-10, $\kappa(\mathbf{G}) \approx 8.7 \times 10^3$ $\to$ requiring $m \gg 10^{24}$, which vastly exceeds practical widths.
4. **Proposition 4.5 (Data-Dependent Kernel Sensitivity)**: The sensitivity of the attention kernel depends on the correlation structure of the entire dataset: $|K_{\text{LinAttn}}(\mathbf{x}_i + \delta, \mathbf{x}_j) - K_{\text{LinAttn}}(\mathbf{x}_i, \mathbf{x}_j)| \leq \|\mathbf{G}\mathbf{x}_j\|_1 \cdot \epsilon$, where $\|\mathbf{G}\mathbf{x}_j\|_1$ grows with dataset scale and correlation density. This is in contrast to the $O(\epsilon)$ data-independent sensitivity of polynomial kernels.

### Influence Malleability

- **Influence Function**: Based on the NTK-based leave-one-out formula $I(\mathbf{x}_i, \mathbf{x}_{\text{test}})$, which is efficiently computed using the empirical finite-width kernel matrix $(\mathbf{K}_m + \lambda \mathbf{I})^{-1}$ without retraining.
- **Flip Rate Definition**: Selecting the top-$\tau$ ($\tau=0.1$) highest-influence training samples, applying PGD adversarial perturbations ($\epsilon=0.3$), and calculating the ratio of flipped influence signs.
- **Complementary Metric**: Spearman's rank correlation coefficient $\rho$ between original and perturbed influence rankings; lower values indicate higher malleability.
- **Three Intervention Strategies**: Curated (removing high-influence samples), Transformed (replacing them with adversarial versions), and Adversarial (full-data PGD perturbations).

## Key Experimental Results

### NTK Non-Convergence Verification

| Model | Dataset | $m=16$ | $m=1024$ | $m=4096$ | Trend |
|------|--------|--------|----------|----------|------|
| 2L-ReLU | MNIST | 45.1 | 39.9 | 39.2 | ↓ Convergent |
| MLP-Attn | MNIST | 10.3 | 33.3 | 43.4 | ↑ Non-monotonic then Divergent |
| 2L-ReLU | CIFAR-10 | 246.2 | 101.7 | 56.9 | ↓ Convergent |
| MLP-Attn | CIFAR-10 | 3.7 | 10.4 | 12.6 | ↑ Monotonic Divergent |

The non-monotonic trend on MNIST vs the monotonic increase on CIFAR-10 reflects differences in the Gram matrix structures: the lower $\kappa(\mathbf{G})$ of MNIST allows a transient near-lazy regime at smaller widths, whereas CIFAR-10 remains in the feature learning regime from initialization.

### Influence Malleability (10-Class Classification, $\epsilon=0.3$)

| Dataset | Model | FGSM | PGD | MIM |
|--------|------|------|-----|-----|
| MNIST | 2L-ReLU | 4.1% | 3.3% | 3.4% |
| MNIST | MLP-Attn | **34.6%** | **28.9%** | **21.9%** |
| CIFAR-10 | 2L-ReLU | 3.3% | 3.1% | 3.2% |
| CIFAR-10 | MLP-Attn | **26.4%** | **19.1%** | **20.5%** |

The flip rate of MLP-Attn is 6-9 times higher than that of ReLU. FGSM yields the highest flip rates, while PGD produces the largest ratio (8.8$\times$) on MNIST.

### Binary Classification Scenarios ($\epsilon=0.3$)

| Dataset | Model | FGSM | PGD | MIM |
|--------|------|------|-----|-----|
| MNIST (3 vs 8) | 2L-ReLU | 8.4% | 8.4% | 8.6% |
| MNIST (3 vs 8) | MLP-Attn | **25.9%** | **41.0%** | **40.5%** |
| CIFAR-10 (cars vs planes) | 2L-ReLU | 15.2% | 15.5% | 15.3% |
| CIFAR-10 (cars vs planes) | MLP-Attn | 14.3% | 14.0% | 14.8% |

Attention enjoys a 3-5$\times$ advantage in MNIST binary classification; this advantage disappears in CIFAR-10 binary classification ($\approx 1\times$), which aligns with Theorem 4.7—the lower $\kappa(\mathbf{G})$ of binary CIFAR-10 weakens the cubic condition number amplification effect.

### Adversarial Training Analysis

| Dataset | Model | Standard Training | Adversarial Training |
|--------|------|----------|----------|
| MNIST | 2L-ReLU | 3.3% | 43.4% |
| MNIST | MLP-Attn | 28.9% | 42.2% |
| CIFAR-10 | 2L-ReLU | 3.1% | 36.5% |
| CIFAR-10 | MLP-Attn | 19.1% | 38.6% |

Adversarial training significantly boosts ReLU malleability (3.3% $\to$ 43.4%), whereas MLP-Attn inherently possesses high malleability under standard training (28.9%). Two distinct mechanisms generate malleability: (1) architectural—the attention's Gram-induced kernel fundamentally creates sensitivity; (2) training-induced—adversarial enhancement forces feature retraining. The sensitivity of attention is intrinsic rather than externally imposed.

## Highlights & Insights

1. **Theoretical Elegance**: Formulates a precise kernel correspondence ($\mathbf{K} = \mathbf{G}^3$) starting from linearized attention, and then explains non-convergence via spectral amplification, offering a complete causal chain.
2. **Novelty of "Influence Malleability"**: Translates the theoretical discovery of NTK non-convergence into a measurable practical indicator—dynamic variations in training data dependency.
3. **Deep Insight into Duality**: The exact same data-dependent kernel mechanism acts both as the source of attention's power (lowering approximation error when data-dependent kernels align with target and data distributions) and its source of vulnerability (high malleability facilitating adversarial manipulation).
4. **Precise Alignment Between Theory and Experiments**: The empirical condition number $\kappa(\mathbf{G}) \approx 10^3$ accurately predicts the observed non-convergence behavior; the disappearance of the advantage in binary classification scenarios where $\kappa(\mathbf{G})$ decreases further validates the theory.
5. **Generalizability Analysis**: Stacking $k$ layers of linearized self-attention yields $\mathbf{G}^{2k+1}$, resulting in a condition number of $\kappa(\mathbf{G})^{2k+1}$, which further exacerbates non-convergence; truncating attention (by retaining only the top-$r$ singular components) can restore convergence.

## Limitations & Future Work

- Only analyzes linearized attention (identity QKV + linearized softmax), leaving a gap with actual softmax attention; the row-wise normalization of softmax could potentially amplify non-convergence further.
- Experiments are limited to MNIST/CIFAR-10 and two-layer networks ($m \leq 4096$), without scaling up to larger architectures or more complex data.
- Theorem 4.7 only provides a lower bound on the NTK deviation at initialization, rather than directly predicting the post-training trajectory—the rise in NTK distance as width increases is a complementary effect (larger networks possess greater capacity for feature learning).
- The influence malleability metric depends on the specific choice of the perturbation budget $\epsilon$ and selection threshold $\tau$, though supplementary experiments confirm consistent rankings across different $\epsilon \in \{0.1, 0.2, 0.3, 0.5\}$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reveal the non-convergence of attention's NTK and its influence malleability consequences
- Experimental Thoroughness: ⭐⭐⭐ Rigorous theoretical validation but limited experimental scale
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and rigorous theoretical proofs with experiments tightly aligning with theory
- Value: ⭐⭐⭐⭐ Provides a fresh perspective on understanding the fundamental properties of attention mechanisms

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Towards Robust Influence Functions with Flat Validation Minima](../../ICML2025/llm_pretraining/towards_robust_influence_functions_with_flat_validation_minima.md)
- [\[CVPR 2025\] Robust Message Embedding via Attention Flow-Based Steganography](robust_message_embedding_via_attention_flow-based_steganography.md)
- [\[ICML 2025\] Benign Overfitting in Token Selection of Attention Mechanism](../../ICML2025/llm_pretraining/benign_overfitting_in_token_selection_of_attention_mechanism.md)
- [\[ICLR 2026\] OptimSyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation](../../ICLR2026/llm_pretraining/optimsyn_influence-guided_rubrics_optimization_for_synthetic_data_generation.md)
- [\[ICLR 2026\] Dual-objective Language Models: Training Efficiency Without Overfitting](../../ICLR2026/llm_pretraining/dual-objective_language_models_training_efficiency_without_overfitting.md)

</div>

<!-- RELATED:END -->
