---
title: >-
  [Paper Note] SAFE: Finding Sparse and Flat Minima to Improve Pruning
description: >-
  [ICML2025][Model Compression][Network Pruning] Models the pruning problem as a sharpness-aware optimization problem under sparse constraints, solved via the Alternating Direction Method of Multipliers (ADMM), simultaneously achieving sparsity and flat minima to improve generalization performance and robustness of pruned networks.
tags:
  - "ICML2025"
  - "Model Compression"
  - "Network Pruning"
  - "Sparse Optimization"
  - "Flat Minima"
  - "SAM"
  - "ADMM"
  - "LLM Pruning"
date: 2026-05-08
content_hash: d8838dd9f7e66ecc
---

# SAFE: Finding Sparse and Flat Minima to Improve Pruning

**Conference**: ICML2025  
**arXiv**: [2506.06866](https://arxiv.org/abs/2506.06866)  
**Code**: JAX & PyTorch (code included in the paper)  
**Area**: Model Pruning / Model Compression  
**Keywords**: Network Pruning, Sparse Optimization, Flat Minima, SAM, ADMM, LLM Pruning

## TL;DR
Models the pruning problem as a sharpness-aware optimization problem under sparse constraints, solved via the Alternating Direction Method of Multipliers (ADMM), simultaneously achieving sparsity and flat minima to improve generalization performance and robustness of pruned networks.

## Background & Motivation

**Problem**: Neural network pruning is typically accompanied by unavoidable performance degradation, and recovering the original performance remains challenging despite significant recent efforts.

**Key Observations**:

- Generalization performance is closely related to the flatness of the loss landscape (Keskar et al., 2017; Jiang et al., 2020)
- SAM (Sharpness-Aware Minimization) improves generalization by explicitly regularizing sharpness, showing outstanding performance across multiple domains
- Existing works introducing sharpness awareness into pruning (such as CrAM, Na et al.) are only loose heuristic combinations, and have not yet systematically integrated them from an optimization perspective

**Motivation**: Can sharpness-aware minimization and sparsity constraints be integrated more organically to find **both sparse and flat minima** through a rigorous optimization framework?

## Method

### Problem Formulation

Pruning is formulated as a min-max optimization problem under sparse constraints:

$$\min_{\|x\|_0 \leq d} \max_{\|\epsilon\|_2 \leq \rho} f(x + \epsilon)$$

- Outer minimization: Finding the optimal weight $x$ under sparse constraints
- Inner maximization: Finding the maximum loss within the $\epsilon$-neighborhood (i.e., pursuing flatness)
- $d$: Number of retained parameters; $\rho$: Perturbing radius

### ADMM Optimization Framework

**Variable Splitting**: Introducing an auxiliary variable $z$ to decouple sparse constraints from the objective optimization:

$$\min_{x,z} \max_{\|\epsilon\|_2 \leq \rho} f(x+\epsilon) + I_{\|\cdot\|_0 \leq d}(z) \quad \text{s.t. } x = z$$

**Augmented Lagrangian**: Adding the penalty term $\frac{\lambda}{2}\|x-z\|_2^2$ yields a three-step alternating iteration:

1. **$x$-update (Sharpness-aware gradient descent)**:
$$x_k^{(t+1)} = x_k^{(t)} - \eta^{(t)} \left[ \nabla f\left(x_k^{(t)} + \rho \frac{\nabla f(x_k^{(t)})}{\|\nabla f(x_k^{(t)})\|_2}\right) + \lambda(x_k^{(t)} - z_k + u_k) \right]$$

2. **$z$-update (Hard thresholding projection)**:
$$z_{k+1} = \text{proj}_{\|\cdot\|_0 \leq d}(x_{k+1} + u_k)$$

3. **$u$-update (Dual variable ascent)**:
$$u_{k+1} = u_k + x_{k+1} - z_{k+1}$$

### Safe⁺: Generalized Projection Extension

Introducing a positive definite diagonal matrix $\mathbf{P}$ instead of Euclidean distance allows the integration of various advanced saliency scores:

| Choice of $\mathbf{P}$ | Corresponding Pruning Method |
|---|---|
| $\mathbf{I}$ (Identity matrix) | Magnitude pruning (Original Safe) |
| $\text{diag}(\nabla^2 f(x))$ | OBD (Optimal Brain Damage) |
| $\text{diag}(\nabla f \cdot \nabla f^\top)$ | SNIP (Gradient sensitivity) |
| $\text{diag}(\mathbf{A}^\top \mathbf{A})$ | Wanda (Activation-aware) |

### Convergence Guarantees

The paper proves that under standard assumptions (lower bounded, $\beta$-smooth, $\mu$-weakly convex):
- The $x$-update sequence converges to a stationary point of the augmented Lagrangian (Lemma 3.5)
- Safe globally converges to a $\delta$-stationary point of the sparse-constrained optimization problem (Corollary 3.6)

### Practical Tips

- The penalty parameter $\lambda$ is gradually increased from 0 to the target value using a cosine schedule, reducing constraint interference in the early stages of training.

## Key Experimental Results

### Image Classification (CIFAR-10/100, Pruning during Training)

On VGG-19 and ResNet-20/32, Safe outperforms baselines such as PBW, GMP, LTH, ADMM, and MLPrune under most sparsity settings. The advantage is more pronounced at extreme sparsity levels (99.5%). No additional retraining is required; only BN tuning is used.

### LLM Post-Training Pruning (Perplexity ↓)

| Model | Sparsity | SparseGPT | Wanda | ALPS | Safe | Safe⁺ |
|---|---|---|---|---|---|---|
| LLaMA-2 7B | 50% | 6.99/9.20 | 6.92/9.23 | 6.87/8.98 | 6.78/8.93 | **6.56/8.71** |
| LLaMA-2 7B | 60% | 10.19/12.86 | 10.75/13.87 | 9.55/11.24 | 9.20/11.51 | **8.30/10.59** |
| LLaMA-2 13B | 50% | 6.06/8.20 | 5.98/8.28 | 5.96/8.09 | 5.76/7.85 | **5.67/7.74** |
| LLaMA-3 8B | 50% | 9.36/13.96 | 9.71/14.88 | 9.05/13.40 | 9.59/14.60 | **8.62/13.26** |

Safe⁺ surpasses SOTA baselines on all models and all sparsity levels (50%/60%/4:8/2:4).

### Noise Robustness (ResNet-20 on CIFAR-10)

| Noise Ratio | Sparsity | ADMM | Safe |
|---|---|---|---|
| 25% | 70% | 77.00 | **90.58** |
| 50% | 70% | 59.18 | **86.51** |
| 75% | 70% | 32.62 | **67.01** |

Safe achieves **+10% to +30%** higher accuracy than ADMM under label noise. It also performs better under common CIFAR-10C corruptions and PGD adversarial attacks.

## Highlights & Insights

1. **Novel Optimization Perspective**: It is the first to strictly combine sharpness awareness (SAM) with sparse constraints via the augmented Lagrangian framework, rather than heuristic combination.
2. **Theoretical Guarantees**: It provides a complete convergence proof, unlike most pruning methods which are solely based on intuition.
3. **Unified Safe⁺ Framework**: The generalized projection matrix $\mathbf{P}$ incorporates classical methods like OBD, SNIP, and Wanda into a unified framework.
4. **Outstanding Robustness**: Shows robustness surpassing baselines against label noise, image corruptions, and adversarial attacks.
5. **Computational Efficiency**: It is 2.54 times faster than ALPS (which is also ADMM-based) and requires no image classification retraining.

## Limitations & Future Work

1. **Image Classification Evaluated Only on Small Models**: CIFAR-10/100 + VGG/ResNet, without testing in-training pruning on ImageNet or larger-scale models.
2. **Limitations in LLM Experiments**: Safe⁺'s LLM pruning relies on layer-wise reconstruction error minimization rather than end-to-end optimization.
3. **Limited Structural Sparsity**: The primary experiments focus on unstructured pruning (50%/60%) and semi-structured pruning (2:4/4:8), without involving channel-level pruning.
4. **Sensitivity to $\lambda$ Scheduling**: The penalty parameter scheduling strategy requires extra hyperparameter tuning; the paper uses a cosine schedule but lacks a thorough analysis of other strategies.
5. **Overhead of Second-order Information**: Although Safe⁺ delivers good performance utilizing information such as the Hessian diagonal, it introduces additional computational overhead.

## Related Work & Insights

- **SAM** (Foret et al., 2021): The foundational method for sharpness-aware minimization.
- **ADMM Pruning** (Zhang et al., 2018): Pioneered ADMM-based pruning, upon which Safe incorporates a flatness objective.
- **CrAM** (Peste et al., 2022): Sharpness-aware minimization for compression, heuristically combining SAM and pruning.
- **Wanda** (Sun et al., 2024): Activation-aware LLM pruning, unified by Safe⁺ into the generalized projection.
- **ALPS** (Meng et al., 2024): Also an ADMM-based LLM pruning method, over which Safe demonstrates advantages in both performance and efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ — The optimization formulation is novel, but the individual components (SAM+ADMM) already exist.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive across images, LLMs, and robustness, but lacks large-scale in-training pruning experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear derivations, complete theoretical proofs, and well-organized experiments.
- Value: ⭐⭐⭐⭐ — Provides a more theoretically grounded optimization framework for pruning; the unified perspective of Safe⁺ is highly inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sparse Spectral Training and Inference on Euclidean and Hyperbolic Neural Networks](sparse_spectral_training_and_inference_on_euclidean_and_hyperbolic_neural_networ.md)
- [\[ACL 2025\] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](../../ACL2025/model_compression/outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)
- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](random_initialization_of_gated_sparse_adapters.md)
- [\[NeurIPS 2025\] SpecAttn: Speculating Sparse Attention](../../NeurIPS2025/model_compression/specattn_speculating_sparse_attention.md)
- [\[ACL 2026\] WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling](../../ACL2026/model_compression/wisca_a_lightweight_model_transition_method_to_improve_llm_training_via_weight_s.md)

</div>

<!-- RELATED:END -->
