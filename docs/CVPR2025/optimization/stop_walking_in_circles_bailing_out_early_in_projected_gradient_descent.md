---
title: >-
  [Paper Note] Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent
description: >-
  [CVPR 2025][Optimization][PGD Attack] It is discovered that the PGD attack exhibits cyclic behavior on the $L_\infty$ ball for robust samples. Detecting cycles via hashing (PGD_CD) enables early stopping, which achieves an iteration reduction of up to 96% while maintaining identical robustness evaluation results.
tags:
  - "CVPR 2025"
  - "Optimization"
  - "PGD Attack"
  - "Cycle Detection"
  - "Adversarial Robustness Evaluation"
  - "Early Stopping"
  - "L∞-constraint"
date: 2026-05-08
content_hash: 7e1e792db64727be
---

# Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent

**Conference**: CVPR 2025  
**arXiv**: [2503.19347](https://arxiv.org/abs/2503.19347)  
**Code**: None  
**Area**: Optimization / Adversarial Robustness  
**Keywords**: PGD Attack, Cycle Detection, Adversarial Robustness Evaluation, Early Stopping, L∞-constraint

## TL;DR

It is discovered that the PGD attack exhibits cyclic behavior on the $L_\infty$ ball for robust samples. Detecting cycles via hashing (PGD_CD) enables early stopping, which achieves an iteration reduction of up to 96% while maintaining identical robustness evaluation results.

## Background & Motivation

**Background**: PGD (Projected Gradient Descent) is the standard method for evaluating adversarial robustness, with current best practices recommending $\geq 1000$ iterations to generate adversarial examples. For the ImageNet test set as an example, a 1000-iteration PGD evaluation requires a computational budget equivalent to 100 training epochs.

**Limitations of Prior Work**: PGD is computationally expensive, yet most computations are wasted: for vulnerable images where adversarial examples are already found, typically 3 iterations suffice; for truly robust images, PGD loops infinitely on the boundary without ever succeeding.

**Key Challenge**: PGD utilizes a fixed step size $\alpha$ and sign gradients. When an image is truly robust, the projection operator on the $L_\infty$ ball boundary repeatedly pushes the perturbation back to the same positions, forming a deterministic cycle. However, standard PGD cannot detect this cycle and must complete all iterations.

**Goal**: To detect cyclic behavior in PGD, enabling safe early stopping while guaranteeing identical attack results.

**Key Insight**: Leveraging the geometric properties of $L_\infty$ projection: fixed step size + sign gradient + boundary projection = deterministic iteration = cycles must produce fixed states. Once $\delta^{(i)}$ is detected to have appeared before, subsequent iterations are guaranteed to repeat entirely.

**Core Idea**: Utilizing a hash set to detect cycles in PGD perturbations, allowing safe early stopping once a cycle is detected, resulting in zero loss in attack strength and up to 96% reduction in computation.

## Method

### Overall Architecture

In each iteration of standard PGD, two checks are integrated: (1) Attack success check: if the current perturbation has already caused misclassification, return success immediately; (2) Cycle detection: if the hash of the current perturbation $\delta^{(i)}$ already exists in the set $\mathcal{S}$, return robust immediately (since subsequent iterations will inevitably repeat), otherwise store the hash in the set and continue iterating.

### Key Designs

1. **Theoretical Analysis of Cycle Formation**:

    - **Function**: Explains why PGD under $L_\infty$ constraints produces cycles for robust samples.
    - **Mechanism**: PGD updates via $\delta^{(i)} = \mathcal{P}_\mathcal{B}(\delta^{(i-1)} + \alpha \cdot \text{sign}(\nabla_X \mathcal{L}))$. Because of the sign function, the gradient direction is discretized into $\{-1, +1\}^d$; due to the fixed step size $\alpha$, the displacement of each step is deterministic; and since the projection onto the $L_\infty$ ball is coordinate-wise clipping, the projected points on the boundary are also deterministic. The combination of these conditions restricts the state space, making cycles inevitable.
    - **Design Motivation**: Unlike Auto-PGD which uses adaptive step sizes and momentum, the fixed step size property of standard PGD can be leveraged directly, turning a drawback into an advantage.

2. **Efficient Hash Detection**:

    - **Function**: Detects cycles in approximately constant time, avoiding the storage of full perturbation tensors.
    - **Mechanism**: Instead of storing the complete $\delta^{(i)}$ (which is memory-intensive), the hash is computed as $\text{hash}(\text{torch.frexp}(h^\top \delta^{(i)}))$ where $h$ is a random vector. `torch.frexp()` is utilized to prevent floating-point denormalization issues. Each iteration requires only one vector inner product and one hash lookup.
    - **Design Motivation**: Computed directly on the GPU with negligible memory overhead, ensuring no impact on PGD batch efficiency.

3. **Early Success/Failure Detection**:

    - **Function**: Computes acceleration for vulnerable samples as well (not just robust ones).
    - **Mechanism**: Checks whether the current perturbation has already caused misclassification after each iteration. Experiments show that the median number of iterations to succeed for vulnerable samples is only 3, while standard PGD would continue iterating up to 1000.
    - **Design Motivation**: Dual-ended acceleration: vulnerable samples exit quickly upon success, and robust samples exit quickly upon failure detected via cycles.

### Loss & Training

The standard cross-entropy loss $\max_\delta \mathcal{L}(f(X+\delta), y)$ is minimized subject to $\|\delta\|_\infty \leq \epsilon$. The step size is set to $\alpha = \epsilon/4$. This method does not modify any optimization objectives, only adding the cycle detection termination condition during iteration.

## Key Experimental Results

### Main Results

PGD vs PGD_CD Comparison (1000 iteration budget, RobustBench models):

| Dataset | Model | PGD Robust Acc | PGD_CD Robust Acc | PGD Iterations | PGD_CD Iterations | Speedup |
|--------|------|-------------|----------------|------------|---------------|--------|
| ImageNet | ConvNeXt-L | 58.15% | 58.15% | 29.1M | 2.7M | **90.79%** |
| ImageNet | Swin-L | 59.27% | 59.27% | 29.4M | 2.6M | **91.00%** |
| CIFAR-10 | XCiT-L12 | 59.05% | 59.05% | 5.9M | 0.27M | **95.36%** |
| CIFAR-10 | R18_ddpm | 59.61% | 59.61% | 6.0M | 0.22M | **96.38%** |
| CIFAR-100 | XCiT-L12 | 38.93% | 38.93% | 3.9M | 0.17M | **95.63%** |

**Key**: Robust accuracy is completely identical, with zero precision loss.

### Ablation Study

PGD_CD vs Auto-PGD Comparison (500 iterations):

| Model | APGD Acc | PGD_CD Acc | APGD Iterations | PGD_CD Iterations |
|------|-----------|-------------|-----------|-------------|
| CIFAR10-XCiT | 70.09% | 59.05% | 3.50M | **0.26M** |
| CIFAR100-XCiT | 38.46% | 38.93% | 1.92M | **0.16M** |

### Key Findings
- **Vulnerable samples are attacked extremely fast**: A median of only 3 iterations is required to find adversarial examples.
- **ImageNet achieves the most prominent acceleration**: Saving approximately 90%+ of iterations, as cycles occur more frequently in high-dimensional spaces.
- **PGD's fixed step size is not a drawback**: PGD_CD matches or even outperforms Auto-PGD (which uses adaptive step sizes) in attack success rates, while being an order of magnitude cheaper computationally.
- **No penalty in the worst-case scenario**: If no cycles are detected, PGD_CD behaves identically to standard PGD.

## Highlights & Insights

- **Turning a drawback into an advantage**: The fixed step size of PGD, criticized by Auto-PGD as a drawback leading to cycles, is leveraged in this work to enable early stopping through deterministic cycles. This paradigm can inspire other "defect-exploiting" optimizations.
- **Zero-risk acceleration**: Unlike other speedup methods, PGD_CD mathematically guarantees identical results to standard PGD—not approximately, but exactly identical. This theoretical guarantee allows it to unconditionally replace standard PGD.
- **Simple and practical**: The entire method can be implemented in about 10 lines of code, with no hyperparameter tuning and no changes to optimization objectives. This minimalist design reflects excellent engineering aesthetics.

## Limitations & Future Work

- **Applicable only to fixed-step PGD**: Methods using momentum or adaptive step sizes (such as Auto-PGD) do not exhibit identical cyclic patterns, rendering this approach directly inapplicable.
- **Limited speedup on certain CIFAR-100 models**: For some models, the savings are less than 2%. Although harmless as a "no-penalty failure", it limits generalizability.
- **Sensitivity to step size**: The step size $\alpha = \epsilon/4$ is not necessarily optimal for all models. Different step sizes may induce different cyclic behaviors.
- **$L_\infty$-norm only**: The theoretical foundation of cycle detection relies on the coordinate-wise clipping property of $L_\infty$ projection; whether similar patterns exist under $L_2$ or other norms remains unexplored.

## Related Work & Insights

- **vs Auto-PGD**: Auto-PGD employs adaptive step sizes and momentum to prevent loops and enhance attacks, whereas PGD_CD demonstrates that standard PGD's attack scale is already sufficient, and cycles can instead be exploited for acceleration. The philosophies of the two approaches are entirely opposite.
- **vs RobustBench Ensemble Attacks**: RobustBench uses an ensemble of multiple attacks to obtain tighter robustness upper bounds. PGD_CD can serve as a fast initial filter in such ensembles, screening out most vulnerable samples at an extremely low cost, before more expensive attacks are deployed on the remaining samples.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet profound observation, with an ingenious approach of turning a drawback into an advantage.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across multiple datasets and models, with thorough comparisons to Auto-PGD, though missing $L_2$-norm experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, rigorous orchestration, and highly convincing charts and figures.
- Value: ⭐⭐⭐⭐ Directly valuable and practical for the adversarial robustness community, offering a plug-and-play drop-in replacement for standard PGD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression](../../ICML2025/optimization/benefits_of_early_stopping_in_gradient_descent_for_overparameterized_logistic_re.md)
- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](../../ICML2025/optimization/quantum_optimization_via_gradient-based_hamiltonian_descent.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[ICLR 2026\] Corner Gradient Descent](../../ICLR2026/optimization/corner_gradient_descent.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](../../NeurIPS2025/optimization/optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)

</div>

<!-- RELATED:END -->
