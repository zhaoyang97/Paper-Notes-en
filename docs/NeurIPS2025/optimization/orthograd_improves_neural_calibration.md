---
title: >-
  [Paper Note] OrthoGrad Improves Neural Calibration
description: >-
  [NeurIPS 2025][Optimization][gradient orthogonalization] This paper presents the first systematic study of OrthoGrad (⊥Grad)—a geometrically constrained optimization method that projects gradients layer-wise onto directions orthogonal to the weights—for neural network calibration. Experiments on CIFAR-10 in low-data regimes demonstrate that OrthoGrad significantly improves calibration metrics (entropy, loss, confidence) without degrading accuracy…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "gradient orthogonalization"
  - "calibration"
  - "uncertainty estimation"
  - "overconfidence"
  - "geometric optimization"
date: 2026-05-08
content_hash: 41b9c18751feb7d3
---

# OrthoGrad Improves Neural Calibration

**Conference**: NeurIPS 2025
**arXiv**: [2506.04487](https://arxiv.org/abs/2506.04487)  
**Code**: None  
**Area**: Optimization
**Keywords**: gradient orthogonalization, calibration, uncertainty estimation, overconfidence, geometric optimization

## TL;DR

This paper presents the first systematic study of OrthoGrad (⊥Grad)—a geometrically constrained optimization method that projects gradients layer-wise onto directions orthogonal to the weights—for neural network calibration. Experiments on CIFAR-10 in low-data regimes demonstrate that OrthoGrad significantly improves calibration metrics (entropy, loss, confidence) without degrading accuracy, and the paper establishes convergence guarantees for a simplified variant under standard assumptions.

## Background & Motivation

**Background**: Reliable predictive confidence is essential for deploying neural networks in practice—when a model says "90% confident," it should indeed be correct 90% of the time. However, Guo et al. (2017) revealed through temperature scaling that modern deep networks, despite high accuracy, suffer from severe overconfidence.

**Limitations of Prior Work**: Existing calibration techniques fall into two categories. Intrinsic methods (e.g., focal loss, mixup) modify the loss function or data augmentation during training; post-hoc methods (e.g., temperature scaling, Platt scaling) adjust the output distribution after training. Post-hoc methods require an additional validation set and cannot fundamentally correct the model's internal uncertainty misestimation—they merely "patch" the output.

**Key Challenge**: The source of overconfidence lies in the optimization process itself. Standard gradient descent offers two pathways to reduce loss: (a) improving decision boundaries for better classification, or (b) simply inflating logit magnitudes to boost confidence. Pathway (b) reduces cross-entropy loss without improving generalization, leading to overconfidence. No existing method addresses this issue from the geometric structure of the optimization trajectory.

**Goal**: Can geometric constraints on gradients block the "confidence inflation" shortcut, forcing the optimizer to focus on improving decision boundaries?

**Key Insight**: Prieto et al. (2025) proposed OrthoGrad to stabilize training near grokking. This paper observes that orthogonalized gradients precisely block the confidence-scaling pathway in positive homogeneous networks, making OrthoGrad a natural candidate for improving calibration.

**Core Idea**: By projecting gradients onto directions orthogonal to the weights, OrthoGrad prevents the optimizer from inflating confidence through weight norm amplification, forcing loss reduction to occur exclusively via improved decision boundaries.

## Method

### Overall Architecture

OrthoGrad is an optimizer-agnostic gradient modification method. It wraps around any base optimizer (SGD, Adam, etc.) and projects gradients into the subspace orthogonal to the current weight vector before each parameter update. The pipeline is: standard forward pass → gradient computation → layer-wise gradient orthogonalization → (optional renormalization) → parameter update with modified gradients. No modifications to network architecture, loss function, or training data are required.

### Key Designs

1. **Gradient Orthogonalization Projection**:

    - Function: Removes the component of the gradient along the weight direction, retaining only the orthogonal component.
    - Mechanism: For parameter $\theta$ and gradient $\nabla L(\theta)$, the orthogonalized gradient is $g = \nabla L(\theta) - \frac{\langle \nabla L(\theta), \theta \rangle}{\|\theta\|^2} \theta$. This is a standard vector projection—subtracting from the gradient its projection onto the weight direction. The operation is applied layer-wise: each layer's gradient is projected onto the direction orthogonal to that layer's weights.
    - Design Motivation: In positive homogeneous networks (e.g., ReLU networks), the radial component of the gradient (parallel to the weights) corresponds to scaling the weight norm (inflating confidence), while the orthogonal component corresponds to rotating the decision boundary. Removing the radial component cuts off the "confidence inflation" pathway.

2. **Gradient Renormalization**:

    - Function: Restores the magnitude of the gradient after orthogonalization to maintain a reasonable update step size.
    - Mechanism: Projection shortens the gradient vector (by removing a component). Renormalization is defined as $\hat{g} = \frac{\|\nabla L(\theta)\|}{\|g\| + \epsilon} g$, rescaling the orthogonalized gradient to the magnitude of the original gradient. $\epsilon$ is a numerical stability constant.
    - Design Motivation: Without renormalization, the orthogonalized gradient may be much smaller than the original, reducing the effective learning rate and slowing training. Renormalization preserves practical optimization efficiency at the cost of losing theoretical convergence guarantees.

3. **Theoretical Analysis for Positive Homogeneous Networks**:

    - Function: Provides a theoretical explanation for why OrthoGrad improves calibration.
    - Mechanism: For the non-renormalized variant, convergence to a point where $\nabla L(\theta^*)$ is parallel to $\theta^*$—i.e., where the orthogonal component is zero—is proven under standard assumptions (bounded-below loss, Lipschitz-continuous gradients). In positive homogeneous networks, this implies that further loss reduction is only possible by scaling weights (inflating confidence) rather than changing the decision boundary. In other words, OrthoGrad stops at a stationary point that is "optimal for the decision boundary."
    - Design Motivation: Establishes a theoretical foundation for the anti-overconfidence mechanism. The renormalized variant lacks convergence guarantees, but if it converges, it must reach a standard stationary point. No significant empirical difference is observed between the two variants.

### Loss & Training

OrthoGrad does not modify the loss function; standard cross-entropy is used. The training protocol is identical to the baseline: SGD optimizer, learning rate 0.01, momentum 0.9, weight decay 5e-4, batch size 64, random flip and crop augmentation. The only difference is the insertion of an orthogonalization step after gradient computation. Computational overhead is negligible (a single projection operation per layer).

## Key Experimental Results

### Main Results: CIFAR-10, ResNet18, 10% labeled data, 20 seeds

| Metric | SGD | OrthoGrad | Effect Size | p-value |
|--------|-----|-----------|-------------|---------|
| Top-1 Accuracy | 75.18 | 75.27 | -0.05 | 0.86 |
| Test Loss | 1.26 | **1.19** | 0.64 | **0.05** |
| ECE | 0.168 | 0.161 | 0.48 | 0.14 |
| Predictive Entropy | 0.208 | **0.224** | -1.11 | **0.001** |
| Max Softmax | 0.920 | **0.914** | 1.06 | **0.002** |
| Max Logit | 13.58 | **13.03** | 1.52 | **2.5e-5** |
| Logit Variance | 45.73 | **42.30** | 2.00 | **2e-7** |

### Ablation Study

| Configuration | Key Finding | Notes |
|---------------|-------------|-------|
| Renormalization vs. no renormalization | No significant difference | 10-seed comparison; core benefit of geometric constraint is independent of renormalization |
| After temperature scaling | No difference in ECE/Brier | OrthoGrad requires lower temperature (2.66 vs. 2.80, p=0.003), indicating better intrinsic calibration |
| WideResNet-28-10 | Consistent improvement | Loss (p=0.004) and entropy (p=2e-4) both significantly improved; architecture-agnostic |
| 1000-epoch overfitting | Greater corruption robustness | CIFAR-10C average accuracy: OrthoGrad 60.4% vs. SGD 59.0% |
| CIFAR-10C corruption | Sustained improvement | Improvements in loss and entropy persist across all corruption severities |
| Weight norm comparison | 79.72 vs. 79.69 | p=0.36; OrthoGrad does not operate via weight norm regularization |

### Key Findings

- **Accuracy is completely unaffected**: No statistically significant difference in Top-1/Top-5 accuracy across all experiments.
- **Confidence is systematically reduced**: Max Softmax, Max Logit, and Logit Variance all decrease significantly, with large effect sizes (Cohen's $d > 1$).
- **Not implicit regularization**: Final weight norms show no difference; calibration improvement is not achieved by constraining weight growth.
- **Compatible with post-hoc methods**: ECE is comparable after temperature scaling, but OrthoGrad requires a lower temperature, indicating better intrinsic calibration.
- **Greater advantage in overfitting regimes**: In the 1000-epoch experiment, OrthoGrad surpasses SGD in accuracy at high corruption severities.

## Highlights & Insights

- **Minimalist and elegant design**: A single formula (gradient projection) achieves systematic calibration improvement without modifying the architecture, loss function, or data. This "minimal intervention" philosophy is highly appealing—when a geometric constraint suffices, complex regularization or training tricks are unnecessary.
- **Bridging theory and practice**: Although the convergence proof applies only to the simplified variant, the analysis of positive homogeneous networks clearly explains *why* orthogonalization improves calibration—by blocking the confidence inflation pathway. This explanation is intuitively transparent, even where the formal proof has a gap.
- **Calibration research from an optimizer perspective**: This work opens a new direction of "improving calibration via geometric constraints on the optimization trajectory." Prior work either modifies the loss (intrinsic methods) or the output (post-hoc methods); modifying the gradient direction itself had not been explored.

## Limitations & Future Work

- **Dataset scope**: Validation is limited to CIFAR-10/CIFAR-10C; ImageNet, NLP tasks, and more realistic application scenarios are not examined. While the low-data (10% labeled) setting is meaningful, it remains to be verified whether the improvements hold in the full-data regime.
- **Theory–practice gap**: The convergence proof applies only to the non-renormalized variant, while experiments use the renormalized variant. Although no significant empirical difference is observed, this gap requires theoretical resolution.
- **ECE improvement not significant**: Despite significant improvements in confidence metrics, the most central calibration metric, ECE (p=0.14), does not reach statistical significance. Larger sample sizes or more complex tasks may be needed.
- **Extended training results based on a single seed**: The "interesting" 1000-epoch results derive from a single seed and lack statistical reliability.
- **Lack of large-scale validation**: Scalability to larger models (ResNet-50+) and larger datasets remains unverified.

## Related Work & Insights

- **vs. Temperature Scaling (Guo et al. 2017)**: Temperature scaling adjusts outputs post-training; OrthoGrad constrains gradients during training. The two are complementary—OrthoGrad improves intrinsic calibration, while temperature scaling provides further fine-tuning. Experiments confirm that OrthoGrad does not compromise the effectiveness of post-hoc calibration.
- **vs. Focal Loss (Mukhoti et al. 2020)**: Focal loss improves calibration by reweighting the loss function and requires tuning the hyperparameter $\gamma$. OrthoGrad does not modify the loss function and represents an orthogonal approach; in principle, the two methods can be combined.
- **vs. Orthogonal Gradient for Grokking (Prieto et al. 2025)**: OrthoGrad was originally designed to stabilize training near grokking. This paper finds that the same geometric constraint is also effective for calibration, suggesting that gradient orthogonalization may have broader regularization effects.

## Rating

- Novelty: ⭐⭐⭐⭐ Opens a new perspective of "improving calibration via geometric constraints on the optimization trajectory," though OrthoGrad itself is not a new method.
- Experimental Thoroughness: ⭐⭐⭐ Statistical analysis is rigorous (20 seeds, effect sizes, p-values), but the dataset and model scales are limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical motivation is clear, experimental presentation is well-structured, and the theory–practice gap is discussed candidly.
- Value: ⭐⭐⭐⭐ The method is simple and practical, can be immediately integrated into existing training pipelines, and the research direction is promising.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](learning_provably_improves_the_convergence_of_gradient_descent.md)
- [\[NeurIPS 2025\] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization](dartquant_efficient_rotational_distribution_calibration_for_llm_quantization.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](../../ICLR2026/optimization/towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] FedMC: Federated Manifold Calibration](../../ICLR2026/optimization/fedmc_federated_manifold_calibration.md)
- [\[CVPR 2025\] Test-Time Augmentation Improves Efficiency in Conformal Prediction](../../CVPR2025/optimization/test-time_augmentation_improves_efficiency_in_conformal_prediction.md)

</div>

<!-- RELATED:END -->
