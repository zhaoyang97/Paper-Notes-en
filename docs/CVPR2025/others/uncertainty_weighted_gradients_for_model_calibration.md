---
title: >-
  [Paper Note] Uncertainty Weighted Gradients for Model Calibration
description: >-
  [CVPR 2025][Model Calibration] By analyzing a unified framework of methods like Focal Loss, this work reveals that directly applying uncertainty weights to the loss function leads to a misalignment between gradients and uncertainty. Hence, the Uncertainty-GRA framework is proposed to apply uncertainty weights directly to gradients, using the Generalized Brier Score as a more precise uncertainty metric, achieving state-of-the-art calibration performance.
tags:
  - "CVPR 2025"
  - "Model Calibration"
  - "Uncertainty Weighting"
  - "Gradient Scaling"
  - "Brier Score"
  - "Focal Loss"
date: 2026-05-08
content_hash: 6a59a9344c82197d
---

# Uncertainty Weighted Gradients for Model Calibration

**Conference**: CVPR 2025  
**arXiv**: [2503.22725](https://arxiv.org/abs/2503.22725)  
**Code**: [https://github.com/Jinxu-Lin/BSCE-GRA](https://github.com/Jinxu-Lin/BSCE-GRA)  
**Area**: Model Calibration  
**Keywords**: Model Calibration, Uncertainty Weighting, Gradient Scaling, Brier Score, Focal Loss

## TL;DR

By analyzing a unified framework of methods like Focal Loss, this work reveals that directly applying uncertainty weights to the loss function leads to a misalignment between gradients and uncertainty. Hence, the Uncertainty-GRA framework is proposed to apply uncertainty weights directly to gradients, using the Generalized Brier Score as a more precise uncertainty metric, achieving state-of-the-art calibration performance.

## Background & Motivation

**Background**: Deep neural networks often yield overconfident or underconfident predictions in classification tasks, preventing the predictive confidence from accurately reflecting the true correctness probability. Existing calibration methods primarily include post-processing methods (Temperature Scaling, Platt Scaling), regularization techniques (Mixup, Label Smoothing), and loss function modifications (Focal Loss, Cross-Entropy + calibration regularization).

**Limitations of Prior Work**: Focal Loss and its variants (e.g., Dual Focal Loss) improve calibration by adjusting the loss function using sample-level weights, but they suffer from two key limitations: (1) The loss weight factor $u$ is differentiable; directly applying it disrupts the positive correlation between the CE loss value and its gradient, causing misalignment between gradient scaling and sample uncertainty. (2) Focal Loss only considers the ground-truth logit to estimate uncertainty, which lacks precision in multi-class scenarios.

**Key Challenge**: The ultimate goal is to enforce larger gradient updates for uncertain samples. However, the differentiable weight of Focal Loss causes its gradient weight function $g(p,\gamma)$ to monotonically increase with $p$ within the interval $[0, p_0]$, meaning moderately uncertain samples receive larger gradient weights than the most uncertain ones, which contradicts the design objective.

**Goal**: (1) Analyze the nature and deficiencies of existing calibration losses from a unified framework; (2) strictly align gradient scaling with sample uncertainty; (3) design a more precise uncertainty metric for multi-class classification.

**Key Insight**: The authors observe that the weight factor $(1-\hat{p}_c)^\gamma$ of Focal Loss is equivalent to the Brier Score in binary classification, and the weight of Dual Focal Loss is equivalent in tri-classification. This suggests the rationale of using the calibration metric itself as the weighting factor. However, the key lies in applying the weighting factor to the gradient rather than the loss.

**Core Idea**: Detach the gradient of the uncertainty estimator (e.g., Brier Score) and directly multiply it by the gradient of the CE loss (instead of multiplying it by the CE loss value), which ensures that gradient scaling is strictly and positively correlated with uncertainty. Meanwhile, a Generalized Brier Score is utilized instead of single-logit uncertainty estimation to leverage the probability outputs of all classes.

## Method

### Overall Architecture

The standard classification training workflow remains unchanged, except for modifying the gradient calculation of the loss function. The forward pass computes the softmax probability distribution $\hat{p}(x)$, the CE loss, and the uncertainty metric $u(\hat{p}(x))$. During backpropagation, the detached $u$ is multiplied by the CE gradient to form the final gradient, rather than taking the gradient of $u$ multiplied by the CE loss value.

### Key Designs

1. **Unified Loss Framework Analysis (Unified Loss Framework)**:

    - Function: Reveal the common nature of the Focal Loss family and clarify their limitations.
    - Mechanism: Unify FL and DFL into the form $\mathcal{L} = u \cdot CE$, where $u$ is the sample-level uncertainty estimator. For FL, $u_{FL} = (1-\hat{p}_c)^\gamma$, and for DFL, $u_{DFL} = (1-\hat{p}_c + \hat{p}_j)^\gamma$. The analysis reveals that the actual gradient scaling factor of FL is $g(p,\gamma) = (1-p)^\gamma - \gamma p(1-p)^{\gamma-1}\log(p)$, which increases in $[0, p_0]$ and decreases in $[p_0, 1]$, leading to incomplete alignment.
    - Design Motivation: Identify the true reason existing methods improve calibration (uncertainty weighting) and why they are sub-optimal (gradient misalignment + imprecise uncertainty estimation).

2. **Gradient-level Uncertainty Weighting (Uncertainty-GRA)**:

    - Function: Ensure gradient scaling is strictly and positively correlated with sample uncertainty.
    - Mechanism: Define the modified gradient as $\frac{\partial}{\partial\theta}\mathcal{L}_{U-GRA} = u(\hat{p}(x)) \cdot \frac{\partial}{\partial\theta}\mathcal{L}_{CE}$. In practice, this is implemented by detaching the gradient of $u$ and multiplying it by the CE loss. The corresponding implicit loss function has the form $\mathcal{L}_{U-GRA} = -\int u(\hat{p}) \cdot \frac{y}{\hat{p}} d\hat{p}$. Under SGD, $\theta_{t+1} = \theta + \alpha \cdot u(\hat{p}) \cdot \nabla_\theta \mathcal{L}_{CE}$, where highly uncertain samples receive larger parameter updates directly.
    - Design Motivation: Directly applying weights to the gradient instead of the loss completely avoids the disruption of the positive correlation caused by the extra gradient term of the differentiable weights.

3. **Generalized Brier Score as Uncertainty Metric (BSCE-GRA)**:

    - Function: Provide a more precise multi-class uncertainty estimation than FL/DFL.
    - Mechanism: Define the Generalized Brier Score as $u_{gBS} = \sum_{i=1}^{K} \|\hat{p}_i - y_i\|_\beta^\gamma$, where the standard BS corresponds to $\beta=2, \gamma=2$. The difference between the BS and the true calibration error, $c(x) - u_{BS}$, depends only on the true probability $\eta(x)$, which is constant for a specific sample. The weight of FL only changes along the $p_i$ axis, and DFL along the $p_i$ and $p_j$ axes, while BS responds to changes across all K axes, offering a more comprehensive uncertainty evaluation. The final formulation is BSCE-GRA = detach(BS) × CE.
    - Design Motivation: Visualizations in 4-class classification clearly demonstrate that $u_{FL}$ and $u_{DFL}$ are only sensitive to 1 or 2 dimensions, whereas $u_{BS}$ captures uncertainty fluctuations across all dimensions. Toy dataset experiments show that the Pearson correlation coefficient between gBS and the true uncertainty (0.664) is higher than that of DFL (0.638) and FL (0.550).

### Loss & Training

- **BSCE-GRA Loss**: Forward pass computes CE and BS. During backpropagation, $\nabla_\theta \mathcal{L} = \text{detach}(u_{BS}) \cdot \nabla_\theta \mathcal{L}_{CE}$.
- Extremely simple implementation: only requires adding one line for BS calculation and the detach operation to the standard CE training code.
- Can be combined with post-processing methods such as Temperature Scaling (TS).
- Few hyperparameters: standard BS with $\beta=2, \gamma=2$ works well, eliminating the need to tune $\gamma$ as in Focal Loss.

## Key Experimental Results

### Main Results (ECE ↓, 15 bins)

| Dataset | Model | CE | CE+TS | FL+TS | DFL+TS | BSCE | **BSCE-GRA** |
|---------|-------|-----|-------|-------|--------|------|------------|
| CIFAR10 | ResNet50 | 4.36 | 1.32 | 1.15 | 1.00 | 0.88 | **0.74** |
| CIFAR10 | ResNet110 | 4.70 | 1.56 | 1.17 | 1.01 | 0.99 | **0.87** |
| CIFAR100 | ResNet50 | 18.05 | 3.05 | 2.57 | 2.56 | 1.90 | **1.59** |
| CIFAR100 | ResNet110 | 18.84 | 4.63 | 3.71 | 3.47 | 2.75 | **2.53** |
| CIFAR100 | DenseNet | 19.10 | 3.43 | 1.30 | 1.83 | 1.62 | **1.61** |
| TinyImageNet | ResNet50 | 14.94 | 5.16 | 2.18 | 2.28 | 1.76 | **1.47** |

### Ablation Study

| Method | Framework | Uncertainty Metric | CIFAR10 ECE | CIFAR100 ECE |
|------|------|-----------|------------|-------------|
| FL-Loss | Loss Weighting | $(1-p_c)^\gamma$ | 1.15 | 2.57 |
| FL-GRA | Gradient Weighting | $(1-p_c)^\gamma$ | 0.95 | 1.92 |
| DFL-Loss | Loss Weighting | $(1-p_c+p_j)^\gamma$ | 1.00 | 2.56 |
| DFL-GRA | Gradient Weighting | $(1-p_c+p_j)^\gamma$ | 0.88 | 1.78 |
| BS-Loss | Loss Weighting | Brier Score | 0.88 | 1.90 |
| BS-GRA | Gradient Weighting | Brier Score | **0.74** | **1.59** |

### Key Findings

- **Consistent Gains from Loss Weighting to GRA**: Regardless of the uncertainty metric used, switching from loss weighting to gradient weighting consistently improves calibration (FL: 1.15 → 0.95, DFL: 1.00 → 0.88, BS: 0.88 → 0.74).
- **BS Consistently Outperforms FL/DFL**: Under the same weighting framework, Brier Score consistently yields better calibration results.
- **BSCE-GRA Eliminates the Need for Post-hoc Temperature Scaling**: Its Pre-TS and Post-TS performances are almost identical, indicating that the model is already well-calibrated during training.
- The proposed method is consistently effective across different model architectures (ResNet, WideResNet, DenseNet) and datasets.
- Toy dataset experiments quantitatively show that the Pearson correlation coefficient of BS (0.664) > DFL (0.638) > FL (0.550).

## Highlights & Insights

- **Precise Theoretical Analysis**: Analyzing the scaling function $g(p,\gamma)$ of Focal Loss from a gradient perspective reveals the theoretical root cause of misalignment.
- **Extremely Elegant Solution**: The core modification is simply detaching the gradient of the weights, incurring virtually zero extra computation.
- **Unified Framework Contribution**: Unifying methods like FL and DFL under the perspective of "uncertainty-weighted CE" and highlighting that the weighting of gradients, rather than losses, is what truly matters.
- **Theoretical Elegance**: The difference between BS and the true calibration error depends solely on the inherent properties of the samples, independent of model predictions.
- The 3D visualization of 4-class classification intuitively displays the differences in uncertainty awareness among FL, DFL, and BS.

## Limitations & Future Work

- The theoretical analysis is mainly based on the SGD optimizer; the impact on adaptive optimizers like Adam is not fully explored.
- Primarily validated on image classification tasks; effectiveness in other tasks (e.g., object detection, segmentation) remains to be explored.
- The hyperparameters $\beta, \gamma$ of the Generalized Brier Score could theoretically be further explored, though this work directly utilizes standard values.
- The combination of the proposed method with other calibration techniques (e.g., Mixup, Label Smoothing) can be further investigated.
- The gradient weighting framework can be generalized to other scenarios requiring sample-level weighting (e.g., long-tailed classification, noisy label learning).

## Related Work & Insights

- **Focal Loss**: The primary analytical subject of this work, which reveals the actual reason for its effectiveness in calibration (uncertainty weighting) and its limitations (gradient misalignment).
- **Dual Focal Loss**: Extends FL by considering the second most likely class, but remains incomplete.
- **Temperature Scaling**: A post-processing calibration method; BSCE-GRA achieves sufficient calibration during training, making TS almost redundant.
- Insights: Many design issues with "loss functions" can be re-evaluated from the gradient perspective, as gradients are the key factors directly steering the optimization.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 8 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Value | 8 |
| Overall Rating | 8.0 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/others/measuring_uncertainty_calibration.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](../../ICML2025/others/cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[AAAI 2026\] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach](../../AAAI2026/others/variance_computation_for_weighted_model_counting_with_knowledge_compilation_appr.md)
- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)

</div>

<!-- RELATED:END -->
