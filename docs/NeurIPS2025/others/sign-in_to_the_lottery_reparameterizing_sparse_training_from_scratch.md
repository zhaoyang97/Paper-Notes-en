---
title: >-
  [Paper Note] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch
description: >-
  [NeurIPS 2025][sparse training] This paper identifies the root cause of poor performance in pruning-at-initialization (PaI) sparse training as the inability to learn correct parameter signs as dense-to-sparse methods do.…
tags:
  - "NeurIPS 2025"
  - "sparse training"
  - "lottery ticket hypothesis"
  - "sign flipping"
  - "reparameterization"
  - "pruning at initialization"
date: 2026-05-08
content_hash: fdb1ac99f59a85e2
---

# Sign-In to the Lottery: Reparameterized Sparse Training from Scratch

**Conference**: NeurIPS 2025
**arXiv**: [2504.12801](https://arxiv.org/abs/2504.12801)  
**Code**: None  
**Area**: Other
**Keywords**: sparse training, lottery ticket hypothesis, sign flipping, reparameterization, pruning at initialization

## TL;DR
This paper identifies the root cause of poor performance in pruning-at-initialization (PaI) sparse training as the inability to learn correct parameter signs as dense-to-sparse methods do. To address this, the authors propose Sign-In reparameterization ($\theta = m \odot w$), which introduces an internal degree of freedom to facilitate sign flipping. The approach is theoretically shown to resolve a class of sign-flipping scenarios complementary to those addressed by overparameterization, and empirically yields substantial improvements in sparse-from-scratch training.

## Background & Motivation

- **Background**: As neural networks scale to billions of parameters, efficient training pipelines become critical. Network sparsification is a key approach to achieving efficiency. Current state-of-the-art sparsification methods (e.g., AC/DC, STR, CAP) rely on a dense-to-sparse training paradigm: a full dense network is trained first and then gradually pruned, incurring full computational and memory overhead during the dense training phase.
- **Limitations of Prior Work**: The Lottery Ticket Hypothesis (LTH) posits that sparse subnetworks ("winning tickets") exist within random initializations and can be trained from scratch to match the performance of dense networks. In practice, however, pruning-at-initialization (PaI) methods (e.g., SNIP, GraSP, Synflow) consistently exhibit a significant performance gap relative to dense-to-sparse methods.
- **Key Challenge**: This paper attributes the gap to parameter sign alignment: dense-to-sparse methods achieve sign alignment early in training, whereas PaI methods fail to reliably learn correct signs. Signs carry more critical information than magnitudes—initializing with learned signs and random magnitudes recovers baseline performance, whereas using learned magnitudes and random signs does not.

## Method

### Overall Architecture
Sign-In reparameterizes each weight parameter $\theta$ as an element-wise product of two parameters: $\theta = m \odot w$. This introduces an internal degree of freedom $\beta$ (determined by initialization) satisfying $m^2 - w^2 = \beta \cdot \mathbf{1}$. This reparameterization induces a Riemannian gradient flow in the original parameter space:

$$d\theta_t = -\sqrt{\theta_t^2 + \beta} \odot \nabla L(\theta)\, dt.$$

The key insight is that $\sqrt{\theta_t^2 + \beta}$ provides a nonzero scaling factor even when $\theta$ is near zero, enabling the parameter to cross zero and flip its sign.

### Key Designs

1. **Importance of Sign Alignment**: Extensive experiments establish three key findings: (a) a large proportion of signs flip and stabilize within the first 10 epochs of dense training; (b) initializing a sparse network with learned signs, learned masks, and random magnitudes recovers baseline performance; (c) using learned magnitudes with random signs is essentially equivalent to random initialization. This demonstrates that signs constitute the critical information in the parameter–mask coupling.

2. **Sign-In Reparameterization**: Each parameter $\theta$ is replaced by the product $m \odot w$. This introduces a preconditioner $\sqrt{\theta^2 + \beta}$ in the Riemannian gradient flow. When $\beta > 0$, the gradient scaling is nonzero even at $\theta = 0$ (whereas in the standard parameterization, $\theta^2 = 0$ prevents sign flipping). Because stochastic noise and weight decay cause $\beta$ to shrink toward zero during training, periodic resets are necessary.

3. **Dynamic Scaling Reset**: Every $p$ epochs, the internal scaling is reset to $\beta = 1$ without changing the actual weight value $\theta = m \odot w$ (only the values of $m$ and $w$ are redistributed). This reset is computed analytically and gives weights more opportunities to align to the correct sign. It is the critical component of Sign-In—ablations show that removing the reset substantially diminishes the performance gain.

### Theoretical Analysis
In a simplified two-layer single-neuron setting:

- **Standard gradient flow** (Theorem 5.1): The true value can be recovered only when $a > 0$ and $w > 0$; all three other sign combinations fail.
- **Sign-In gradient flow** (Theorem 5.2): Recovery additionally succeeds when $a < 0$ and $w > 0$, complementing what overparameterization addresses.
- **Impossibility theorem** (Theorem 5.4): No continuous reparameterization can recover the true value when $w < 0$—establishing a fundamental limit showing that reparameterization alone cannot fully substitute for dense training.

## Key Experimental Results

### Main Results: Sign-In Improves Sparse-from-Scratch Training

| Dataset + Model | Sparsity | Random Mask | Random + Sign-In | Gain |
|----------------|----------|------------|-----------------|------|
| CIFAR10 + ResNet20 | 80% | 88.25 | **89.37** | +1.12 |
| CIFAR10 + ResNet20 | 90% | 86.25 | **87.83** | +1.58 |
| CIFAR10 + ResNet20 | 95% | 83.56 | **84.74** | +1.18 |
| CIFAR100 + ResNet18 | 80% | 73.95 | **75.32** | +1.37 |
| CIFAR100 + ResNet18 | 90% | 72.96 | **73.94** | +0.98 |
| CIFAR100 + ResNet18 | 95% | 71.36 | **72.51** | +1.15 |
| ImageNet + ResNet50 | 80% | 73.87 | **74.12** | +0.25 |
| ImageNet + ResNet50 | 90% | 71.56 | **72.19** | +0.63 |
| ImageNet + ResNet50 | 95% | 68.72 | **69.38** | +0.66 |

### Sign Alignment Validation: ResNet50 on ImageNet, 90% Sparsity

| Initialization | AC/DC | STR | RiGL |
|---------------|-------|-----|------|
| Baseline (dense-to-sparse) | 74.68 | 73.65 | 73.75 |
| Epoch 10 sign + learned mask | 73.97 | 71.7 | 73.32 |
| Epoch 30 sign + learned mask | **74.89** | **73.91** | 73.7 |
| Final sign + learned mask | 74.88 | 73.77 | **73.74** |
| Final mag + learned mask | 70.94 | 68.35 | 72.40 |
| Random init + learned mask | 70.6 | 68.38 | 71.89 |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Sign-In w/ vs. w/o scaling reset | Accuracy | Improvement is marginal without reset; reset is critical |
| Sign-In + AC/DC | Accuracy | Further improves dense-to-sparse methods |
| Sign-In + RiGL | Accuracy | Improves dynamic sparse training |
| DeiT Small (ViT) | Accuracy | Sign-In yields >1% improvement on Vision Transformers |
| $\beta=1$ uniform vs. layer-wise scaling | Success rate | Theory requires $\beta_1 > \beta_2$, but uniform $\beta=1$ suffices in practice |

### Key Findings
- Signs matter far more than magnitudes: learned signs with random magnitudes restore baseline performance, but the reverse does not.
- Signs stabilize early in training (approximately during the first 10–30 epochs of warmup).
- Sign-In promotes more sign flips throughout training and finds flatter minima (smaller maximum Hessian eigenvalue).
- Sign-In facilitates more sign flips in layers closer to the output, consistent with theoretical predictions.
- The impossibility theorem establishes a fundamental gap between PaI and dense-to-sparse methods: no reparameterization can fully substitute for dense training.

## Highlights & Insights
- **Precise problem diagnosis**: Attributing PaI underperformance to sign alignment failure is a well-motivated and convincingly supported claim.
- **Strong theory–practice synergy**: Each theorem derived from the single-neuron analysis has clear practical implications; the impossibility theorem honestly acknowledges the fundamental limits of the approach.
- **Orthogonality**: Sign-In is complementary to dense-to-sparse methods—the sign flips it facilitates are distinct from those promoted by dense training, and combining the two yields further gains.
- **Low computational overhead**: Although the number of parameters doubles, training time on ResNet50/ImageNet increases by only approximately 5%, and the two parameters are merged back into one for inference.

## Limitations & Future Work
- Despite the improvements, Sign-In does not close the full performance gap between PaI and dense-to-sparse methods.
- The impossibility theorem indicates a fundamental limit for purely reparameterization-based approaches.
- Experiments are primarily conducted on vision tasks; validation on NLP tasks (e.g., language model training) is absent.
- Theoretical analysis is restricted to the single-neuron setting; sign-flipping dynamics in deep multi-layer networks remain to be studied.
- The reset frequency $p$ is an additional hyperparameter; although $\beta=1$ with fixed $p$ performs robustly in experiments, tuning may be required for different tasks.

## Related Work & Insights
- The work is closely related to Gadhikar & Burkholz (2024), which first analyzed the mechanism by which dense training facilitates sign flipping in the single-neuron setting.
- Continuous sparsification methods (STR, spred, PILoT) employ the same $m \odot w$ reparameterization to induce sparsity; Sign-In inverts this idea and uses it to promote sign flipping instead.
- **Practical implication for efficient training**: If a better method for sign initialization can be found in the future, it may genuinely enable sparse networks trained from scratch to match the performance of dense networks.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[ICML 2026\] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](../../ICML2026/others/haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)
- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[NeurIPS 2025\] Training the Untrainable: Introducing Inductive Bias via Representational Alignment](training_the_untrainable_introducing_inductive_bias_via_representational_alignme.md)
- [\[NeurIPS 2025\] Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback](meta-learning_three-factor_plasticity_rules_for_structured_credit_assignment_wit.md)

</div>

<!-- RELATED:END -->
