---
title: >-
  [Paper Note] Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss
description: >-
  [ICML 2025 Spotlight][AI Safety][Lipschitz Neural Networks] This paper proposes an efficient low-rank orthogonal layer parameterization method (BRO Layer) and an annealing-based loss function (Logit Annealing Loss) to construct BRONet, a Lipschitz neural network with stronger certified robustness, achieving SOTA on CIFAR-10/100, Tiny-ImageNet, and ImageNet.
tags:
  - "ICML 2025 Spotlight"
  - "AI Safety"
  - "Lipschitz Neural Networks"
  - "Certified Robustness"
  - "Orthogonal Layers"
  - "Logit Annealing"
date: 2026-05-08
content_hash: 12351a957a2cee6c
---

# Enhancing Certified Robustness via Block Reflector Orthogonal Layers and Logit Annealing Loss

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2505.15174](https://arxiv.org/abs/2505.15174)  
**Code**: [https://github.com/](https://github.com/)  
**Area**: Others  
**Keywords**: Lipschitz Neural Networks, Certified Robustness, Orthogonal Layers, Logit Annealing

## TL;DR
This paper proposes an efficient low-rank orthogonal layer parameterization method (BRO Layer) and an annealing-based loss function (Logit Annealing Loss) to construct BRONet, a Lipschitz neural network with stronger certified robustness, achieving SOTA on CIFAR-10/100, Tiny-ImageNet, and ImageNet.

## Background & Motivation
**Background**: Deep learning models are vulnerable to adversarial attacks. Among certified defense methods, Lipschitz neural networks offer inference-time efficiency advantages since they can compute the certified radius in a single forward pass.

**Limitations of Prior Work**: Existing orthogonal layer construction methods (such as SOC and LOT) are computationally expensive, relying on iterative approximation algorithms (like Newton's method or Taylor expansion), and may violate the 1-Lipschitz constraint due to approximation errors.

**Key Challenge**: The computational overhead of orthogonal layers restricts their application in more complex architectures; moreover, the cross-entropy loss cannot effectively increase the classification margin of Lipschitz networks.

**Key Insight**: Leveraging the low-rank parameterization concept of Block Reflectors to construct orthogonal layers without iterative approximations; and analyzing the limited model capacity of Lipschitz networks to design an annealing loss function.

**Core Idea**: A low-rank orthogonal parameterization of the form $W = I - 2V(V^TV)^{-1}V^T$ combined with an annealing mechanism to progressively increase the classification margin.

## Method

### Overall Architecture
Input image $\to$ Lipschitz network (BRONet) composed of BRO orthogonal convolutional layers $\to$ logit output $\to$ Logit Annealing Loss training. The entire network guarantees a global 1-Lipschitz constraint, and the certified radius can be directly computed during inference as $\varepsilon = \mathcal{M}_f(x) / \sqrt{2}$.

### Key Designs

1. **BRO Layer (Block Reflector Orthogonal Layer)**:

    - Function: Constructs an orthogonal weight matrix to implement linear layers with 1-Lipschitz constraints.
    - Mechanism: Given a non-square parameter matrix $V \in \mathbb{R}^{m \times n}$, the orthogonal matrix is parameterized as $W = I - 2V(V^TV)^{-1}V^T$. Here, $W$ automatically satisfies $W^TW = I$ without any iterative approximation.
    - For convolutional layers, 2D convolution theorem is applied to perform orthogonal convolutions in the Fourier domain: $\tilde{W}_{:,:,i,j} = I - 2\tilde{V}_{:,:,i,j}(\tilde{V}^*_{:,:,i,j}\tilde{V}_{:,:,i,j})^{-1}\tilde{V}^*_{:,:,i,j}$
    - Design Motivation: To avoid the Taylor expansion errors of SOC and the numerical instability of LOT's Newton method, while significantly saving computation and memory due to the low-rank parameterization ($n \leq c$).

2. **Logit Annealing Loss**:

    - Function: Replaces the cross-entropy with CR regularization training objective, more effectively increasing the classification margin.
    - Mechanism: Theoretical analysis using Rademacher Complexity reveals that Lipschitz networks have limited model capacity, making it difficult to minimize empirical margin loss risks. Logit Annealing progressively increases the margin requirement for most data points through an annealing mechanism.
    - Design Motivation: To address the discontinuous gradient issues and gradient dominance problems inherent in CR regularization terms.

3. **BRONet Architecture**:

    - Function: Builds a complete Lipschitz network based on the BRO Layer.
    - Mechanism: Integrates BRO layers into a ResNet-style architecture, utilising the MaxMin activation function to preserve gradient norms.
    - Design Motivation: The computational efficiency of BRO makes it feasible to construct deeper and wider Lipschitz networks.

### Loss & Training
The Logit Annealing Loss combines an annealing mechanism, allowing a smaller margin in the early stages of training and progressively increasing the margin requirement as training progresses, thereby avoiding excessively difficult constraints at the start.

## Key Experimental Results

### Main Results

| Dataset | Perturbation Radius | BRONet | Prev. SOTA | Gain |
|--------|----------|--------|----------|------|
| CIFAR-10 | ε=36/255 | 70.1% | 68.5% (LOT) | +1.6% |
| CIFAR-100 | ε=36/255 | 40.0% | 38.3% | +1.7% |
| Tiny-ImageNet | ε=36/255 | 28.2% | 26.1% | +2.1% |
| ImageNet | ε=36/255 | 40.5% | - | First reported |

### Ablation Study

| Configuration | CIFAR-10 (ε=36/255) | Description |
|------|---------------------|------|
| BRONet + Logit Annealing | 70.1% | Full model |
| BRONet + CE+CR | 67.8% | Without Logit Annealing, drops by 2.3% |
| LOT + Logit Annealing | 68.9% | Replaced orthogonal layers, drops by 1.2% |
| SOC + Logit Annealing | 68.5% | Replaced orthogonal layers, drops by 1.6% |

### Key Findings
- BRO outperforms both SOC and LOT in terms of computational time and memory (with training speed increased by approximately 2-3 times).
- LOT suffers from numerical instability when using Kaiming initialization, which leads to non-orthogonality, whereas BRO does not have this issue.
- Logit Annealing Loss is effective across different orthogonal layer architectures.

## Highlights & Insights
- The low-rank orthogonal parameterization is elegant: it directly constructs orthogonal matrices using Block Reflectors, entirely avoiding iterative approximations. This ensures theoretical correctness while enhancing efficiency.
- It analyzes the limited capacity of Lipschitz networks from the perspective of Rademacher Complexity, providing theoretical guidance for the design of the loss function.
- Although BRO is not a universal orthogonal parameterization (a single layer cannot express all orthogonal matrices), the expressiveness of deep networks can compensate for this.

## Limitations & Future Work
- The expressiveness of a single BRO layer is limited (it has only $n$ eigenvalues equal to -1). Although experiments indicate that deep networks can compensate for this, it is theoretically less flexible than LOT or SOC.
- The certified robustness is only verified under the $\ell_2$ norm, leaving other norms such as $\ell_\infty$ unexplored.
- There is still significant room for improvement in ImageNet-scale experiments.

## Rating
- Novelty: ⭐⭐⭐⭐ The low-rank orthogonal parameterization concept is clever but not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, multiple models, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations.
- Value: ⭐⭐⭐⭐ Provides a practical advancement for the field of certified robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](../../NeurIPS2025/ai_safety/bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [\[ICCV 2025\] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](../../ICCV2025/ai_safety/towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)
- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](../../NeurIPS2025/ai_safety/enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[CVPR 2026\] Enhancing Out-of-Distribution Detection with Extended Logit Normalization](../../CVPR2026/ai_safety/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [\[ICLR 2026\] HyCAS: Simultaneous Certified and Empirical Robustness via Hybrid Convolutional and Attentional Stochasticity](../../ICLR2026/ai_safety/certified_vs_empirical_adversarial_robustness_via_hybrid_convolutions_with_atten.md)

</div>

<!-- RELATED:END -->
