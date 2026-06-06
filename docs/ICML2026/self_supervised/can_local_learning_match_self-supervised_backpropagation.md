---
title: >-
  [Paper Note] Can Local Learning Match Self-Supervised Backpropagation?
description: >-
  [ICML 2026][Self-Supervised Learning][Local learning rules] Ours theoretically proves that local self-supervised learning (local-SSL) can precisely achieve the gradient updates of global backpropagation self-supervised l…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Local learning rules"
  - "Backpropagation approximation"
  - "CLAPP++"
  - "Biological plausibility"
date: 2026-05-08
content_hash: b432a01f386ba7f4
---

# Can Local Learning Match Self-Supervised Backpropagation?

**Conference**: ICML 2026  
**arXiv**: [2601.21683](https://arxiv.org/abs/2601.21683)  
**Code**: To be confirmed  
**Area**: Self-Supervised/Representation Learning  
**Keywords**: Local learning rules, Self-supervised learning, Backpropagation approximation, CLAPP++, Biological plausibility  

## TL;DR
Ours theoretically proves that local self-supervised learning (local-SSL) can precisely achieve the gradient updates of global backpropagation self-supervised learning (BP-SSL) in deep linear networks. Based on this, the CLAPP++ algorithm is proposed (incorporating 2D spatial dependency and direct feedback), reaching performance parity with global BP-SSL on CIFAR-10/STL-10/Tiny ImageNet and setting a new Prev. SOTA for local-SSL.

## Background & Motivation

**Background**: End-to-end self-supervised learning (global BP-SSL) has become the core method for training modern AI systems, but its backpropagation (BP) feedback network structure lacks a biological counterpart. Local self-supervised learning (local-SSL) methods such as CLAPP, Forward-forward, and LPL attempt to replace global BP with layer-wise Hebbian-like update rules, which better align with biological neuroplasticity mechanisms.

**Limitations of Prior Work**: Existing local-SSL methods are significantly less effective at constructing representations in deep networks compared to global BP-SSL. Although local learning rules can approximate BP well in supervised settings, a larger performance gap exists in self-supervised settings, and there is a lack of theoretical foundation linking local-SSL to global representation learning principles.

**Key Challenge**: Local-SSL optimizes local losses independently at each layer, and gradients do not propagate across layers. This implies that weight updates in shallow layers do not directly optimize the deep SSL objective—yet the relationship between the two remains theoretically unclear.

**Goal**: (1) Establish a theoretical connection between local-SSL and global BP-SSL; (2) Design improved local-SSL algorithms based on theoretical insights to close the performance gap.

**Key Insight**: The authors first analyze deep linear networks and find that when weight matrices are orthonormal, the layer-wise updates of local-SSL are exactly equivalent to the gradient updates of global BP-SSL. This theoretical insight guides algorithm improvements in non-linear convolutional networks.

**Core Idea**: By introducing 2D spatial dependency projections and top-layer direct feedback, local-SSL gradients better approximate global BP-SSL, matching its performance without requiring global backpropagation.

## Method

### Overall Architecture
Ours first establishes a unified formal framework that incorporates various local-SSL algorithms like CLAPP, Forward-forward, PhyLL, and SCFF into the same formulation. In this framework, the local contrastive loss for each layer is defined as $\mathcal{L}^l = f(z_{\text{pos}}^{l\top} B^l c_{\text{pos}}^l) + f(-z_{\text{neg}}^{l\top} B^l c_{\text{neg}}^l)$, where $f$ is a decreasing function, $B^l$ is a trainable or fixed projection matrix, and $c^l$ is a reference vector. Differences between algorithms lie only in the specific choices of $f$, $B^l$, and $c^l$. Building on this, the authors prove an exact equivalence theorem (Theorem 3.1) in deep linear networks and transfer these insights to convolutional networks to propose CLAPP++ and its variants.

### Key Designs

1.  **Exact Equivalence Theory in Deep Linear Networks**:
    - **Function**: Proves that local-SSL layer-wise gradient updates are identical to global BP-SSL gradient updates.
    - **Mechanism**: In an $L$-layer linear network, assuming all weight matrices $W^l$ are orthonormal and $B^l$ is trainable and reaches its optimum $B_*^l$, then $\frac{\partial \mathcal{L}_*^l}{\partial W_{ij}^l} = \frac{\partial \mathbf{L}_*}{\partial W_{ij}^l}$. Intuitively, orthonormal weight matrices ensure that the gradient backpropagated from the top layer to layer $l$, $(W^L \cdots W^{l+1})^\top B_*^L c^L$, exactly equals the local optimal projection $B_*^l c^l$, as the product of orthogonal matrices cancels out.
    - **Design Motivation**: Provides a theoretical foundation for local-SSL—under certain conditions, local optimization is equivalent to global optimization, breaking the intuition that local learning inevitably loses information.

2.  **Direct Feedback Mechanism (DFB)**:
    - **Function**: Improves the quality of local-SSL's approximation of BP-SSL gradients when network layer widths decrease.
    - **Mechanism**: Replaces the reference vector from same-layer activity $c^l = z'^l$ with top-layer activity $c^l = z'^L$ (Theorem 3.3). In linear networks with semi-orthogonal weight matrices (row-orthogonal but with decreasing dimensions), it can be proven that $\|\frac{\partial \mathcal{L}_*^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2 \geq \|\frac{\partial \mathcal{L}_{*,\text{fb}}^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2$, indicating the DFB version has a smaller gap with BP-SSL.
    - **Design Motivation**: Actual network layer widths usually decrease, breaking the equi-width orthogonality condition. DFB compensates for information loss due to dimensionality reduction by using top-layer signals directly, which biologically corresponds to apical dendritic integration in the cortex.

3.  **2D Spatial Dependency Projection**:
    - **Function**: Grants the projection matrix $B^l$ spatial position correlation in convolutional networks, significantly improving the BP approximation quality of local-SSL.
    - **Mechanism**: Original CLAPP calculates loss after global average pooling: $\mathcal{L}^l = f(\text{pool}(z^l)^\top B^l \text{pool}(c^l))$, causing gradients to be shared across spatial dimensions. The improvement uses block pooling $\mathcal{L}^l = f(\text{flatten}(\text{pool}_{k_1}(z^l))^\top B^l \text{flatten}(\text{pool}_{k_2}(c^l)))$, allowing $B^l$ to learn cross-dependencies between different spatial positions; gradients are shared only within a local patch of size $k_1$.
    - **Design Motivation**: The gradient $\partial \mathbf{L}/\partial z^l$ in BP-SSL is not shared spatially. Global average pooling loses this spatial structural information. Theoretically (Proposition 3.5), when $k_1 = k_2 = 1$, local-SSL with 2D spatial dependency can precisely compute BP gradients.

## Key Experimental Results

### Main Results

| Method | Local Update | 2D Spatial Dep | CIFAR-10 | STL-10 | Tiny-ImageNet | ImageNet |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BP-CLAPP++ | No | - | 80.49 | 80.36 | 37.55 | 48.52 |
| BP-InfoNCE | No | - | 80.69 | 81.97 | 36.78 | 55.19 |
| CLAPP | Yes | No | - | 73.6 | - | - |
| LPL | Yes | No | 59.4 | 63.2 | - | - |
| SoftHebb | Yes | No | 80.31 | 76.23 | - | 27.3 |
| SCFF | Yes | Yes | 80.60 | 77.14 | 35.67 | - |
| CLAPP++ | Yes | Yes | **80.51** | 78.66 | 36.63 | 42.55 |
| CLAPP++DFB | Yes | Yes | **80.65** | 79.38 | 36.70 | 44.16 |
| CLAPP++both | Yes | Yes | **81.18** | **79.62** | **37.78** | 42.49 |

### Ablation Study

| Configuration | STL-10 Accuracy | Description |
|:---|:---:|:---|
| CLAPP++ (No 2D spatial dependency) | 75.10 | Significant drop without spatial dependency |
| CLAPP++ | 78.66 | With 2D spatial dependency, +3.56% |
| CLAPP++DFB | 79.38 | Plus direct feedback, +0.72% |
| CLAPP++both | 79.62 | Dual loss combination, +0.24% |
| BP-CLAPP++ (Global BP) | 80.36 | Upper bound of global backpropagation |

### Key Findings
- **2D spatial dependency is the most critical improvement**: Accuracy on STL-10 Gain 3.56% (75.10 $\to$ 78.66) when moving from no spatial dependency to including it, marking the largest contribution among all enhancements.
- **local-SSL matches BP-SSL for the first time**: On CIFAR-10/STL-10/Tiny-ImageNet, the gap between CLAPP++ variants and BP-CLAPP++/BP-InfoNCE disappears, a first in local-SSL history.
- **Significant VRAM savings**: local-SSL does not need to store activations for all layers. CLAPP++ saves 38% peak VRAM on STL-10 and 59% on ImageNet.
- **Gap remains on ImageNet**: On higher-resolution ImageNet, local-SSL (42-44%) still lags behind BP-SSL (48-55%), indicating that the quality of local approximation in high-resolution scenarios requires further improvement.

## Highlights & Insights
- **Unified Formal Framework**: Different local-SSL algorithms like CLAPP, Forward-forward, PhyLL, and SCFF are unified into a parametric formula, distinguished only by choices of $f, B^l, c^l$. This abstraction makes cross-algorithm theoretical analysis possible.
- **"Theory $\to$ Practice" Closed Loop**: Starting from the equivalence theorem in linear networks, identifying spatial dependency and direct feedback as key improvement directions, and finally verifying consistent theoretical predictions and performance gains in real convolutional networks.
- **Biological Significance**: The weight update in CLAPP++ $\Delta W_{ji}^l = \gamma \cdot (B^l c^l)_j \cdot \rho'(a_j^l) z_i^{l-1}$ can be decomposed into "neuromodulatory factor $\times$ dendritic prediction $\times$ Hebbian term." Top-layer feedback in DFB bridges AI and neuroscience by providing evidence for apical dendritic integration.

## Limitations & Future Work
- **Architectural Limitations**: Experiments only use VGG convolutional networks and have not extended to modern architectures with residual connections like ResNet. Defining local loss in residual connections remains an open problem.
- **ImageNet Gap**: local-SSL lags behind BP-SSL by approximately 6-13% on large-scale high-resolution datasets, suggesting local gradient approximation errors accumulate in high-resolution scenarios.
- **Orthogonality Assumption**: Core theorems rely on orthonormal weight matrices. In actual training, weights are far from orthogonal. While experiments show findings generalize qualitatively, theoretical guarantees remain limited.
- **Future Work**: (1) Design local-SSL rules for residual connections; (2) Explore similar theories for non-contrastive local-SSL (e.g., VICReg style); (3) Further bridge the ImageNet gap by combining modern SSL techniques like projection heads.

## Related Work & Insights
- **CLAPP** (Illing et al., 2021): Directly fundamental to this work, using layer-wise contrastive predictive plasticity rules.
- **Forward-forward** (Hinton, 2022): Another local-SSL approach using activation norm contrasts between positive and negative samples.
- **SCFF** (Chen et al., 2025): A self-contrastive variant of Forward-forward, the Prev. SOTA for local-SSL before this paper.
- **LPL** (Halvagal & Zenke, 2023): Non-contrastive local learning using VICReg-like layer-wise losses.
- The "local approximating global" concept in Ours can be transferred to other distributed training scenarios requiring reduced communication or memory overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](../../CVPR2026/self_supervised/mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](../../NeurIPS2025/self_supervised/the_complexity_of_finding_local_optima_in_contrastive_learning.md)

</div>

<!-- RELATED:END -->
