---
title: >-
  [Paper Note] Can Local Learning Match Self-Supervised Backpropagation?
description: >-
  [ICML 2026][Self-Supervised Learning][CLAPP++] This paper theoretically proves that local self-supervised learning (local-SSL) can precisely achieve the gradient updates of global backpropagation self-supervised learning (BP-SSL) in deep linear networks. Based on this, the authors propose CLAPP++ (introducing 2D spatial dependency and direct feedback), which achiev
tags:
  - ICML 2026
  - Self-Supervised Learning
  - CLAPP++
date: 2026-05-08
content_hash: 43e5d82e74106932
---
# Can Local Learning Match Self-Supervised Backpropagation?

**Conference**: ICML 2026  
**arXiv**: [2601.21683](https://arxiv.org/abs/2601.21683)  
**Code**: To be confirmed  
**Area**: Self-Supervised/Representation Learning  
**Keywords**: Local learning rules, self-supervised learning, backpropagation approximation, CLAPP++, biological plausibility  

## TL;DR
This paper theoretically proves that local self-supervised learning (local-SSL) can precisely achieve the gradient updates of global backpropagation self-supervised learning (BP-SSL) in deep linear networks. Based on this, the authors propose CLAPP++ (introducing 2D spatial dependency and direct feedback), which achieves performance comparable to global BP-SSL on CIFAR-10/STL-10/Tiny ImageNet, setting a new SOTA for local-SSL.

## Background & Motivation

**Background**: End-to-end self-supervised learning (global BP-SSL) has become the core method for training modern AI systems, but its backpropagation (BP) feedback network structure lacks biological counterparts. Local self-supervised learning (local-SSL) methods such as CLAPP, Forward-forward, and LPL attempt to replace global BP with layer-wise Hebbian-like update rules, aligning more closely with biological neuroplasticity mechanisms.

**Limitations of Prior Work**: Existing local-SSL methods are significantly less capable than global BP-SSL in constructing effective representations in deep networks. While local learning rules can approximate BP well in supervised settings, the performance gap is larger in self-supervised settings, and there is a lack of theoretical foundation linking local-SSL to global representation learning principles.

**Key Challenge**: Local-SSL optimizes local losses independently at each layer without cross-layer gradient propagation. This implies that shallow weight updates do not directly optimize the deep SSL objective—yet the theoretical relationship between the two remains unclear.

**Goal**: (1) Establish a theoretical connection between local-SSL and global BP-SSL; (2) Design improved local-SSL algorithms based on theoretical findings to close the performance gap.

**Key Insight**: Analysis in deep linear networks reveals that when weight matrices are orthonormal, layer-wise local-SSL updates are exactly equivalent to global BP-SSL gradient updates. This theoretical insight guides algorithm improvements in non-linear convolutional networks.

**Core Idea**: By introducing 2D spatial dependency projection and top-layer direct feedback, local-SSL gradients better approximate global BP-SSL, matching its performance without requiring global backpropagation.

## Method

### Overall Architecture
The paper first establishes a unified formal framework that incorporates various local-SSL algorithms like CLAPP, Forward-forward, PhyLL, and SCFF into a single formulation. Within this framework, the local contrastive loss for each layer is defined as $\mathcal{L}^l = f(z_{\text{pos}}^{l\top} B^l c_{\text{pos}}^l) + f(-z_{\text{neg}}^{l\top} B^l c_{\text{neg}}^l)$, where $f$ is a decreasing function, $B^l$ is a trainable or fixed projection matrix, and $c^l$ is a reference vector. Differences between algorithms lie solely in the choices of $f$, $B^l$, and $c^l$. Based on this, the authors prove an exact equivalence theorem (Theorem 3.1) in deep linear networks and then transfer these insights to convolutional networks to propose CLAPP++ and its variants.

### Key Designs

**1. Exact Equivalence Theory in Deep Linear Networks: Local is Global**

Local-SSL has long been suspected of having a structural gap: because layers optimize independently without cross-layer gradients, shallow layers cannot optimize deep objectives. This paper refutes this in $L$-layer linear networks: assuming all weight matrices $W^l$ are orthonormal and $B^l$ is trainable and reaches the optimum $B_*^l$, the layer-wise local gradient is element-wise equal to the global BP gradient, $\frac{\partial \mathcal{L}_*^l}{\partial W_{ij}^l} = \frac{\partial \mathbf{L}_*}{\partial W_{ij}^l}$.

The intuition is: the gradient backpropagated from the top layer to layer $l$ involves a long product of weights $(W^L \cdots W^{l+1})^\top B_*^L c^L$. When these weights are orthogonal, the product cancels out, leaving exactly the local optimal projection $B_*^l c^l$. This theorem breaks the intuition that "local learning must lose information" and provides a theoretical anchor—local-SSL can approximate BP if these equivalence conditions are approached.

**2. Direct Feedback Mechanism (DFB): Compensating for Layer Width Decay with Top-layer Signals**

Exact equivalence requires equal layer widths and orthogonal weights. However, real networks typically have decreasing layer widths, breaking the equal-width condition and degrading the approximation. DFB fixes this by changing the reference vector in the local loss from the same-layer activity $c^l = z'^l$ to the top-layer activity $c^l = z'^L$. In semi-orthogonal linear networks with decreasing dimensions, it can be proven that $\|\frac{\partial \mathcal{L}_*^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2 \geq \|\frac{\partial \mathcal{L}_{*,\text{fb}}^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2$, meaning the DFB version is closer to the BP gradient. It uses high-level signals from the top layer to recover information lost due to dimension reduction, which biologically corresponds to the mechanism where apical dendrites integrate distal inputs from higher brain areas to regulate synaptic plasticity.

**3. 2D Spatial Dependency Projection: Restoring the Spatial Structure of BP Gradients**

Original CLAPP applies Global Average Pooling to feature maps before calculating the loss $\mathcal{L}^l = f(\text{pool}(z^l)^\top B^l \text{pool}(c^l))$, forcing gradients to be shared across the entire spatial dimension. However, the BP-SSL gradient $\partial \mathbf{L}/\partial z^l$ is inherently not spatially shared, so this step discard spatial structure. The improvement uses block pooling $\mathcal{L}^l = f(\text{flatten}(\text{pool}_{k_1}(z^l))^\top B^l \text{flatten}(\text{pool}_{k_2}(c^l)))$, allowing $B^l$ to learn cross-dependencies between different spatial positions, with gradients shared only within local patches of size $k_1$. Theoretically (Proposition 3.5), when $k_1 = k_2 = 1$, 2D spatial dependency local-SSL can precisely compute the BP gradient—this is the most significant improvement in experiments.

## Key Experimental Results

### Main Results

| Method | Local Update | 2D Spatial Dep. | CIFAR-10 | STL-10 | Tiny-ImageNet | ImageNet |
|------|---------|-----------|----------|--------|---------------|----------|
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
|------|-------------|------|
| CLAPP++ (No 2D spatial dep.) | 75.10 | Significant drop without spatial dependency |
| CLAPP++ | 78.66 | With 2D spatial dependency, +3.56% |
| CLAPP++DFB | 79.38 | Adding direct feedback, +0.72% |
| CLAPP++both | 79.62 | Dual loss combination, +0.24% |
| BP-CLAPP++ (Global BP) | 80.36 | Global backpropagation upper bound |

### Key Findings
- **2D spatial dependency is the most critical improvement**: STL-10 accuracy increased by 3.56% (75.10→78.66) when adding spatial dependency, marking the highest contribution among all modifications.
- **Local-SSL matches BP-SSL for the first time**: On CIFAR-10/STL-10/Tiny-ImageNet, the gap between CLAPP++ variants and BP-CLAPP++/BP-InfoNCE disappears, a first in local-SSL history.
- **Significant VRAM savings**: Local-SSL does not need to store activations for all layers. CLAPP++ saves 38% peak VRAM on STL-10 and 59% on ImageNet.
- **Gap remains on ImageNet**: On higher resolution ImageNet, local-SSL (42-44%) still lags behind BP-SSL (48-55%), indicating that the quality of local approximations needs further improvement for high-resolution scenarios.

## Highlights & Insights
- **Unified Formal Framework**: The paper unifies seemingly different local-SSL algorithms (CLAPP, Forward-forward, PhyLL, SCFF) into a single parameterized formula, distinguished only by $f$, $B^l$, and $c^l$. This abstraction enables cross-algorithm theoretical analysis.
- **"Theory-to-Practice" Loop**: Starting from the linear network equivalence theorem, the authors identify 2D spatial dependency and direct feedback as critical improvement directions, ultimately verifying the consistency between theoretical predictions and performance gains in real convolutional networks.
- **Biological Significance**: The weight update in CLAPP++, $\Delta W_{ji}^l = \gamma \cdot (B^l c^l)_j \cdot \rho'(a_j^l) z_i^{l-1}$, can be decomposed into "neuromodulatory factor × dendritic prediction × Hebbian term." Top-layer feedback in DFB corresponds to biological evidence of apical dendrites integrating distal inputs to regulate synaptic plasticity in the cerebral cortex, providing a bridge between AI and neuroscience.

## Limitations & Future Work
- **Architectural Limitations**: Experiments are limited to VGG convolutional networks and do not extend to modern architectures with residual connections like ResNet. Defining local losses for residual connections remains an open question.
- **ImageNet Gap**: Local-SSL still lags behind BP-SSL by approximately 6-13% on large-scale high-resolution datasets, suggesting cumulative errors in local gradient approximation for high-res scenarios.
- **Orthogonality Assumption**: The core theorem relies on orthonormal weight matrices. In practice, weights are far from orthogonal. While experiments show qualitative generalization, theoretical guarantees are limited.
- **Future Directions**: (1) Designing local-SSL rules for residual connections; (2) Exploring similar theories for non-contrastive local-SSL (e.g., VICReg style); (3) Combining modern SSL techniques like projection heads to further close the ImageNet gap.

## Related Work & Insights
- **CLAPP** (Illing et al., 2021): The direct foundation of this paper, using layer-wise contrastive predictive plasticity rules.
- **Forward-forward** (Hinton, 2022): Another local-SSL method using contrastive activation norms of positive and negative samples.
- **SCFF** (Chen et al., 2025): A self-contrastive variant of Forward-forward and the previous local-SSL SOTA.
- **LPL** (Halvagal & Zenke, 2023): Non-contrastive local learning using VICReg-like layer-wise losses.
- The "local approximating global" concept is transferable to other distributed training scenarios requiring reduced communication or memory overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[AAAI 2026\] Self-Supervised Inductive Logic Programming](../../AAAI2026/self_supervised/self-supervised_inductive_logic_programming.md)
- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](../../ICML2025/self_supervised/clustering_properties_of_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
