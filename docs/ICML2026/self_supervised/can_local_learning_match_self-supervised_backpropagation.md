---
title: >-
  [Paper Note] Can Local Learning Match Self-Supervised Backpropagation?
description: >-
  [ICML 2026][Self-Supervised Learning][Local learning rules] This paper theoretically proves that local self-supervised learning (local-SSL) can precisely achieve the gradient updates of global backpropagation (BP-SSL) in deep linear networks. Based on this insight, the authors propose CLAPP++ (introducing 2D spatial dependence and direct feedback), which achieves performance comparable to global BP-SSL on CIFAR-10/STL-10/Tiny ImageNet, setting a new SOTA for local-SSL.
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Local learning rules"
  - "backpropagation approximation"
  - "CLAPP++"
  - "biological plausibility"
date: 2026-05-08
content_hash: 17d127ac224840af
---

# Can Local Learning Match Self-Supervised Backpropagation?

**Conference**: ICML 2026  
**arXiv**: [2601.21683](https://arxiv.org/abs/2601.21683)  
**Code**: To be confirmed  
**Area**: Self-Supervised/Representation Learning  
**Keywords**: Local learning rules, self-supervised learning, backpropagation approximation, CLAPP++, biological plausibility  

## TL;DR
This paper theoretically proves that local self-supervised learning (local-SSL) can precisely achieve the gradient updates of global backpropagation (BP-SSL) in deep linear networks. Based on this insight, the authors propose CLAPP++ (introducing 2D spatial dependence and direct feedback), which achieves performance comparable to global BP-SSL on CIFAR-10/STL-10/Tiny ImageNet, setting a new SOTA for local-SSL.

## Background & Motivation

**Background**: End-to-end self-supervised learning (global BP-SSL) has become the core method for training modern AI systems, but its backpropagation (BP) feedback structure lacks a biological counterpart. Local self-supervised learning (local-SSL) methods, such as CLAPP, Forward-Forward, and LPL, attempt to replace global BP with layer-wise Hebbian-like update rules, aligning more closely with biological synaptic plasticity mechanisms.

**Limitations of Prior Work**: Existing local-SSL methods are significantly less capable than global BP-SSL in building effective representations in deep networks. While local learning rules can approximate BP well in supervised settings, the performance gap is much larger in self-supervised settings, and a theoretical foundation connecting local-SSL with global representation learning principles is lacking.

**Key Challenge**: Local-SSL optimizes local losses independently at each layer without gradients propagating across layers. This implies that weight updates in shallow layers do not directly optimize the deep SSL objective—the theoretical relationship between the two remains unclear.

**Goal**: (1) Establish a theoretical connection between local-SSL and global BP-SSL; (2) design improved local-SSL algorithms based on theoretical findings to close the performance gap.

**Key Insight**: Through analysis of deep linear networks, the authors found that when weight matrices are orthonormal, layer-wise updates in local-SSL are exactly equivalent to gradient updates in global BP-SSL. This insight guides further algorithmic improvements in non-linear convolutional networks.

**Core Idea**: By introducing 2D spatial dependence projections and top-layer direct feedback, local-SSL gradients can better approximate global BP-SSL, matching its performance without requiring global backpropagation.

## Method

### Overall Architecture
The paper first establishes a unified formal framework that incorporates various local-SSL algorithms, including CLAPP, Forward-Forward, PhyLL, and SCFF, into a single formulation. Within this framework, the local contrastive loss for each layer is defined as $\mathcal{L}^l = f(z_{\text{pos}}^{l\top} B^l c_{\text{pos}}^l) + f(-z_{\text{neg}}^{l\top} B^l c_{\text{neg}}^l)$, where $f$ is a decreasing function, $B^l$ is a trainable or fixed projection matrix, and $c^l$ is a reference vector. Differences between algorithms lie solely in the choices of $f$, $B^l$, and $c^l$. Based on this, the authors prove an exact equivalence theorem (Theorem 3.1) in deep linear networks and transfer these insights to convolutional networks to propose CLAPP++ and its variants.

### Key Designs

**1. Exact Equivalence Theory in Deep Linear Networks: Local is Global**

Local-SSL has long been questioned: "If each layer optimizes independently and gradients don't propagate, shallow layers cannot optimize deep objectives." The performance gap was assumed to be structural. This paper disproves this for $L$-layer linear networks: assuming all weight matrices $W^l$ are orthonormal and $B^l$ is trainable and reaches an optimum $B_*^l$, the layer-wise local gradient is element-wise equal to the global BP gradient, $\frac{\partial \mathcal{L}_*^l}{\partial W_{ij}^l} = \frac{\partial \mathbf{L}_*}{\partial W_{ij}^l}$.

The intuition is that the gradient propagated from the top layer to layer $l$ involves a product of weights $(W^L \cdots W^{l+1})^\top B_*^L c^L$. When these weights are orthogonal, the product cancels out, leaving exactly the local optimal projection $B_*^l c^l$. This theorem breaks the intuition that "local learning necessarily loses information," providing a theoretical anchor for improvements.

**2. Direct Feedback Mechanism (DFB): Compensating Width Decay with Top-layer Signals**

Exact equivalence requires equal layer widths and orthogonal weights. However, real network widths often decrease, breaking the equal-width condition and degrading the approximation. DFB fixes this by replacing the reference vector in the local loss from same-layer activity $c^l = z'^l$ with top-layer activity $c^l = z'^L$. In semi-orthogonal linear networks where dimensions decrease, it can be proven that $\|\frac{\partial \mathcal{L}_*^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2 \geq \|\frac{\partial \mathcal{L}_{*,\text{fb}}^l}{\partial W^l} - \frac{\partial \mathbf{L}_*}{\partial W^l}\|_F^2$, meaning the DFB version is closer to the BP gradient. It uses high-level signals to recover information lost due to dimensionality reduction, mirroring cortical mechanisms where apical dendrites integrate distal high-level inputs to modulate plasticity.

**3. 2D Spatial Dependence Projection: Restoring Spatial Structure of BP Gradients**

Original CLAPP uses Global Average Pooling (GAP) before calculating loss $\mathcal{L}^l = f(\text{pool}(z^l)^\top B^l \text{pool}(c^l))$, forcing gradients to be shared across all spatial dimensions. However, global BP-SSL gradients $\partial \mathbf{L}/\partial z^l$ are not naturally shared spatially; GAP discards this structure. The improvement uses patch pooling $\mathcal{L}^l = f(\text{flatten}(\text{pool}_{k_1}(z^l))^\top B^l \text{flatten}(\text{pool}_{k_2}(c^l)))$, allowing $B^l$ to learn cross-dependencies between spatial positions. Theoretically (Proposition 3.5), when $k_1 = k_2 = 1$, local-SSL with 2D spatial dependence can precisely calculate BP gradients—this is the most significant contributor to performance in experiments.

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
| CLAPP++ (No 2D spatial dep.) | 75.10 | Significant drop without spatial dependence |
| CLAPP++ | 78.66 | Adding 2D spatial dependence, +3.56% |
| CLAPP++DFB | 79.38 | Adding direct feedback, +0.72% |
| CLAPP++both | 79.62 | Combination of both losses, +0.24% |
| BP-CLAPP++ (Global BP) | 80.36 | Upper bound with global backpropagation |

### Key Findings
- **2D Spatial Dependence is the most critical improvement**: Transitioning to spatial dependence yielded a 3.56% gain on STL-10 (75.10→78.66), the largest contributors among all modifications.
- **Local-SSL matches BP-SSL for the first time**: On CIFAR-10/STL-10/Tiny-ImageNet, the gap between CLAPP++ variants and BP-CLAPP++/BP-InfoNCE disappears, a milestone in local-SSL research.
- **Significant VRAM Savings**: Local-SSL does not require storing activations for all layers. CLAPP++ saves 38% peak VRAM on STL-10 and 59% on ImageNet.
- **Gap remains on ImageNet**: On higher-resolution ImageNet, local-SSL (42-44%) still trails BP-SSL (48-55%), suggesting the quality of local approximations needs further improvement for high-resolution scenarios.

## Highlights & Insights
- **Unified Formal Framework**: The paper unifies seemingly different local-SSL algorithms (CLAPP, Forward-Forward, PhyLL, SCFF) into a single parameterized formula via the choice of $f$, $B^l$, and $c^l$. This abstraction enables elegant theoretical analysis across algorithms.
- **"Theory-to-Practice" Closed Loop**: Starting from the linear network equivalence theorem, the authors identified 2D spatial dependence and direct feedback as key improvements, validating theoretical predictions with performance gains in actual convolutional networks.
- **Biological Significance**: The CLAPP++ weight update $\Delta W_{ji}^l = \gamma \cdot (B^l c^l)_j \cdot \rho'(a_j^l) z_i^{l-1}$ can be decomposed into "neuromodulatory factor × dendritic prediction × Hebbian term." The top-layer feedback in DFB aligns with biological evidence of apical dendrites integrating long-range inputs to regulate synaptic plasticity.

## Limitations & Future Work
- **Architectural Limitations**: Experiments were limited to VGG convolutional networks. Extending local loss definitions to modern architectures with residual connections like ResNet remains an open question.
- **ImageNet Gap**: Local-SSL lags behind BP-SSL by approximately 6-13% on large-scale high-resolution datasets, indicating that local gradient approximation errors accumulate in high-resolution settings.
- **Orthogonality Assumption**: The core theorem relies on orthonormal weight matrices. While real-world weights are far from orthogonal, and experiments show qualitative generalization, theoretical guarantees remain limited.
- **Future Directions**: (1) Designing local-SSL rules for residual connections; (2) exploring similar theories for non-contrastive local-SSL (e.g., VICReg style); (3) narrowing the ImageNet gap using modern SSL techniques like projection heads.

## Related Work & Insights
- **CLAPP** (Illing et al., 2021): The direct foundation of this work, utilizing layer-wise contrastive predictive plasticity rules.
- **Forward-Forward** (Hinton, 2022): Another local-SSL method learning via contrastive activation norms of positive and negative samples.
- **SCFF** (Chen et al., 2025): A self-contrastive variant of Forward-Forward and the previous local-SSL SOTA.
- **LPL** (Halvagal & Zenke, 2023): Non-contrastive local learning using VICReg-like layer-wise losses.
- The "local approximating global" approach could be transferred to other distributed training scenarios requiring reduced communication or memory overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](../../ICLR2026/self_supervised/on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)
- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[ICLR 2026\] Understanding the Learning Phases in Self-Supervised Learning via Critical Periods](../../ICLR2026/self_supervised/understanding_the_learning_phases_in_self-supervised_learning_via_critical_perio.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](../../ICLR2026/self_supervised/soft_equivariance_regularization_for_invariant_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
