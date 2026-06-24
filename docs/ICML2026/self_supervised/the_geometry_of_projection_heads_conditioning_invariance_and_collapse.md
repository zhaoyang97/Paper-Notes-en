---
title: >-
  [Paper Note] The Geometry of Projection Heads: Conditioning, Invariance and Collapse
description: >-
  [ICML 2026][Self-Supervised Learning][Projection Head] This paper analyzes projection heads in self-supervised learning as trainable metric tensors from a Riemannian geometry perspective. It demonstrates that their role is to dynamically whiten the optimization landscape, escape collapse saddle points via negative curvature from smooth activations, and induce metric singularities along data augmentation directions—collectively explaining the long-standing mystery of why these…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Projection Head"
  - "Riemannian Geometry"
  - "Representation Collapse"
  - "Invariance"
date: 2026-05-08
content_hash: 15feb64d8ebc23c4
---

# The Geometry of Projection Heads: Conditioning, Invariance and Collapse

**Conference**: ICML 2026  
**arXiv**: [2605.17180](https://arxiv.org/abs/2605.17180)  
**Code**: TBD  
**Area**: Self-Supervised Learning / Representation Learning Theory  
**Keywords**: Projection Head, Self-Supervised Learning, Riemannian Geometry, Representation Collapse, Invariance

## TL;DR
This paper analyzes projection heads in self-supervised learning as trainable metric tensors from a Riemannian geometry perspective. It demonstrates that their role is to dynamically whiten the optimization landscape, escape collapse saddle points via negative curvature from smooth activations, and induce metric singularities along data augmentation directions—collectively explaining the long-standing mystery of why these heads are "required during training but discarded for inference."

## Background & Motivation

**Classic Phenomenon**: The "train-then-discard" paradigm in Self-Supervised Learning (SSL)—an MLP projection head $h_\phi$ is added during training, but only the backbone $f_\theta$ is retained for inference while the head is discarded. This appears paradoxical: if the projection head is necessary for training, why is it not utilized during inference?

**Limitations of Prior Work**: Existing studies use information bottleneck or dimension collapse as post-hoc descriptions but lack a mechanistic understanding: why can non-linear heads simultaneously filter information, accelerate convergence, and escape collapse saddle points?

**Key Insight**: The projection head is essentially metric learning on the representation manifold. Rigid invariance constraints (e.g., color invariance) in contrastive losses force the network to "destroy" certain information. The projection head absorbs this destruction by altering the local geometry of the representation space, thereby protecting the backbone.

**Key Challenge**: The contradiction between the strong constraints of the loss function on augmentation directions and the need for the backbone representation to retain rich information for downstream tasks must be absorbed by an intermediate layer.

**Goal**: To model the projection head as a dynamic metric tensor acting on the backbone representation manifold using Riemannian geometry tools, deriving and empirically validating three primary geometric functions.

**Core Idea**: The projection head serves as a "disposable pre-conditioner"—it whitens the optimization landscape; injects negative curvature under smooth activations to escape collapse; and induces metric singularity along augmentation directions to push augmentation-irrelevant information out of the representation space.

## Method

### Overall Architecture
Let the backbone be $f_\theta: \mathcal{X} \to \mathcal{Z}$ and the projection head be $h_\phi: \mathcal{Z} \to \mathcal{H}$. Define the **effective Hessian** $H_{\text{eff}}(z) = J_h(z)^\top \nabla_h^2 \mathcal{L} J_h(z) + \sum_i [\nabla_h \mathcal{L}]_i \nabla_z^2 h_i(z)$. The first term is the Gauss-Newton pull-back metric, and the second term is the interaction term driven by the intrinsic curvature of the projection head. The **augmentation tangent space** $\mathcal{V}_{\text{aug}}(z)$ consists of directions spanned by infinitesimal changes in continuous augmentation parameters—the projection head must compress orbits in these directions.

### Key Designs

**1. Global Mahalanobis Whitening under Linear Heads: Projection Head as Metric Learning**

To understand the projection head, consider the simplest case of a linear head. When $h(z) = W z$, the similarity $\langle h(z_i), h(z_j) \rangle = z_i^\top (W^\top W) z_j$, which is equivalent to learning a global metric $M = W^\top W$—the projection head performs metric learning from the outset. Theorem 3.1 proves the existence of a linear head that makes the effective Hessian isomorphic to the identity matrix on an $r$-dimensional subspace, performing implicit whitening on the loss-relevant subspace. This explains why adding even a linear head improves convergence over head-less schemes.

However, a linear head is a global fixed transformation and cannot adapt to the changing geometry along the optimization trajectory. Proposition 3.3 formalizes this limitation: when the intrinsic geometry of the loss has non-zero Riemann curvature, no global constant linear transformation can make the effective Hessian simultaneously non-degenerate and isotropic everywhere. In other words, non-linear heads are a geometric necessity, not just an auxiliary enhancement.

**2. Trajectory Linearization + Capacity Thresholds: Depth and Width as Topological Necessities**

Since linear heads are insufficient, what is the advantage of non-linear heads? Theorem 3.2 states: for any smooth, non-self-intersecting optimization trajectory $\gamma(t)$, there exists an MLP head such that the induced effective Hessian is $\epsilon$-isotropic along the entire trajectory. It learns a state-dependent metric that "straightens" the curved optimization landscape.

This requires sufficient head capacity. Proposition 3.4 quantifies the approximation error upper bound $\|H_{\text{eff}}^\phi - H_{\text{eff}}^*\|_2 \leq 2 L M \epsilon + M \epsilon^2$. Corollary 3.5 provides a threshold: to maintain isotropic condition numbers, $\epsilon < \lambda_{\min}(H_{\text{eff}}^*) / (2 L M)$ must hold. Once the approximation error exceeds this threshold, collapse points transition from escapable to stable. This implies that the required depth and width of the head are topological necessities rather than empirical heuristics—insufficient capacity causes the model to be locked into collapse.

**3. Escaping Collapse via Negative Eigenvalues from Intrinsic Curvature in Smooth Activations**

Non-contrastive SSL (e.g., BYOL, SimSiam) avoids collapse without explicit negative samples, a mechanism that has remained mysterious. This paper points to the curvature of activation functions as the key. In a collapse configuration $z^*$ (where all inputs map to a constant), the interaction term of a linear head $M(z^*) = 0$ (since $\nabla^2 \text{linear} = 0$), leaving the effective Hessian positive semi-definite and the collapse as a non-repulsive critical region. With smooth non-linear heads (Swish, GELU), $\nabla^2 h$ is non-zero. Since the loss Hessian $G(z^*)$ often has a non-trivial null space in high-dimensional representation spaces, the interaction term $M(z^*)$ generates negative eigenvalues in these directions, turning stable minima into strict saddle points—where gradient descent is guaranteed to escape almost surely.

This attributes the anti-collapse property of non-contrastive SSL to geometric curvature rather than stochastic noise, predicting that ReLU heads ($\nabla^2 \text{ReLU} = 0$ almost everywhere) lack this guarantee and must rely on discrete dynamics or BatchNorm.

**4. Inducing Metric Singularity Along Augmentation Directions: The Necessity of "Discarding"**

The previous points explain how heads assist training, but why discard them during inference (the "guillotine effect" identified in SimCLR)? This relates to "Invariance." Proposition 5.2 provides the key insight: when a smooth head achieves local invariance to continuous augmentations, the pull-back metric $G(z) = J_h(z)^\top J_h(z)$ must be singular on the augmentation tangent space $\mathcal{V}_{\text{aug}}$, i.e., $v^\top G(z) v = 0,\ \forall v \in \mathcal{V}_{\text{aug}}$. Thus, the projection head acts as a **geometric low-pass filter**, compressing finite distances along augmentation orbits in backbone space $\mathcal{Z}$ to zero in embedding space $\mathcal{H}$.

Theorem 5.3 (Information Hierarchy) quantifies this loss using the Fisher Information Matrix: $\text{rank}(\mathcal{I}_{h(z)}) \leq \text{rank}(\mathcal{I}_z) - \dim(\mathcal{V}_{\text{aug}})$. The information rank of the projection output is exactly lower by the augmentation dimension. Because the backbone is **upstream** of this singular metric, it retains the full dimensionality of the data manifold. Consequently, "discarding the projection head" is a theoretical necessity to recover discriminative information filtered by invariance learning.

## Key Experimental Results

### Main Results: Hessian Tracking + Activation Effects

| Activation | Initialization | Condition Number Behavior | $\lambda_{\min} < 0$ Injection | Escape Collapse |
|:---|:---|:---|:---|:---|
| Swish (Smooth) | Normal | Rapid peak then plateau | Yes | ✓ Fast |
| Swish (Pseudo-collapse) | Collapse-like | Violently peak $\rho_s = 0.609$ | Yes | ✓ Mechanistic |
| ReLU | Normal | Slow, no negative eigenvalues | No | ✗ Fail |
| ReLU | Pseudo-collapse | Static oscillation | No | ✗ Needs BN/Large LR |
| Linear | Pseudo-collapse | Slow drift | No | ✓ Slow escape |

Smooth activations actively inject negative eigenvalues to trigger a "topological phase transition," driving a surge in representation variance. ReLU lacks this mechanism and falls into collapse under continuous gradient flow without BatchNorm.

### Ablation Study: Orbit Compression + Information Entanglement

| Metric | Backbone $z$ | Projection Head $h(z)$ | Ratio | Note |
|:---|:---|:---|:---|:---|
| Orbit Mean Square Spread ($\times 10^{-2}$) | 2.25 ± 1.07 | 0.10 ± 0.06 | 22.5× Comp. | Prop 5.2: $\mathcal{V}_{\text{aug}}$ metric singularity |
| Intra-orbit Distance $D_{\text{intra}}$ | 0.211 ± 0.045 | 0.044 ± 0.011 | 4.76× Comp. | Targeted compression of augmentations |
| Inter-class Distance $D_{\text{inter}}$ | 0.432 ± 0.052 | 0.111 ± 0.014 | 3.89× Comp. | Semantic structure relatively preserved |
| $D_{\text{inter}} / D_{\text{intra}}$ | 2.04 ± 0.52 | 2.50 ± 0.72 | 1.22× ↑ | Selective compression |
| Linear Probe Accuracy | 52.27% | 37.55% | -14.72 | Information cost of linear invariance |
| MLP Probe Accuracy | 55.46% | 43.56% | -11.90 | Doubled MLP-Linear gap: Information entanglement |

These results validate the core theory: the metric $G(z) = J_h(z)^\top J_h(z)$ learned by the projection head selectively induces metric singularity in the augmentation tangent space while relatively maintaining semantic clustering. The significant growth in the MLP probe advantage indicates that information is non-linearly entangled rather than erased, supporting the optimality of discarding the head.

### Key Findings
- **Intrinsic Difference between ReLU and Smooth Activations**: The non-zero $\nabla^2 h$ of smooth activations is critical for escaping collapse; ReLU fails due to its second derivative being zero almost everywhere.
- **22.5× Orbit Compression**: Direct evidence that the projection head induces a near-singular metric in the $\mathcal{V}_{\text{aug}}$ direction, while semantic distance is compressed only by 3.89×, indicating selective rather than global compression.
- **Doubled MLP-Linear Gap**: The drop in linear separability at the head output compared to the recovery of MLP separability suggests information is non-linearly entangled, justifying the removal of projection heads.

## Highlights & Insights
- **Unified Geometric Framework**: Uses Riemannian metric tensors to provide a unified explanation for the three roles of the projection head (whitening, escaping collapse, and inducing singularity), offering deeper insight than information or optimization theory alone.
- **The ReLU Theoretical Gap**: Identifies the fundamental difference between smooth activations and ReLU under continuous gradient flow, explaining why Swish/GELU are preferred in practice.
- **Geometric Explanation of Metric Singularity**: Theorems 5.1–5.3 quantify the "guillotine effect"—the head induces metric degradation in $\mathcal{V}_{\text{aug}}$ directions, making backbone representations superior for downstream tasks.
- **Quantification of Capacity Thresholds**: Proposition 3.4 elevates the heuristic of "how deep/wide a head should be" into a topological theorem.

## Limitations & Future Work
- While the theory proves the existence of heads that optimize the landscape, how SGD dynamics reach these specific configurations remains unknown.
- Extension to data-dependent metrics induced by ViT self-attention is currently missing.
- Infinitesimal modeling does not apply to discrete augmentations (e.g., horizontal flips) lacking smooth group structures.
- The assumption of fast projection head equilibrium may not hold due to time-scale coupling between the head and backbone during actual training.

## Related Work & Insights
- **vs. Information Bottleneck (Tishby)**: Both posit that heads filter irrelevant variables; this work adds the geometric mechanism (metric singularity), moving from "what" is filtered to "how" it is filtered.
- **vs. Dimensional Collapse Defense** (Jing 2022; Tian 2021): Prior work focused on BatchNorm/gradient stop; this work proves that the intrinsic curvature of smooth activations is sufficient to escape collapse.
- **vs. Natural Gradient Descent** (Amari 1998): The dynamic metric learned by the head is the geometric basis for natural gradients.
- **vs. Explicit Whitening** (VICReg, Barlow Twins): Explicit whitening relies on loss constraints; implicit whitening relies on the projection head metric. This paper unifies their geometric essence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Reinterprets projection heads through Riemannian geometry, introducing a conceptual framework of metric tensors, orbit compression, and curvature injection.
- Experimental Thoroughness: ⭐⭐⭐⭐  Comprehensive Hessian tracking, orbit visualization, and base model validation; edge cases for discrete augmentations and large-scale data are missing.
- Writing Quality: ⭐⭐⭐⭐⭐  Precise theorem statements, clear proof logic, and powerful, intuitive visualizations.
- Value: ⭐⭐⭐⭐⭐  Resolves two classic SSL mysteries and provides a geometric basis for algorithm design (activation choice, head depth).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[CVPR 2026\] Teaching DINOv3 About Partial 3D Geometry: A Self-Supervised Geometry-Aware Approach](../../CVPR2026/self_supervised/teaching_dinov3_about_partial_3d_geometry_a_self-supervised_geometry-aware_appro.md)
- [\[ICML 2026\] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)

</div>

<!-- RELATED:END -->
