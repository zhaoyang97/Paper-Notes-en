---
title: >-
  [Paper Note] The Geometry of Projection Heads: Conditioning, Invariance and Collapse
description: >-
  [ICML 2026][Self-Supervised Learning][Paper Note] This paper analyzes the projection head in self-supervised learning (SSL) as a trainable metric tensor from a Riemannian geometry perspective. It demonstrates that its role is to dynamically whiten the optimization landscape, escape collapse saddle points via negative curvature from smooth activations, and induce metri
tags:
  - ICML 2026
  - Self-Supervised Learning
date: 2026-05-08
content_hash: 21520e00a1a30ce8
---
# The Geometry of Projection Heads: Conditioning, Invariance and Collapse

**Conference**: ICML 2026  
**arXiv**: [2605.17180](https://arxiv.org/abs/2605.17180)  
**Code**: TBD  
**Area**: Self-Supervised Learning / Representation Learning Theory  
**Keywords**: Projection Head, Self-Supervised Learning, Riemannian Geometry, Dimensional Collapse, Invariance

## TL;DR
This paper analyzes the projection head in self-supervised learning (SSL) as a trainable metric tensor from a Riemannian geometry perspective. It demonstrates that its role is to dynamically whiten the optimization landscape, escape collapse saddle points via negative curvature from smooth activations, and induce metric singularity along data augmentation directions—collectively explaining the long-standing mystery of why it is "required during training but discarded during inference."

## Background & Motivation

**Background**: The "train-then-discard" phenomenon in SSL—training with a multi-layer perceptron (MLP) projection head $h_\phi$ but only using the backbone $f_\theta$ for inference. This seems paradoxical: if the projection head is necessary for training, why is it discarded during inference?

**Limitations of Prior Work**: Existing works use information bottleneck or dimensional collapse prevention as post-hoc descriptions but lack a mechanistic understanding: why do non-linear heads simultaneously filter information, accelerate convergence, and escape collapse saddle points?

**Key Insight**: The projection head is essentially metric learning on the representation manifold. Rigid invariance constraints (e.g., color invariance) in contrastive losses force the network to "destroy" certain information. The projection head absorbs this damage by altering the local geometry of the representation space, thereby protecting the backbone.

**Key Challenge**: The strong constraints of the loss function on augmentation directions versus the backbone representations' need to maintain informational richness for downstream tasks. This contradiction must be absorbed by an intermediate layer.

**Goal**: To model the projection head using Riemannian geometry tools as a dynamic metric tensor acting on the backbone representation manifold, and to derive and empirically verify three major geometric functions.

**Core Idea**: The projection head is a "disposable pre-processor" that whitens the optimization landscape, injects negative curvature under smooth activations to escape collapse, and induces metric singularity along augmentation directions to push augmentation-irrelevant information out of the representation space.

## Method

### Overall Architecture
Let the backbone be $f_\theta: \mathcal{X} \to \mathcal{Z}$ and the projection head be $h_\phi: \mathcal{Z} \to \mathcal{H}$. Define the **effective Hessian** as $H_{\text{eff}}(z) = J_h(z)^\top \nabla_h^2 \mathcal{L} J_h(z) + \sum_i [\nabla_h \mathcal{L}]_i \nabla_z^2 h_i(z)$; the first term is the Gauss-Newton pullback metric, and the second term is an interaction term driven by the intrinsic curvature of the projection head. The **augmentation tangent space** $\mathcal{V}_{\text{aug}}(z)$ is the direction spanned by infinitesimal changes in continuous augmentation parameters—the projection head must compress orbits in these directions.

### Key Designs

**1. Global Mahalanobis Whitening from a Linear Head Perspective: Projection Heads as Metric Learning**

To understand the projection head, consider the simplest linear case. When $h(z) = W z$, the similarity $\langle h(z_i), h(z_j) \rangle = z_i^\top (W^\top W) z_j$, which is equivalent to learning a global metric $M = W^\top W$. Theorem 3.1 proves that there exists a linear head such that the effective Hessian is isomorphic to the identity matrix on an $r$-dimensional subspace, performing implicit whitening on the loss-relevant subspace. This explains why adding even a linear head results in faster convergence than a head-less approach.

However, a linear head is a global fixed transformation and cannot adapt to the changing geometry along the optimization trajectory. Proposition 3.3 formalizes this limitation: when the intrinsic geometry of the loss has non-zero Riemann curvature, no global constant linear transformation can make the effective Hessian simultaneously non-degenerate and isotropic everywhere. In other words, non-linear heads are a geometric necessity, not just an auxiliary enhancement.

**2. Trajectory Linearization and Capacity Threshold of Non-linear Heads: Depth and Width as Topological Necessities**

Since linear heads are insufficient, where does the strength of non-linear heads lie? Theorem 3.2 shows that for any smooth, non-self-intersecting optimization trajectory $\gamma(t)$, there exists an MLP head that makes the induced effective Hessian $\epsilon$-isotropic along the entire trajectory—it can learn a state-dependent metric to straighten the curved optimization landscape.

But this requires sufficient capacity. Proposition 3.4 quantifies the approximation error upper bound as $\|H_{\text{eff}}^\phi - H_{\text{eff}}^*\|_2 \leq 2 L M \epsilon + M \epsilon^2$. Corollary 3.5 provides a threshold: to maintain isotropic condition numbers, $\epsilon < \lambda_{\min}(H_{\text{eff}}^*) / (2 L M)$ must hold. Once the approximation error exceeds this threshold, collapse points transition from escapable to stable. This suggests that the depth and width of the head are not empirical heuristics but topological necessities—insufficient capacity causes the model to cross the boundary into locked collapse.

**3. Injecting Negative Eigenvalues via Smooth Activations to Escape Collapse**

Non-contrastive SSL (e.g., BYOL, SimSiam) avoids collapse without explicit negative samples, a mechanism that has remained mysterious. This paper identifies the curvature of activation functions as the key. At a collapse configuration $z^*$ (where all inputs map to a constant), the interaction term of a linear head $M(z^*) = 0$ (since $\nabla^2 \text{linear} = 0$), leaving the effective Hessian positive semi-definite and making collapse a non-repelling critical region. With smooth non-linear heads (Swish, GELU), $\nabla^2 h$ is non-zero. Since the loss Hessian $G(z^*)$ often has a non-trivial null space in high-dimensional representation space (intrinsic rank $r < d$), the interaction term $M(z^*)$ generates negative eigenvalues in these directions, turning stable minima into strict saddle points—which gradient descent is almost certain to escape according to non-convex optimization theory (Lee et al.).

This attributes the lack of collapse in non-contrastive SSL to geometric curvature rather than stochastic noise, and predicts that ReLU heads ($\nabla^2 \text{ReLU} = 0$ almost everywhere) lack this guarantee, instead relying on discrete dynamics and BatchNorm.

**4. Inducing Metric Singularity along Augmentation Directions: Theoretical Necessity of "Discarding after Training"**

The previous points explain how the head assists training, but do not explain why the head should be discarded during inference (the "guillotine effect" found in SimCLR). This is addressed by "Invariance." Proposition 5.2 reveals that when a smooth head achieves local invariance to continuous augmentations, the pullback metric $G(z) = J_h(z)^\top J_h(z)$ must be singular on the augmentation tangent space $\mathcal{V}_{\text{aug}}$—$v^\top G(z) v = 0,\ \forall v \in \mathcal{V}_{\text{aug}}$. Thus, the projection head acts as a **geometric low-pass filter**, compressing finite distances along augmentation orbits in backbone space $\mathcal{Z}$ to zero in embedding space $\mathcal{H}$. This satisfies the SSL objective but results in **irreversible information loss**.

Theorem 5.3 quantifies this loss using the Fisher Information Matrix: $\text{rank}(\mathcal{I}_{h(z)}) \leq \text{rank}(\mathcal{I}_z) - \dim(\mathcal{V}_{\text{aug}})$. The rank of information in the projection output is exactly one augmentation dimension less than the backbone. Because the backbone is **upstream** of this singular metric, it retains the full dimensions of the data manifold. Discarding the projection head is therefore a theoretical necessity to recover discriminative information filtered by invariance learning—the geometric root of why linear probes on the backbone out-perform those on the head by 14.72 points in later ablations.

## Key Experimental Results

### Main Results: Hessian Tracking and Activation Effects

| Activation | Initialization | Conditioning Behavior | $\lambda_{\min} < 0$ Injection | Escape Collapse |
| :--- | :--- | :--- | :--- | :--- |
| Swish (Smooth) | Normal | Rapid peak then plateau | Yes | ✓ Fast |
| Swish (Pseudo-collapse) | Collapse-like | Violent peak $\rho_s = 0.609$ | Yes | ✓ Mechanistic |
| ReLU | Normal | Slow, no negative eigenvalues | No | ✗ Failure |
| ReLU | Pseudo-collapse | Static oscillation | No | ✗ Needs BN / Large LR |
| Linear | Pseudo-collapse | Slow drift | No | ✓ Eventual escape (slow) |

Smooth activations actively inject negative eigenvalues to trigger a "topological phase transition," driving a surge in representation variance. ReLU lacks this mechanism and remains trapped in collapse under continuous gradient flow without BatchNorm.

### Ablation Study: Orbit Compression and Information Entanglement

| Metric | Backbone $z$ | Projection Head $h(z)$ | Ratio | Description |
| :--- | :--- | :--- | :--- | :--- |
| Orbit Mean Square Spread ($\times 10^{-2}$) | 2.25 ± 1.07 | 0.10 ± 0.06 | 22.5× compression | Prop 5.2 verified: metric singularity in $\mathcal{V}_{\text{aug}}$ |
| Intra-orbit distance $D_{\text{intra}}$ | 0.211 ± 0.045 | 0.044 ± 0.011 | 4.76× comp. | Augmentation direction targetedly compressed |
| Inter-class distance $D_{\text{inter}}$ | 0.432 ± 0.052 | 0.111 ± 0.014 | 3.89× comp. | Semantic structure relatively maintained |
| $D_{\text{inter}} / D_{\text{intra}}$ | 2.04 ± 0.52 | 2.50 ± 0.72 | 1.22× ↑ | Selective compression |
| Linear probe accuracy | 52.27% | 37.55% | -14.72 | Information cost of linear invariance |
| MLP probe accuracy | 55.46% | 43.56% | -11.90 | MLP-linear gap doubled: Information entanglement |

Core theory verification: the metric $G(z) = J_h(z)^\top J_h(z)$ learned by the head selectively induces metric singularity in the augmentation tangent space while relatively preserving semantic clustering. The significant increase in the MLP probe advantage indicates that information is non-linearly entangled rather than erased, supporting the optimality of "discarding the projection head."

### Key Findings
- **Intrinsic difference between ReLU and smooth activations**: The non-zero $\nabla^2 h$ of smooth activations is critical to escaping collapse. ReLU fails due to its second derivative being zero almost everywhere.
- **22.5× Orbit Compression**: Direct evidence that the projection head induces a near-singular metric in the $\mathcal{V}_{\text{aug}}$ direction while semantic distance is only compressed by 3.89×, indicating selective rather than global compression.
- **Doubling of the MLP-linear gap**: The decline in linear separability at the head output versus the recovery in MLP separability suggests information is non-linearly entangled rather than erased, justifying the use of the backbone for downstream tasks.

## Highlights & Insights
- **Unified Geometric Framework**: Uses the Riemannian metric tensor to provide a unified explanation for the three roles of projection heads (whitening, escaping collapse, and inducing singularity), which is deeper than information-theoretic or optimization-centric views.
- **ReLU Theoretical Chasm**: Identifies a fundamental difference between smooth activations and ReLU under continuous gradient flow, explaining the practical preference for Swish/GELU over ReLU.
- **Geometric Explanation of Metric Singularity**: Theorem 5.1–5.3 quantifies why discarding the head is necessary—the head induces metric degradation in the $\mathcal{V}_{\text{aug}}$ direction, making the backbone representation superior for downstream tasks.
- **Quantification of Capacity Thresholds**: Proposition 3.4 elevates the question of "head depth/width" from an empirical heuristic to a topological theorem.

## Limitations & Future Work
- The theory proves that a projection head *can* optimize the landscape, but how SGD dynamics reach these configurations remains unknown.
- Extensions to data-dependent metrics induced by ViT self-attention are missing.
- Infinitesimal modeling does not apply to discrete augmentations (e.g., horizontal flips) without smooth group structures.
- Assumes the projection head reaches equilibrium rapidly, but the real-world time-scale coupling between the head and backbone may not satisfy this.

## Related Work & Insights
- **vs. Information Bottleneck (Tishby)**: Both suggest the head filters irrelevant variables; Ours adds the geometric mechanism (metric singularity), moving from "what is filtered" to "how it is filtered."
- **vs. Dimensional Collapse Defense (Jing 2022; Tian 2021)**: Previous work focused on BatchNorm/gradient stopping; Ours proves smooth activation's intrinsic curvature is sufficient to escape collapse.
- **vs. Natural Gradient Descent (Amari 1998)**: The dynamic metric learned by the head is the geometric basis for natural gradients.
- **vs. Explicit Whitening (VICReg, Barlow Twins)**: Explicit whitening relies on loss constraints, while implicit whitening relies on the projection head metric; Ours unifies their geometric essence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reinterprets projection heads via Riemannian geometry, introducing a conceptual system for metric tensors, orbit compression, and curvature injection.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive Hessian tracking, orbit visualization, and base model validation, though lacks edge cases for discrete augmentations and massive datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise theorem statements, clear proof outlines, and powerful intuitive visualizations.
- Value: ⭐⭐⭐⭐⭐ Resolves two classic mysteries of SSL and provides geometric grounds for algorithm design (activation choice, head depth).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[ICML 2026\] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[CVPR 2026\] Geometry-driven OOD Detectors Are Class-Incremental Learners](../../CVPR2026/self_supervised/geometry-driven_ood_detectors_are_class-incremental_learners.md)

</div>

<!-- RELATED:END -->
