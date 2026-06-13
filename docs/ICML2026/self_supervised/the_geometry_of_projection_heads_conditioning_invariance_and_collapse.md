---
title: >-
  [Paper Note] The Geometry of Projection Heads: Conditioning, Invariance and Collapse
description: >-
  [ICML 2026][Self-Supervised Learning][Projection Head] This paper analyzes projection heads in self-supervised learning (SSL) as trainable metric tensors from a Riemannian geometry perspective. It demonstrates that their…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Projection Head"
  - "Riemannian Geometry"
  - "Representation Collapse"
  - "Invariance"
date: 2026-05-08
content_hash: 40fd13a018b10aba
---

# The Geometry of Projection Heads: Conditioning, Invariance and Collapse

**Conference**: ICML 2026  
**arXiv**: [2605.17180](https://arxiv.org/abs/2605.17180)  
**Code**: To be confirmed  
**Area**: Self-Supervised Learning / Representation Learning Theory  
**Keywords**: Projection Head, Self-Supervised Learning, Riemannian Geometry, Representation Collapse, Invariance

## TL;DR
This paper analyzes projection heads in self-supervised learning (SSL) as trainable metric tensors from a Riemannian geometry perspective. It demonstrates that their role is to dynamically whiten the optimization landscape, escape collapse saddles using negative curvature from smooth activations, and induce metric singularities along data augmentation directions—three mechanisms that together explain the long-standing mystery of "required during training, discarded during inference."

## Background & Motivation

**Historical Phenomenon**: The "train-then-discard" nature of SSL—an MLP projection head $h_\phi$ is added during training, but only the backbone $f_\theta$ is used for inference while the head is discarded. This seems paradoxical: if the projection head is necessary for training, why is it not used for inference?

**Limitations of Prior Work**: Existing studies use information bottlenecks or dimensional collapse defense as ex-post descriptions but lack a mechanistic understanding: why can non-linear heads simultaneously filter information, accelerate convergence, and escape collapse saddles?

**Key Insight**: The projection head is essentially metric learning on the representation manifold. The rigid invariance constraints of contrastive loss (e.g., color invariance) force the network to "destroy" certain information; the projection head absorbs this destruction by altering the local geometry of the representation space, thereby protecting the backbone.

**Key Challenge**: The strong constraints of the loss function on augmentation directions vs. the requirement that backbone representations retain the information richness needed for downstream tasks. This contradiction must be absorbed by an intermediate layer.

**Goal**: To model the projection head as a dynamic metric tensor acting on the backbone representation manifold using Riemannian geometry tools, deriving and empirically validating three primary geometric roles.

**Core Idea**: The projection head acts as a "disposable pre-processor"—whitening the optimization landscape, injecting negative curvature under smooth activations to escape collapse, and inducing metric singularities along augmentation directions to push augmentation-irrelevant information out of the representation space.

## Method

### Overall Architecture
Let the backbone be $f_\theta: \mathcal{X} \to \mathcal{Z}$ and the projection head be $h_\phi: \mathcal{Z} \to \mathcal{H}$. Define the **effective Hessian** $H_{\text{eff}}(z) = J_h(z)^\top \nabla_h^2 \mathcal{L} J_h(z) + \sum_i [\nabla_h \mathcal{L}]_i \nabla_z^2 h_i(z)$. The first term is the Gauss-Newton pullback metric, and the second term is an interaction term driven by the intrinsic curvature of the projection head. The **augmentation tangent space** $\mathcal{V}_{\text{aug}}(z)$ is the set of directions spanned by infinitesimal changes in continuous augmentation parameters—the projection head must compress orbits in these directions.

### Key Designs

1. **Global Mahalanobis Whitening from a Linear Head Perspective**:
    - **Function**: Proves that a linear projection head is equivalent to learning a global metric $M = W^\top W$, performing implicit whitening on the loss-relevant subspace.
    - **Mechanism**: With a linear head $h(z) = W z$, the similarity $\langle h(z_i), h(z_j) \rangle = z_i^\top (W^\top W) z_j$ directly reflects metric learning. Theorem 3.1 shows that a linear head exists such that the effective Hessian is isomorphic to the identity matrix on an $r$-dimensional subspace. However, a linear head is a global fixed transformation and cannot adapt to the changing geometry along a curved optimization trajectory.
    - **Design Motivation**: Explains why linear heads are better than no-head solutions (accelerating convergence) but why non-linearity is essential—Proposition 3.3: when the intrinsic geometry of the loss has non-zero Riemann curvature, no global constant linear transformation can make the effective Hessian non-degenerate and isotropic everywhere.

2. **Trajectory Linearization and Capacity Thresholds of Non-linear Heads**:
    - **Function**: Proves that MLP heads can learn state-dependent metrics along the optimization trajectory and quantifies how approximation errors in the head degrade the optimization geometry.
    - **Mechanism**: Theorem 3.2: for any smooth non-self-intersecting optimization trajectory $\gamma(t)$, there exists an MLP head such that the induced effective Hessian is $\epsilon$-isotropic along the trajectory. Proposition 3.4 quantifies the approximation error upper bound $\|H_{\text{eff}}^\phi - H_{\text{eff}}^*\|_2 \leq 2 L M \epsilon + M \epsilon^2$. Corollary 3.5: maintaining isotropic condition numbers requires $\epsilon < \lambda_{\min}(H_{\text{eff}}^*) / (2 L M)$; exceeding this threshold makes collapse points stable.
    - **Design Motivation**: Explains that the importance of depth/width for projection head performance is not merely empirical heuristics but a topological necessity—insufficient capacity → threshold violation → stable collapse.

3. **Escaping Collapse via Negative Eigenvalues Injected by Smooth Activations**:
    - **Function**: Reveals the geometric mechanism for the stability of non-contrastive SSL (BYOL, SimSiam)—smooth activations induce negative eigenvalues at collapse points, turning stable minima into strict saddles.
    - **Mechanism**: Theorem 4.1: in a collapsed configuration $z^*$ (where all inputs map to a constant), a linear head yields an interaction term $M(z^*) = 0$ (since $\nabla^2 \text{linear} = 0$), leaving the effective Hessian PSD and the collapse as a non-repelling critical region. Smooth non-linear heads (Swish, GELU) have non-zero $\nabla^2 h$. The loss Hessian $G(z^*)$ often has a non-trivial null space in high-dimensional representation spaces (intrinsic rank $r < d$), allowing the interaction term $M(z^*)$ to generate negative eigenvalues in these directions. Standard theory for non-convex optimization (Lee et al.) guarantees that gradient descent almost surely escapes strict saddles.
    - **Design Motivation**: Answers "why non-contrastive SSL does not collapse"—it is geometric curvature, not stochastic noise. Predicts that ReLU heads ($\nabla^2 \text{ReLU} = 0$ almost everywhere) provide no such guarantee and must rely on discrete dynamics and BatchNorm.

## Key Experimental Results

### Main Results: Hessian Tracking + Activation Effects

| Activation | Initialization | Condition Number Behavior | $\lambda_{\min} < 0$ Injection | Escape Collapse |
| :--- | :--- | :--- | :--- | :--- |
| Swish (Smooth) | Normal | Rapid peak followed by plateau | Yes | ✓ Fast |
| Swish (Pseudo-collapse) | Collapse-like | Sharp peak $\rho_s = 0.609$ | Yes | ✓ Mechanistic |
| ReLU | Normal | Slow, no negative eigenvalues | No | ✗ Fail |
| ReLU | Pseudo-collapse | Static oscillation | No | ✗ Requires BN/Large LR |
| Linear | Pseudo-collapse | Slow drift | No | ✓ Final escape but slow |

Smooth activations actively inject negative eigenvalues to trigger a "topological phase transition," driving a surge in representation variance. ReLU lacks this mechanism and falls into collapse under continuous gradient flow without BatchNorm.

### Ablation Study: Orbit Compression + Information Entanglement

| Metric | Backbone $z$ | Projection Head $h(z)$ | Ratio | Description |
| :--- | :--- | :--- | :--- | :--- |
| Orbit Mean Square Spread ($\times 10^{-2}$) | 2.25 ± 1.07 | 0.10 ± 0.06 | 22.5× Compression | Prop 5.2 verified: Singularity in $\mathcal{V}_{\text{aug}}$ |
| Intra-orbit Distance $D_{\text{intra}}$ | 0.211 ± 0.045 | 0.044 ± 0.011 | 4.76× Reduction | Targeted compression of augmentations |
| Inter-class Distance $D_{\text{inter}}$ | 0.432 ± 0.052 | 0.111 ± 0.014 | 3.89× Reduction | Relative preservation of semantic structure |
| $D_{\text{inter}} / D_{\text{intra}}$ | 2.04 ± 0.52 | 2.50 ± 0.72 | 1.22× ↑ | Selective compression |
| Linear Probe Accuracy | 52.27% | 37.55% | -14.72 | Information cost of linear invariance |
| MLP Probe Accuracy | 55.46% | 43.56% | -11.90 | MLP-Linear gap doubled: Info entanglement |

Directly validates the core theory: the metric $G(z) = J_h(z)^\top J_h(z)$ learned by the projection head selectively induces metric singularities in the augmentation tangent space while relatively preserving semantic clustering. The significant increase in MLP probe advantage proves that information is not erased but non-linearly entangled, supporting the optimality of "discarding the projection head."

### Key Findings
- **Essential Difference between ReLU and Smooth Activations**: The non-zero $\nabla^2 h$ of smooth activations is key to escaping collapse; ReLU fails because its second derivative is zero almost everywhere.
- **22.5× Orbit Compression**: Directly proves that the projection head induces a near-singular metric in the $\mathcal{V}_{\text{aug}}$ direction, whereas semantic distance is only reduced by 3.89×—selective rather than global compression.
- **Doubled MLP-Linear Gap**: The downstream linear separability of the projection head output decreases, but MLP separability recovers, indicating that information is non-linearly entangled rather than erased—supporting the "discarding the head" strategy.

## Highlights & Insights
- **Unified Geometric Framework**: Uses Riemannian metric tensors to provide a unified explanation for the three roles of the projection head (whitening, escaping collapse, inducing singularity), offering deeper insight than information-theoretic or optimization-only perspectives.
- **ReLU Theoretical Gap**: Identifies the fundamental difference between smooth activations and ReLU under continuous gradient flow, explaining why Swish/GELU are preferred over ReLU in practice.
- **Geometric Explanation of Metric Singularity**: Theorems 5.1–5.3 quantify "why discard the head"—the head induces metric degradation in the $\mathcal{V}_{\text{aug}}$ direction, making backbone representations superior for downstream tasks.
- **Quantification of Capacity Thresholds**: Proposition 3.4 elevates the question of "how deep/wide a head should be" from empirical heuristics to a topological theorem.

## Limitations & Future Work
- The theory proves that projection heads *can* express an optimized landscape, but how SGD dynamics reach these configurations remains unknown.
- Extension to data-dependent metrics induced by ViT self-attention is still lacking.
- Discrete augmentations (e.g., flips) lack smooth group structures, making infinitesimal modeling inapplicable.
- Assumes the projection head reaches equilibrium rapidly, but the time-scale coupling between the head and backbone in actual training may not satisfy this.

## Related Work & Insights
- **vs. Information Bottleneck (Tishby)**: Both suggest the head filters irrelevant variables; this work adds the geometric mechanism (metric singularity), moving from "what is filtered" to "how it is filtered."
- **vs. Dimension Collapse Defense** (Jing 2022; Tian 2021): Prior work focused on BatchNorm/Stop-gradient; this work proves that the intrinsic curvature of smooth activations alone is sufficient to escape collapse.
- **vs. Natural Gradient Descent** (Amari 1998): The dynamic metric learned by the head is exactly the geometric foundation of the natural gradient.
- **vs. Explicit Whitening** (VICReg, Barlow Twins): Explicit whitening relies on loss constraints, while implicit whitening relies on the projection head metric; this paper unifies their geometric essence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Reinterprets projection heads through Riemannian geometry, introducing a conceptual system of metric tensors, orbit compression, and curvature injection—a disruptive innovation for SSL theory.
- Experimental Thoroughness: ⭐⭐⭐⭐  Hessian tracking, orbit visualization, and base model validation are complete; however, boundary cases for discrete augmentations and large-scale data are missing.
- Writing Quality: ⭐⭐⭐⭐⭐  The theorems are precisely stated, the proof logic is clear, and the visualizations are intuitive and powerful.
- Value: ⭐⭐⭐⭐⭐  Resolves two classic mysteries in SSL and provides geometric grounds for algorithm design (activation choice, head depth).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)
- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[NeurIPS 2025\] SegMASt3R: Geometry Grounded Segment Matching](../../NeurIPS2025/self_supervised/segmast3r_geometry_grounded_segment_matching.md)

</div>

<!-- RELATED:END -->
