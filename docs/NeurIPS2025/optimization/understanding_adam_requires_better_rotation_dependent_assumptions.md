---
title: >-
  [Paper Note] Understanding Adam Requires Better Rotation Dependent Assumptions
description: >-
  [NeurIPS 2025][Optimization][Adam optimizer] Through systematic empirical investigation, this paper demonstrates that the Adam optimizer exhibits strong dependence on the choice of coordinate basis in parameter space, showing that existing rotation-invariant theoretical assumptions are insufficient to explain Adam's superiority. The orthogonality of per-layer updates is identified as a reliable indicator for predicting Adam's performance under different bases.
tags:
  - NeurIPS 2025
  - Optimization
  - Adam optimizer
  - rotation equivariance
  - parameter space rotation
  - orthogonality
  - adaptive optimization
date: 2026-05-08
content_hash: 3daad626f5bc098e
---

# Understanding Adam Requires Better Rotation Dependent Assumptions

**Conference**: NeurIPS 2025
**arXiv**: [2410.19964](https://arxiv.org/abs/2410.19964)
**Code**: Unavailable
**Area**: Optimization
**Keywords**: Adam optimizer, rotation equivariance, parameter space rotation, orthogonality, adaptive optimization

## TL;DR

Through systematic empirical investigation, this paper demonstrates that the Adam optimizer exhibits strong dependence on the choice of coordinate basis in parameter space, showing that existing rotation-invariant theoretical assumptions are insufficient to explain Adam's superiority. The orthogonality of per-layer updates is identified as a reliable indicator for predicting Adam's performance under different bases.

## Background & Motivation

Although Adam is the standard optimizer for training Transformer models, a comprehensive theoretical explanation of its advantage over SGD remains elusive. SGD is rotation-equivariant—rotating the parameter space yields a correspondingly rotated optimization trajectory—whereas Adam lacks this property due to its element-wise division operation.

Most theoretical assumptions used in the literature to analyze Adam are **rotation-invariant** (e.g., bounded gradient variance, Lipschitz smoothness), meaning these analyses provide identical convergence guarantees for any basis and thus fail to explain why Adam performs better in the standard basis. The authors pose a central question: **how does the choice of basis in parameter space affect Adam's performance?** and **can existing rotation-dependent assumptions account for these effects?**

Furthermore, recent optimizers such as SOAP and Muon achieve practical gains by optimizing in rotated parameter spaces, yet these rotations are designed more by intuition than by systematic theory, further motivating a principled understanding of the relationship between rotation and Adam.

## Method

### Overall Architecture

The methodological core of this paper is not the proposal of a new algorithm, but rather the design of a systematic experimental framework to study Adam's behavior in rotated parameter spaces. Concretely, forward and backward passes are performed in the standard basis, but gradients are rotated into a target space before Adam computes the update, which is then rotated back to the standard space and applied to the parameters.

### Key Designs

1. **Multi-scale rotation scope study**: The authors define four types of random rotations with different scopes—global rotation (entire parameter space), layer-wise rotation (within each layer's subspace), input-side rotation (weights of the same input neuron), and output-side rotation (weights of the same output neuron). Experiments show that broader rotation scopes lead to greater performance degradation. For GPT-2 (124M), global rotation causes a ~16% training slowdown; for ViT/S (22M), the slowdown reaches ~96%. Output-side rotation, however, has virtually no negative effect and even slightly improves performance, indicating that Adam's adaptivity within output neurons is relatively weak.

2. **SVD-based structured rotation**: Inspired by GaLore, the authors apply SVD decomposition $\mathbf{G} = \mathbf{U}\mathbf{S}\mathbf{V}^\top$ to the per-layer gradient matrix $\mathbf{G}$, and run Adam in the rotated space $\mathbf{U}^\top \mathbf{G} \mathbf{V}$. This structured rotation not only avoids performance degradation but **significantly improves** GPT-2 training, demonstrating that coordinate systems superior to the standard basis exist.

3. **Per-layer update orthogonality metric**: To identify a rotation-dependent quantity that predicts Adam's performance, the authors measure the coefficient of variation (CV) of the singular values of the per-layer weight update $\mathbf{A} = \mathbf{R}^\top \mathbf{M}_t^{(\mathbf{R})} / (\sqrt{\mathbf{V}_t^{(\mathbf{R})}} + \epsilon)$. A lower CV indicates that the update is closer to a scaled orthogonal matrix. Experiments reveal high consistency between CV and Adam's performance: SVD rotation → lowest CV → best performance; global random rotation → highest CV → worst performance.

### Evaluation of Existing Rotation-Dependent Assumptions

The authors systematically examine three classes of rotation-dependent assumptions from the literature:

- **$L_\infty$ gradient bound**: Under global rotation, $\tilde{C}$ decreases substantially (suggesting better performance), yet Adam's actual performance degrades, indicating that this assumption points in the **opposite** direction.
- **Block-diagonal Hessian**: While the Hessian does exhibit an approximately block-diagonal structure in the standard basis, quantitative analysis shows that off-diagonal blocks, though numerically small, play a **dominant role** in gradient variation due to their much higher dimensionality. Strict block-diagonal approximation is therefore an oversimplification.
- **$(1,1)$-norm / $L_\infty$ smoothness**: Correlates with performance under global and SVD rotations but fails under output-side rotation (where performance slightly improves while the norm decreases), suggesting limited reliability.

## Key Experimental Results

### Main Results

| Model | Rotation Type | Training Slowdown | Performance Change |
|-------|--------------|-------------------|-------------------|
| GPT-2 (124M) | Global | ~16% | Significant degradation |
| GPT-2 (124M) | Layer-wise | ~8% | Moderate degradation |
| GPT-2 (124M) | Output-side | ~0% | Slight improvement |
| GPT-2 (124M) | SVD | — | Significant improvement |
| ViT/S (22M) | Global | ~96% | Severe degradation |
| ViT/S (22M) | Output-side | ~0% | No change |
| ResNet-50 | Global | Negligible | Almost no effect |

### Ablation Study

| Metric | Global Rotation | SVD Rotation | Output-side Rotation | Consistency with Adam Performance |
|--------|----------------|-------------|---------------------|----------------------------------|
| $L_\infty$ gradient bound | Decreases (↓) | — | — | ❌ Inconsistent |
| $(1,1)$-norm | Increases (↑) | Decreases (↓) | Increases (↑) | ⚠️ Partially consistent |
| Update orthogonality CV | High (poor) | Low (good) | Medium (normal) | ✅ Fully consistent |

### Key Findings

- ResNet is insensitive to rotation, which may explain why SGD does not underperform Adam on ResNet.
- Global rotation concentrates Adam's second-moment distribution, reducing effective learning rate diversity and thus weakening adaptivity.
- Updating the SVD rotation every 250 steps yields meaningful gains, and the CV exhibits a clear drop at each update step.
- Different parameter block types show varying sensitivity to rotation; Appendix C further analyzes the independent effects of K/Q/V projection layers in Transformers.

## Highlights & Insights

- **Core findings are highly illuminating**: The systematic characterization of the relationship between Adam's performance and basis choice reveals an important dimension that has largely been overlooked.
- **The orthogonality metric resonates with the Muon optimizer**: Muon achieves strong performance by approximately orthogonalizing gradient updates; this paper provides complementary evidence from the perspective of Adam.
- **Elegant experimental design**: The four rotation types spanning from local to global form a coherent spectrum, lending the study strong systematic rigor.

## Limitations & Future Work

- The work is primarily empirical; a **rigorous theoretical analysis** of why orthogonal updates are preferable is lacking.
- SVD rotation is intended as an analytical tool rather than a practical optimizer; its computational overhead limits direct application.
- It remains unexplained why Adam produces more orthogonal updates under SVD rotation.
- The reason for ResNet's low rotation sensitivity requires further evidence to confirm.

## Related Work & Insights

- Closely related to the Muon and SOAP optimizers, which improve performance through rotation; this paper lays the groundwork for a theoretical understanding of these approaches. Muon approximates orthogonalization of gradient matrices via Nesterov iteration, while SOAP applies Shampoo-style rotation to Adam.
- Points toward a direction for designing new rotation-aware optimizers: a good basis is one that makes per-layer updates more orthogonal.
- Challenges the block-diagonal Hessian assumption, offering important caveats for subsequent theoretical work.
- Bernstein (2025) argues from the perspective of linear layers that orthogonalized updates can control the scale of feature representations, reducing dependence on normalization layers.
- Connected to GaLore: GaLore uses low-rank SVD to compress optimizer states, and the full-rank SVD rotation in this paper can be viewed as its natural extension.
- Points to a new direction for convergence analysis of Adam—rotation-dependent theoretical frameworks are needed in place of traditional rotation-invariant assumptions.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic study of the relationship between Adam and coordinate basis choice; novel perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple models, rotation types, and assumption tests; highly comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear exposition, rich figures, convincing conclusions
- Value: ⭐⭐⭐⭐ Provides an important empirical foundation for understanding and improving adaptive optimizers

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] In Search of Adam's Secret Sauce](in_search_of_adams_secret_sauce.md)
- [\[NeurIPS 2025\] Exploring Landscapes for Better Minima along Valleys](exploring_landscapes_for_better_minima_along_valleys.md)
- [\[NeurIPS 2025\] Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry](projecting_assumptions_the_duality_between_sparse_autoencoders_and_concept_geome.md)
- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)

<!-- RELATED:END -->
