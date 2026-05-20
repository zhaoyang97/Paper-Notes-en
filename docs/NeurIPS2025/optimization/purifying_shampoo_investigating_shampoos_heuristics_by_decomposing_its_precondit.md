---
title: >-
  [Paper Note] Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner
description: >-
  [NeurIPS 2025][Optimization][Shampoo optimizer] By decomposing Shampoo's preconditioner into eigenvalue and eigenbasis components, this work reveals that learning rate grafting essentially compensates for eigenvalue stal…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Shampoo optimizer"
  - "preconditioner"
  - "Kronecker factor"
  - "learning rate grafting"
  - "adaptive eigenbasis update"
date: 2026-05-08
content_hash: 062a5d0e5eae48d4
---

# Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner

**Conference**: NeurIPS 2025
**arXiv**: [2506.03595](https://arxiv.org/abs/2506.03595)  
**Code**: Unavailable  
**Area**: Optimization
**Keywords**: Shampoo optimizer, preconditioner, Kronecker factor, learning rate grafting, adaptive eigenbasis update

## TL;DR

By decomposing Shampoo's preconditioner into eigenvalue and eigenbasis components, this work reveals that learning rate grafting essentially compensates for eigenvalue staleness and scaling bias, and proposes eigenvalue correction and adaptive eigenbasis update frequency as principled replacements for these heuristics.

## Background & Motivation

Shampoo is a preconditioned stochastic gradient optimizer based on Kronecker factor decomposition. Its recent victory in the external tuning track of the AlgoPerf neural network training competition has renewed interest in non-diagonal preconditioning methods.

**Core Limitations of Prior Work**: Shampoo's success relies heavily on two key heuristics: (1) **learning rate grafting**—rescaling Shampoo's update direction using the per-layer update magnitude from Adam; and (2) **stale preconditioning**—recomputing the inverse root of the preconditioner only every 100 steps. These two techniques increase algorithmic complexity, introduce additional hyperparameters, and lack theoretical justification. Empirically, Shampoo without grafting underperforms across a wide range of hyperparameter settings, sometimes even falling behind AdamW.

**Limitations of Prior Work**: Although SOAP proposed an eigenvalue-corrected variant of Shampoo with favorable results, a systematic understanding of what role grafting plays, why it is necessary, and how to replace it on principled grounds has been lacking.

**Core Idea**: Decouple the preconditioner update into eigenvalue and eigenbasis components. Grafting essentially compensates for eigenvalue staleness and the scaling bias induced by the Kronecker structure. Directly correcting the eigenvalues eliminates the need for grafting; adaptively determining eigenbasis update frequency improves computational efficiency.

## Method

### Overall Architecture

Shampoo approximates the $mn \times mn$ preconditioner of full-matrix Adam as a Kronecker product of two smaller matrices: $\mathbf{C}_t^{\text{Shampoo}} = (\mathbf{R}_t \otimes \mathbf{L}_t)^{1/2}$, where $\mathbf{L}_t \in \mathbb{R}^{m \times m}$ and $\mathbf{R}_t \in \mathbb{R}^{n \times n}$. This paper analyzes the sources of error in this approximation through the lens of eigendecomposition.

### Key Designs

1. **Decoupled Analysis of Eigenvalues and Eigenbasis**: Shampoo's update can be decomposed into three steps: (a) rotating the gradient into a new coordinate system via the eigenbasis $\tilde{\mathbf{G}}_t = \mathbf{Q}_{\mathbf{L}_t}^\top \mathbf{G}_t \mathbf{Q}_{\mathbf{R}_t}$; (b) applying coordinate-wise scaling to the rotated gradient; and (c) rotating back to the original coordinate system. Lemma 1 establishes that the update magnitude is entirely determined by the eigenvalues (the eigenbasis does not affect magnitude), and that Shampoo's magnitude bound carries an additional dimension-dependent factor $m^{-p/2} n^{-p/2}$:

$$m^{-p/2}n^{-p/2}\lambda_{\max}^{-p}\|\mathbf{G}\|_F \leq \|\mathbf{U}^{\text{Shampoo}}\|_F \leq m^{-p/2}n^{-p/2}\lambda_{\min}^{-p}\|\mathbf{G}\|_F$$

whereas the magnitude bounds of full-matrix Adam and EShampoo contain no such dimension factors. This explains why grafting is so critical for Shampoo—it compensates for the inter-layer update magnitude inconsistency caused by varying layer dimensions.

2. **Eigenvalue Correction (EShampoo)**: Rather than constraining the eigenvalues to a Kronecker product structure, a dense scaling matrix is maintained independently in the rotated coordinate system: $\mathbf{D}_t = \beta_2 \mathbf{D}_{t-1} + (1 - \beta_2) \tilde{\mathbf{G}}_t^{\odot 2}$, where $\tilde{\mathbf{G}}_t^{\odot 2}$ denotes the element-wise square of the transformed gradient. This enables per-step "eigenvalue" updates at $O(mn)$ cost, entirely eliminating the need for grafting. Three theoretical predictions are empirically validated: (a) Shampoo2 (a tighter Kronecker approximation) with $S^{-1}$ scaling can substitute for grafting; (b) eigenvalue correction can match or exceed grafted Shampoo; and (c) the optimal learning rate for Adam transfers directly to EShampoo.

3. **Adaptive Eigenbasis Update Frequency**: An adaptive criterion based on the QR algorithm's termination condition is proposed. For each Kronecker factor $\mathbf{L}_t$, the approximate eigenvalue matrix $\hat{\Lambda}_{\mathbf{L}_t} = \hat{\mathbf{Q}}_{\mathbf{L}_{t-1}}^\top \mathbf{L}_t \hat{\mathbf{Q}}_{\mathbf{L}_{t-1}}$ is computed; if the relative off-diagonal error falls below threshold $\tau$:

$$\frac{\|\hat{\Lambda}_{\mathbf{L}_t} - \text{diag}(\hat{\Lambda}_{\mathbf{L}_t})\|_F}{\|\hat{\Lambda}_{\mathbf{L}_t}\|_F} \leq \tau$$

the eigenbasis update is skipped and the previous eigenbasis is reused. This allows different layers and different Kronecker factors to have independent update frequencies.

### Loss & Training

Standard neural network training objective $\min_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$. EShampoo updates the eigenvalue correction matrix $\mathbf{D}_t$ at every step (same memory overhead as Adam grafting) and updates the eigenbasis at a fixed or adaptive frequency via `torch.linalg.eigh` or a warm-started QR algorithm.

## Key Experimental Results

### Main Results (AlgoPerf Subset Workloads)

| Workload | Shampoo Variant | Pass Rate | Steps | Time (min) |
|---------|-----------|-------|------|----------|
| FastMRI | Adam grafting (F=100) | 4/5 | 4301±109 | 13.96±0.44 |
| FastMRI | EShampoo (F=100) | 5/5 | 2536±66 | 10.44±0.21 |
| FastMRI | EShampoo (τ=0.1, F=50) | 5/5 | 2468±145 | 10.81±0.72 |
| ImageNet ViT | Adam grafting (F=100) | 1/1 | 79907 | 894.27 |
| ImageNet ViT | EShampoo (F=100) | 1/1 | 76226 | 894.85 |
| OGBG | Adam grafting (F=100) | 2/5 | 12574±708 | 39.20±1.88 |
| OGBG | EShampoo (F=100) | 3/5 | 8320±1203 | 33.02±4.05 |
| OGBG | EShampoo (τ=0.1, F=50) | 5/5 | 7117±328 | 27.55±3.49 |

### Ablation Study (Imagewoof ViT / ConvNeXt V2)

| Configuration | Result | Remark |
|------|------|------|
| Shampoo (F=1, no grafting) | Matches grafted on ViT; not on ConvNeXt V2 | Validates scaling issues from dimension differences |
| Shampoo2 + $S^{-1}$ (F=1) | Matches or exceeds grafted version | Tighter Frobenius approximation |
| EShampoo (F=100) | Matches or exceeds grafted version | Eigenvalue correction eliminates need for grafting |
| τ=0.99 (eigenbasis computed only once) | Significantly outperforms AdamW (0.12 vs 0.3) | Single eigenbasis computation already yields large gains |
| 1D params with Adam + 2D params with EShampoo | Equivalent to full EShampoo | Eigenbasis updates for 1D parameters are unnecessary |
| 2D params with Adam + 1D params with EShampoo | Close to pure Adam | All gains stem from eigenbasis of 2D parameters |

### Key Findings

- **Eigenbasis update importance is concentrated in early training**: Using high-precision eigenbasis (τ=0.01) in the first 10% of iterations has a far greater impact on final loss than in the remaining 90%
- **Eigenbasis change rates differ across parameter types**: Bias and LayerNorm parameters exhibit eigenbasis drift approximately four times faster than weight matrices
- **Adaptive frequency reduces wall-clock time by approximately 20% compared to fixed F=100**

## Highlights & Insights

- **Depth of the decoupled perspective**: Grafting, previously a black-box heuristic, is reduced to an eigenvalue correction problem with both theoretical justification (dimension-dependent magnitude bounds) and empirical validation
- **Strong practicality**: EShampoo requires only an $mn$-dimensional buffer on top of Shampoo (the same overhead as grafting), yet eliminates grafting and achieves superior performance
- **Hyperparameter transfer from AdamW to EShampoo**: This is practically significant, reducing the tuning cost of adopting a new optimizer
- **The finding that "frozen eigenbasis still outperforms AdamW"** is surprising and practically valuable

## Limitations & Future Work

- Main experiments are conducted on small-to-medium-scale models; thorough validation on large language models is lacking (only a 324M-parameter Llama 3 analysis is included)
- Adaptive update frequency slightly degrades performance on certain workloads (e.g., ImageNet ViT), indicating that the optimal task-specific strategy remains to be explored
- An open theoretical question remains unresolved: how to incorporate approximation quality into Shampoo's regret bound
- Applicability to non-AdaGrad-family methods (e.g., K-FAC, TNT) is not extensively discussed

## Related Work & Insights

- This work is closely related to SOAP but provides deeper theoretical analysis and more systematic experiments
- The QR-based criterion for adaptive eigenbasis updates draws from matrix factorization theory in numerical linear algebra
- The approach of "purifying" Kronecker factor methods is generalizable to all Kronecker approximation-based optimizers

## Rating

- Novelty: ⭐⭐⭐⭐ The decoupled perspective is original, though eigenvalue correction itself already appeared in SOAP
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers AlgoPerf workloads and diverse model architectures with detailed ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are presented progressively, with clear correspondence between experimental validation and theoretical predictions
- Value: ⭐⭐⭐⭐⭐ Contributes significantly to understanding and improving Shampoo-class optimizers with strong practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](../../AAAI2026/optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[NeurIPS 2025\] Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs](learning_sparse_approximate_inverse_preconditioners_for_conjugate_gradient_solve.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains](memory-augmented_potential_field_theory_a_framework_for_adaptive_control_in_non-.md)

</div>

<!-- RELATED:END -->
