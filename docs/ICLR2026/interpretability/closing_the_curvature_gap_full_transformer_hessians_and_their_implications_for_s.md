---
title: >-
  [Paper Note] Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws
description: >-
  [ICLR 2026][Interpretability][Transformer Hessian] This paper presents the first explicit Hessian expressions and spectral norm upper bounds for a complete Transformer block (including LayerNorm and FFN)…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Transformer Hessian"
  - "LayerNorm"
  - "scaling laws"
  - "loss landscape"
  - "optimization theory"
date: 2026-05-08
content_hash: 9b349a7411f55a14
---

# Closing the Curvature Gap: Full Transformer Hessians and Their Implications for Scaling Laws

**Conference**: ICLR 2026  
**arXiv**: [2510.16927](https://arxiv.org/abs/2510.16927)  
**Code**: [https://github.com/modernTalker/transformer_hessian](https://github.com/modernTalker/transformer_hessian)  
**Area**: Interpretability  
**Keywords**: Transformer Hessian, LayerNorm, scaling laws, loss landscape, optimization theory

## TL;DR
This paper presents the first explicit Hessian expressions and spectral norm upper bounds for a complete Transformer block (including LayerNorm and FFN), and establishes a theoretical framework showing that the loss landscape converges at an $O(1/k)$ rate as data volume increases, providing a mathematical foundation for scaling laws and curvature-aware training.

## Background & Motivation

**Background**: The empirical success of Transformers is characterized by predictable improvements described by neural scaling laws. Prior work has derived Hessian expressions for self-attention, but second-order analysis of LayerNorm and FFN has remained absent.

**Limitations of Prior Work**: The lack of a complete Transformer block Hessian means: (1) it is impossible to fully understand how the optimization landscape evolves with data volume; (2) there is no theoretical account of how curvature propagates across sub-layers; (3) scaling laws lack a rigorous mathematical foundation.

**Key Challenge**: The nonlinearities in LayerNorm and FFN make second-order derivative derivations highly complex. Prior theoretical work was limited to self-attention, leaving a "curvature gap."

**Goal**: To derive the Jacobians and Hessians of a complete Transformer block — including LayerNorm and FFN — and to establish theoretical bounds on loss landscape convergence.

**Key Insight**: The paper employs a row-vectorization framework $\text{vec}_r(\cdot)$ and Gauss-Newton decomposition to systematically decompose the Hessian into per-sublayer contributions, deriving results layer by layer.

**Core Idea**: The paper closes the gap in second-order Transformer theory by explicitly deriving the Hessians of LayerNorm and FFN, and uses Taylor expansion to analyze the convergence behavior of the loss landscape as data volume grows.

## Method

### Overall Architecture

The theoretical derivation chain proceeds as follows: Self-Attention Hessian (prior work) → LayerNorm Jacobian/Hessian (Theorems 2–3) → ReLU FFN derivatives (Lemma 1) → Complete Transformer Block Hessian (Theorems 4–5) → Spectral norm upper bounds (Theorems 1, 6) → Loss landscape convergence theorem (Theorem 7).

The Transformer block (post-norm) is defined as:
$$\mathbf{Y} = \text{LayerNorm}(\mathbf{X} + \mathbf{F}(\mathbf{X}))$$
$$\mathbf{Z} = \text{LayerNorm}(\mathbf{Y} + \text{FFN}(\mathbf{Y}))$$

### Key Designs

1. **LayerNorm Jacobian and Hessian (Theorems 2–3)**:

    - Function: Derives the first- and second-order derivatives of LayerNorm with respect to its input.
    - Mechanism: LayerNorm is decomposed as $\text{LN}(\mathbf{X}) = \mathbf{P}(\mathbf{X})\mathbf{M}(\mathbf{X})$, where $\mathbf{M}$ denotes centering (mean subtraction) and $\mathbf{P}$ is the diagonal inverse-standard-deviation matrix. Applying the product rule yields a Jacobian consisting of two terms: the scaling of the centering operation by $\mathbf{P}$, and the contribution of changes in $\mathbf{P}$ to $\mathbf{M}$. The Hessian is obtained by further differentiating the Jacobian; since centering is linear, $\frac{\partial^2 \mathbf{M}}{\partial \mathbf{X}^2} = 0$, but the second-order derivative of $\mathbf{P}$ is non-zero.
    - Design Motivation: The Hessian of LayerNorm had not previously been derived. It contributes curvature through per-row variance and represents a critical missing component for understanding the Transformer optimization landscape.

2. **Complete Transformer Block Hessian (Theorems 4–5)**:

    - Function: Assembles the Hessian of the full block comprising self-attention, LayerNorm, FFN, and residual connections.
    - Mechanism: Let $\mathbf{S} = \text{ReLU}(\mathbf{Y}\mathbf{W}_1)\mathbf{W}_2 + \mathbf{Y}$ (FFN with residual) and $\mathbf{Z} = \text{LN}(\mathbf{S})$. Via the chain rule:
    $$\mathbf{H}_{\text{tr}}^{(i,j)} = (\mathbf{J}_Z \otimes \mathbf{I}_{n_i})\bm{\xi}_{ij} + (\mathbf{I}_{Ld_V} \otimes \mathbf{B}_i^\top)\mathbf{H}_Z\mathbf{B}_j$$
      where $\mathbf{J}_Z$ is the Jacobian of LayerNorm, $\mathbf{H}_Z$ is its Hessian, $\bm{\xi}_{ij}$ denotes the mixed second-order derivatives of $\mathbf{S}$, and $\mathbf{B}_i$ is the Jacobian of $\mathbf{S}$ with respect to the parameters.
    - Design Motivation: The Gauss-Newton decomposition allows the loss Hessian to be separated into an "outer product term" (first-order information) and a "function Hessian term" (second-order information), each corresponding to distinct optimization properties.

3. **Spectral Norm Upper Bounds (Theorems 1, 6)**:

    - Function: Provides explicit spectral norm upper bounds for the Hessians of self-attention and the complete Transformer block.
    - Mechanism: Using the sub-multiplicativity of matrix norms and Kronecker product properties, the Hessian norm bound is decomposed as a function of input norm $\|\mathbf{X}\|_2$, weight norms $\|\mathbf{W}\|_2$, sequence length $L$, and dimensions $d_V, d_K$. The full-block bound satisfies $\leq 5\max_{i,j}(\cdots)$, where 5 corresponds to $\sqrt{m_b n_b}$ across 5 parameter groups.
    - Design Motivation: The explicit bounds reveal each sublayer's contribution to overall curvature — Value- and Key-related terms dominate through softmax derivatives, FFN curvature is controlled by the piecewise linearity of ReLU (whose Hessian is almost everywhere zero), and LayerNorm contributes through per-row variance.

4. **Loss Landscape Convergence Theorem (Theorem 7)**:

    - Function: Establishes the convergence rate of the loss function as data volume increases.
    - Mechanism: Using Taylor expansion and the Hessian bounds, the theorem proves:
    $$|\mathcal{L}_{k+1}(\mathbf{w}) - \mathcal{L}_k(\mathbf{w})| \leq \frac{2L}{k+1} + \frac{M\|\mathbf{w} - \mathbf{w}^*\|_2^2}{k+1}$$
      where $M$ is derived from the spectral norm bounds of Theorems 1 and 6. This shows that changes in the loss landscape decay at an $O(1/k)$ rate.
    - Design Motivation: This provides a theoretical explanation for the empirical observation that the loss landscape stabilizes as data volume grows, and offers a principled criterion for determining when to shift from data scaling to model scaling.

### Loss & Training

- Theoretical analysis employs MSE loss: $l(\cdot, \text{Target}) = \frac{1}{Ld_V}\|\cdot - \text{Target}\|_F^2$
- Experiments are conducted on ViT, trained on MNIST (1 block, dim=16) and CIFAR-100 (8 blocks, dim=128).

## Key Experimental Results

### Main Results

Hessian structure validation (MNIST, 1 Transformer block):

| Observation | Result |
|------|------|
| Hessian at initialization | Entry magnitudes are highly non-uniform; Value-related block is largest |
| Hessian after training | Magnitudes increase across all blocks; Value-Value block remains dominant |
| Parameter block norm ranking | Key, Value >> Query, W1, W2 |

### Ablation Study

Loss landscape convergence validation (CIFAR-100, 8 blocks, log-log scale):

| Data volume $k$ | Trend of $|\mathcal{L}_{k+1} - \mathcal{L}_k|$ |
|---------|--------------------------------------|
| Small $k$ | Large variation, unstable |
| Large $k$ | Approximately linear decay (log-log), consistent with $O(1/k)$ |

### Key Findings
- **Value-Value Hessian dominates**: The Value-Value block exhibits the largest magnitude both before and after training, indicating that the Value matrix has the highest curvature and exerts the greatest influence on optimization — theoretically explaining why Adam is important for Transformers, as the vast curvature differences across parameters necessitate adaptive learning rates.
- **FFN Hessian is governed by ReLU**: The second-order derivative of ReLU is almost everywhere zero, so FFN curvature arises primarily from combinations of first-order terms rather than its own nonlinearity.
- **LayerNorm contributes curvature through per-row variance**: Smaller variance (more similar features) leads to larger LayerNorm curvature, which may induce training instability.
- **$O(1/k)$ convergence rate validated experimentally**: Log-log plots on CIFAR-100 show that the EMA of loss differences decays approximately linearly, consistent with the theoretical prediction.

## Highlights & Insights
- **Fills a critical theoretical gap**: While the Hessian of self-attention was previously known, the absence of analysis for LayerNorm and FFN left prior theoretical treatments incomplete. The full derivation in this paper enables end-to-end curvature analysis.
- **Practical implications of heterogeneous block Hessians**: The dramatic curvature differences across parameter blocks (Value >> Query) theoretically motivate per-block learning rates and curvature-aware preconditioning.
- **$O(1/k)$ convergence informs data budgeting**: When curvature stabilizes, the marginal returns of additional data diminish; the theory suggests that this inflection point marks a principled transition from data scaling to model scaling.

## Limitations & Future Work
- **Local analysis**: The Taylor expansion and Assumption 1 (shared minimizer) hold only locally, limiting the characterization of the global optimization landscape.
- **Single-block analysis**: The theoretical derivations target a single Transformer block and do not extend to multi-layer stacking; inter-layer Hessian propagation is not analyzed.
- **Post-norm and MSE only**: The theory is developed for post-norm (whereas modern LLMs typically use pre-norm) and MSE loss (whereas cross-entropy is used in practice). Although the paper claims extensibility to cross-entropy loss, no formal derivation is provided.
- **Limited experimental scale**: The ViT models on MNIST and CIFAR-100 are far smaller than modern LLMs; whether the theoretical predictions remain accurate at large scale has yet to be verified.
- **$M$ is not a true constant**: The $M$ in Theorem 7 depends on $\|\mathbf{X}\|_2$ and varies with the data, so the $O(1/k)$ characterization is not fully rigorous in the strict sense.

## Related Work & Insights
- **vs. Zhang et al. (NeurIPS 2024, "Why Transformers Need Adam")**: Similarly analyzes Transformer optimization from a Hessian perspective, but the present work derives the complete block Hessian (including LayerNorm and FFN), offering a more comprehensive treatment.
- **vs. Ormaniec et al. (2024)**: Analyzes the Hessian decomposition of the self-attention block; the present work extends this to the full Transformer block.
- **vs. Kaplan et al. / Hoffmann et al. (Scaling Laws)**: Those works establish empirical scaling laws; the present work provides theoretical tools for understanding scaling from a curvature perspective.

## Rating
- Novelty: ⭐⭐⭐⭐ First complete derivation of the Transformer Hessian, filling a well-defined theoretical gap.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are limited to small models on MNIST/CIFAR-100, lacking large-scale empirical support.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and the structure is clear; however, the density of notation makes the paper less accessible to readers outside the theory community.
- Value: ⭐⭐⭐⭐ Establishes a new foundation for Transformer optimization theory, though practical utility at scale remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[NeurIPS 2025\] Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](../../NeurIPS2025/interpretability/sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)
- [\[ICLR 2026\] Noise Stability of Transformer Models](noise_stability_of_transformer_models.md)
- [\[CVPR 2026\] Pixel2Phys: Distilling Governing Laws from Visual Dynamics](../../CVPR2026/interpretability/pixel2phys_distilling_governing_laws_from_visual_dynamics.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)

</div>

<!-- RELATED:END -->
