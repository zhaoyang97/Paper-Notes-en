---
title: >-
  [Paper Note] Wavy Transformer
description: >-
  [NeurIPS 2025][Graph Learning][Transformer] This paper establishes a formal equivalence between Transformer attention layers and graph neural diffusion on complete graphs, and proposes the Wavy Transformer based on second-order wave equations. By exploiting energy conservation properties, the method mitigates over-smoothing in deep Transformers and achieves consistent improvements across NLP, CV, and sparse graph tasks.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Transformer
  - over-smoothing
  - wave equation
  - graph neural diffusion
  - attention mechanism
  - physics-inspired
date: 2026-05-08
content_hash: ad8b365a6e56d10b
---

# Wavy Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2508.12787](https://arxiv.org/abs/2508.12787)
**Authors**: Satoshi Noguchi (JAMSTEC/RIKEN), Yoshinobu Kawahara (Osaka Univ/RIKEN)
**Code**: [GitHub](https://github.com/noguchisatoshi/Wavy-Transformer)
**Area**: Graph Learning
**Keywords**: Transformer, over-smoothing, wave equation, graph neural diffusion, attention mechanism, physics-inspired

## TL;DR

This paper establishes a formal equivalence between Transformer attention layers and graph neural diffusion on complete graphs, and proposes the Wavy Transformer based on second-order wave equations. By exploiting energy conservation properties, the method mitigates over-smoothing in deep Transformers and achieves consistent improvements across NLP, CV, and sparse graph tasks.

## Background & Motivation

### State of the Field
Deep Transformer models commonly suffer from **over-smoothing**: as network depth increases, all token representations converge toward a uniform state, causing deeper Transformers to not necessarily outperform shallower ones. This issue has been extensively studied in the GNN literature but remains insufficiently explored in the Transformer context.

### Limitations of Prior Work
- Existing methods for mitigating over-smoothing in Transformers (e.g., FeatScale) primarily rely on externally injecting high-frequency perturbations into hidden states to prevent convergence.
- There is a lack of analysis on the **intrinsic dynamical mechanisms** underlying over-smoothing in Transformers.
- Wave-equation-based methods (Graph-CON, PDE-GCN) have been developed for GNNs but have not been transferred to Transformer architectures.

### Core Motivation
From the perspective of physical dynamical systems, the paper interprets the hidden-state dynamics of attention layers as a diffusion process on a complete graph. It then leverages the energy conservation and oscillatory properties of wave equations to fundamentally alter the intrinsic dynamics of Transformers, thereby addressing over-smoothing at its root.

## Method

### Key Insight: Attention as Graph Neural Diffusion

The graph neural diffusion equation on a complete graph is defined as:

$$\frac{\partial \mathbf{X}}{\partial t} = (\mathbf{A} - \mathbf{I})\mathbf{X}$$

where $\mathbf{A}$ is the attention matrix (a row-stochastic matrix) and $\mathbf{I} - \mathbf{A}$ can be viewed as a normalized graph Laplacian. Discretizing in time yields:

$$\mathbf{X}^{l+1} = \tau \mathbf{A}\mathbf{X}^l + (1-\tau)\mathbf{X}^l$$

At $\tau = 1/2$, neglecting the scaling effect of layer normalization, this expression is essentially equivalent to the standard attention residual update $\mathbf{X}^{l+1} = \mathbf{A}\mathbf{X}^l + \mathbf{X}^l$. This reveals that **conventional Transformers implicitly perform a diffusion process**, and the dissipative nature of this process is the root cause of over-smoothing.

### Wave-Dynamical Attention

Based on the wave equation on a complete graph, a velocity variable $\mathbf{Y} = \frac{\partial \mathbf{X}}{\partial t}$ is introduced to reformulate the second-order equation as a first-order system:

$$\mathbf{Y}^{l+1} = \tau(\mathbf{A} - \mathbf{I})\mathbf{X}^l + \mathbf{Y}^l, \quad \mathbf{X}^{l+1} = \tau \mathbf{Y}^{l+1} + \mathbf{X}^l$$

This discrete system is **symplectic** and preserves system energy. Compared to pure diffusion updates, the wave update additionally incorporates a momentum term $(\mathbf{X}^l - \mathbf{X}^{l-1})$, preventing excessive feature smoothing.

### Mixed Residual Connection

A learnable mixture of diffusion and wave dynamics is supported: $\mathbf{X}^{l+1} = \boldsymbol{\lambda} \mathbf{X}_{\text{wave}}^{l+1} + (1-\boldsymbol{\lambda}) \mathbf{X}_{\text{diffuse}}^{l+1}$, where $\boldsymbol{\lambda} = \text{sigmoid}(\boldsymbol{\theta}) \in [0,1]^d$ is a trainable parameter vector.

### Physically Consistent Layer Normalization and FFN

To maintain the physical consistency of the state–velocity relationship $\mathbf{Y} = \frac{\partial \mathbf{X}}{\partial t}$ under the chain rule:

- **Velocity Layer Normalization $\text{LN}_v$**: Retains only the scaling parameters ($\sigma^2, \gamma$), removes the shift parameters ($\mu, \beta$), and normalizes the velocity $\mathbf{Y}$ using the mean and variance of the state $\mathbf{X}$.
- **Velocity FFN**: $\text{FFN}_v(\mathbf{Y}^l) = \phi'(\mathbf{X}^l \mathbf{W}_1 + \mathbf{b}_1) \mathbf{Y}^l \mathbf{W}_1 \mathbf{W}_2$, scaled by the derivative $\phi'$ of the activation function.

### Two Variants
- **Full Wave**: Includes a complete velocity branch (FFN + LN), imposing stronger physical constraints at a slightly higher computational cost.
- **Light Wave**: Retains only the momentum term $\boldsymbol{\lambda}(\mathbf{X}^l - \mathbf{X}^{l-1})$ without additional FFN/LN, incurring negligible overhead.

## Key Experimental Results

### Experiment 1: NLP Tasks (BERT Pre-training + GLUE Fine-tuning)

| Residual Type | PPL (↓) | MLM Acc (↑) | GLUE Avg (↑) | STS-B |
|--------------|---------|-------------|-------------|-------|
| Diffusion | 31.76 | 44.39% | 64.13 | 52.11 |
| Full Wave | 31.99 | 44.52% | 62.27 | 32.91 |
| Mix (+Full) | 29.00 | **45.56%** | 62.44 | 29.40 |
| Mix (+Light) | 32.29 | 44.53% | **66.12** | **64.76** |

The mixed residual outperforms the pure diffusion baseline on both PPL and GLUE average score. Mix (+Light) achieves a GLUE average gain of +1.99 and an STS-B gain of +12.65.

### Experiment 2: CV Tasks (ImageNet Classification) and Sparse Graph Tasks

**ImageNet Classification (DeiT/CaiT):**

| Method | Residual | Layers | Params | Top-1 Acc (%) |
|--------|----------|--------|--------|--------------|
| DeiT-Ti | Diffusion | 12 | 5.7M | 72.17 |
| DeiT-Ti | + Full Wave | 12 | 5.7M | 72.33 (↑0.16) |
| DeiT-Ti | + Light Wave | 12 | 5.7M | **73.09** (↑0.92) |
| DeiT-Ti + FeatScale | Diffusion | 12 | 5.7M | 72.35 |
| DeiT-Ti + FeatScale | + Full Wave | 12 | 5.7M | 72.62 (↑0.26) |
| CaiT-XXS-24 | Diffusion | 24 | 12.0M | 77.6 |
| CaiT-XXS-24 | + Full Wave | 24 | 11.1M | **78.6** (↑1.0) |

**Sparse Graph Tasks (DIFFormer):**

| Dataset | Metric | Layers | Diffusion | + Light Wave | Δ |
|---------|--------|--------|-----------|-------------|---|
| OGBN-Arxiv | Acc | 7 | 24.44±4.51 | **66.73±0.33** | **+42.29** |
| OGBN-Proteins | ROC-AUC | 5 | 69.42±2.31 | **80.14±0.67** | **+10.72** |

Improvements are particularly pronounced on sparse graph tasks: on OGBN-Arxiv with 7 layers, accuracy increases from 24.44% to 66.73%, demonstrating that wave residuals effectively mitigate deep-layer collapse.

### Over-Smoothing Diagnostics

| Dynamics | Spectral Gap (↓) | Node Feature Variance (↑) | Inter-class Variance |
|----------|-----------------|--------------------------|---------------------|
| Diffusion | 0.836±0.003 | 2.480±0.078 | 0.195 |
| + Full Wave | 0.629±0.009 | 2.609±0.090 | 0.211 |
| + Light Wave | 0.730±0.008 | 2.109±0.070 | **0.308** |

### Computational Efficiency

| Model | Variant | Inference | Training | Peak GPU Memory |
|-------|---------|-----------|----------|----------------|
| BERT | Diffusion | 101.6 | 415.6 | 18.31 |
| BERT | Light Wave | 101.3 | 436.2 | 18.69 |
| DeiT-Tiny | Diffusion | 2631.1 | 618.6 | 8.25 |
| DeiT-Tiny | Light Wave | 2644.2 | 617.6 | 9.14 |

The Light Wave variant matches the baseline in inference speed, training throughput, and memory overhead (differences within a few percentage points).

## Highlights & Insights

- **Elegant theoretical insight**: This work is the first to rigorously establish the equivalence between attention layers and graph neural diffusion on complete graphs, providing a clear physical interpretation of Transformer over-smoothing (rooted in the dissipative nature of diffusion).
- **Plug-and-play**: The Wavy Transformer block integrates seamlessly into existing Transformer architectures (BERT, DeiT, CaiT, DIFFormer) without additional hyperparameter tuning and with negligible parameter overhead.
- **Consistent cross-domain gains**: Improvements are demonstrated across NLP, CV, and sparse graph tasks, validating the generality of the approach.
- **Extreme efficiency of Light Wave**: Only a single learnable vector $\boldsymbol{\lambda} \in \mathbb{R}^d$ is required; significant gains are achieved through the momentum term alone.
- **Physically consistent design**: The velocity-specific LN and FFN are derived via the chain rule, preserving the physical self-consistency of the state–velocity relationship.

## Limitations & Future Work

- **Theory–practice gap**: Although the wave equation is theoretically energy-conserving, the introduction of $\boldsymbol{\lambda}$ in the mixed residual breaks the strict symplectic structure, weakening the physical interpretation.
- **Limited experimental scale**: NLP experiments use a smaller pre-training configuration than standard BERT (10k steps, batch size 64); large-scale pre-training effects remain unvalidated.
- **Instability of Full Wave**: Full Wave exhibits significant degradation on certain tasks (e.g., STS-B), possibly due to unstable gradient propagation in the velocity branch.
- **Only classification tasks evaluated**: Generative tasks (e.g., language generation, image generation) are not addressed; the impact of wave dynamics on decoder architectures is unknown.
- **Strong assumptions in the diffusion–wave equivalence**: The derivation neglects the feature transformation by $\mathbf{W}_V$ and the nonlinear effects of layer normalization, making the equivalence approximate.
- **Incomplete comparison with over-smoothing baselines**: Methods such as SkipInit and ReZero (simple residual scaling) are not included in the comparison.

## Related Work & Insights

- **Graph-CON / PDE-GCN**: Introduce oscillatory/PDE dynamics in sparse-graph GNNs to mitigate over-smoothing; this paper extends the idea to full-graph attention (Transformers), a complementary rather than competing contribution.
- **FeatScale (Wang et al. 2022)**: Enhances high-frequency signals via feature re-weighting (an external perturbation strategy); this paper modifies intrinsic dynamics and can be combined with FeatScale.
- **Deng et al. (Denoising Hamiltonian Network)**: Introduces Hamiltonian structure via auxiliary losses; this paper directly replaces residual dynamics without requiring additional losses.
- **GRAND (Chamberlain et al. 2021)**: Proposes the graph neural diffusion framework; this paper establishes the equivalence between attention and this framework on complete graphs and further generalizes to wave equations.
- **DIFFormer (Wu et al. 2023)**: A diffusion-based graph Transformer; this paper augments it with wave residuals, substantially improving performance collapse in deep settings.
- **Dong et al. 2021**: Theoretically proves that the rank of pure attention decays exponentially with depth; this paper offers a complementary over-smoothing explanation from the diffusion perspective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The insight of attention-as-diffusion is novel and elegant, though wave equations on GNNs have precedent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers NLP, CV, and graph tasks, but NLP experiments are small-scale and generative tasks are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Physical intuition and mathematical derivations are tightly integrated; the logical chain from PDE to discretization to architecture design is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ — Provides a new perspective for understanding Transformer over-smoothing and a lightweight, general-purpose solution with practical implications for deep Transformer design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Relational Graph Transformer](../../ICLR2026/graph_learning/relational_graph_transformer.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)
- [\[AAAI 2026\] NTSFormer: A Self-Teaching Graph Transformer for Multimodal Isolated Cold-Start Node Classification](../../AAAI2026/graph_learning/ntsformer_a_self-teaching_graph_transformer_for_multimodal_isolated_cold-start_n.md)
- [\[AAAI 2026\] GT-SNT: A Linear-Time Transformer for Large-Scale Graphs via Spiking Node Tokenization](../../AAAI2026/graph_learning/gt-snt_a_linear-time_transformer_for_large-scale_graphs_via_spiking_node_tokeniz.md)
- [\[NeurIPS 2025\] DuetGraph: Coarse-to-Fine Knowledge Graph Reasoning with Dual-Pathway Global-Local Fusion](duetgraph_coarse-to-fine_knowledge_graph_reasoning_with_dual-pathway_global-loca.md)

</div>

<!-- RELATED:END -->
