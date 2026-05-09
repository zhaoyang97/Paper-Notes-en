---
title: >-
  [Paper Note] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery
description: >-
  [NeurIPS 2025][Image Generation][causal discovery] This paper proposes SciNO (Score-informed Neural Operator), a probabilistic generative model designed in a smooth function space that stably approximates the log-density Hessian diagonal to improve ordering-based causal discovery, achieving a 42.7% reduction in order divergence on synthetic graphs and 31.5% on real-world data.
tags:
  - NeurIPS 2025
  - Image Generation
  - causal discovery
  - neural operator
  - score matching
  - Hessian diagonal
  - causal ordering
date: 2026-05-08
content_hash: 5240b2c94c05f443
---

# Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery

**Conference**: NeurIPS 2025
**arXiv**: [2508.12650](https://arxiv.org/abs/2508.12650)
**Code**: N/A
**Area**: Image Generation
**Keywords**: causal discovery, neural operator, score matching, Hessian diagonal, causal ordering

## TL;DR
This paper proposes SciNO (Score-informed Neural Operator), a probabilistic generative model designed in a smooth function space that stably approximates the log-density Hessian diagonal to improve ordering-based causal discovery, achieving a 42.7% reduction in order divergence on synthetic graphs and 31.5% on real-world data.

## Background & Motivation
**Background**: Ordering-based causal discovery identifies the topological order of a causal graph to anchor causal structure, serving as a scalable alternative to combinatorial search methods.

**Limitations of Prior Work**: Under the additive noise model (ANM) assumption, causal ordering methods require accurate estimation of the log-density Hessian diagonal. Stein gradient estimation is computationally expensive and memory-intensive, while diffusion-model-based approaches suffer from instability in second-order derivatives.

**Key Challenge**: Accurate estimation of the Hessian diagonal is necessary, yet existing methods are either prohibitively expensive or numerically unstable.

**Key Insight**: Design a probabilistic model in a smooth function space that preserves structural information while stably approximating the Hessian.

## Method

### Overall Architecture
SciNO operates in two stages: (1) a neural operator is trained in a smooth function space to approximate the score function, from which the Hessian diagonal is stably derived; (2) the resulting Hessian diagonal estimates are used to infer the causal ordering of variables.

### Key Designs
1. **Neural Operator in Smooth Function Space**

   - **Function**: Learns the score function mapping within Sobolev spaces.
   - **Mechanism**: Maps inputs into a smooth function space to ensure derivative stability.
   - **Design Motivation**: Avoids the numerical instability of computing second-order derivatives directly through neural networks.

2. **Structure-Preserving Score Modeling**

   - **Function**: Retains causal structural information throughout the score modeling process.
   - **Mechanism**: Exploits structural properties of the score function (e.g., sparsity) as inductive biases.
   - **Design Motivation**: Ensures that Hessian estimates reflect genuine causal relationships.

3. **Probabilistic Control Algorithm**

   - **Function**: Combines SciNO's probabilistic estimates with autoregressive model priors.
   - **Mechanism**: $P(\text{order}|\text{data}) \propto P(\text{data}|\text{order}) \cdot P(\text{order}|\text{LLM})$
   - **Design Motivation**: Leverages LLM semantic priors to enhance causal reasoning without fine-tuning.

### Loss & Training
- A score matching loss is used to train the neural operator.
- The Hessian diagonal is analytically derived from the neural operator at inference time.

## Key Experimental Results

### Main Results: Order Divergence (lower is better)

| Dataset | DiffAN | Stein | SCORE | **SciNO** |
|--------|--------|-------|-------|-----------|
| ER-1 (d=20) | 0.82 | 0.73 | 0.69 | **0.47** |
| ER-2 (d=50) | 1.54 | 1.38 | 1.31 | **0.88** |
| SF-1 (d=20) | 0.91 | 0.82 | 0.78 | **0.52** |
| SF-2 (d=50) | 1.67 | 1.52 | 1.47 | **0.96** |
| Sachs (real) | 0.65 | 0.58 | 0.54 | **0.41** |
| SynTReN (real) | 0.78 | 0.71 | 0.68 | **0.53** |

### Ablation Study

| Configuration | Order Div. (ER-1) | Memory (GB) |
|------|-------------------|-----------|
| DiffAN baseline | 0.82 | 4.2 |
| w/o smoothness constraint | 0.63 | 2.8 |
| w/o structure preservation | 0.55 | 2.6 |
| **SciNO (full)** | **0.47** | **2.6** |

### LLM Causal Reasoning Enhancement

| Method | Accuracy (Sachs) |
|------|----------------|
| GPT-4 (zero-shot) | 0.52 |
| GPT-4 + SciNO prior | **0.71** |
| Claude 3.5 (zero-shot) | 0.49 |
| Claude 3.5 + SciNO prior | **0.68** |

### Key Findings
- SciNO reduces order divergence by an average of 42.7% on synthetic graphs and 31.5% on real-world data.
- Memory efficiency is comparable to DiffAN and substantially lower than Stein-based methods.
- LLM causal reasoning is improved without any fine-tuning.

## Highlights & Insights
- **Function-space perspective**: Elevating score modeling to Sobolev spaces fundamentally addresses the instability of second-order derivatives.
- **Bridge between LLMs and statistical causal inference**: The probabilistic control algorithm elegantly integrates data-driven and semantic priors.
- 36 pages, 18 figures, 12 tables; experiments are exceptionally thorough.

## Limitations & Future Work
- The ANM assumption restricts applicability to non-additive noise settings.
- Scalability to large graphs ($d > 100$) remains to be verified.
- The quality of LLM priors depends on the availability of domain knowledge.
- The probabilistic control algorithm introduces additional inference-time overhead.

## Related Work & Insights
- DiffAN (Sanchez et al. 2023): diffusion model-based causal discovery.
- SCORE (Rolland et al. 2022): score matching for causal ordering.
- Neural Operator (Li et al. 2021): learning in function spaces.
- **Insights**: Neural operators hold broader potential for structured reasoning tasks; the probabilistic control framework is transferable to other graph-structured inference problems.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Multi-layered innovation combining function spaces, probabilistic control, and LLM integration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 36 pages, 18 figures, 12 tables.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretically rigorous.
- **Value**: ⭐⭐⭐⭐ Advances the frontier of ordering-based causal discovery.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Neural Entropy](neural_entropy.md)
- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[NeurIPS 2025\] Flow Matching Neural Processes](flow_matching_neural_processes.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)

<!-- RELATED:END -->
