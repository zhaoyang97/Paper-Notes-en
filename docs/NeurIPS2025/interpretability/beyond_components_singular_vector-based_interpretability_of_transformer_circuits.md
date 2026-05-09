---
title: >-
  [Paper Note] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits
description: >-
  [NeurIPS 2025][SVD interpretability] This paper proposes a direction-level interpretability framework based on singular vectors. By applying SVD to augmented matrices of Transformer attention heads and MLPs, combined with learnable diagonal mask optimization (KL+L₁), it reveals orthogonal low-rank subfunctions within individual components. On the IOI task, retaining only ~9% of directions suffices to reproduce model behavior with KLD=0.21; moreover, different singular directions within Head 9.6 separately encode distinct computational primitives such as semantic entity separation, entity salience, and sequence initialization.
tags:
  - NeurIPS 2025
  - SVD interpretability
  - transformer circuits
  - singular vectors
  - mechanistic interpretability
  - directional masking
  - low-rank subfunctions
date: 2026-05-08
content_hash: efd92117bb27567e
---

# Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits

**Conference**: NeurIPS 2025
**arXiv**: [2511.20273](https://arxiv.org/abs/2511.20273)
**Code**: [GitHub](https://github.com/Exploration-Lab/Beyond-Components)
**Area**: LLM Interpretability / Transformer Circuit Analysis
**Keywords**: SVD interpretability, transformer circuits, singular vectors, mechanistic interpretability, directional masking

## TL;DR
This paper proposes a direction-level interpretability framework based on SVD singular vectors. By applying unified SVD decomposition to augmented matrices of attention heads and MLPs, combined with a learnable diagonal mask optimized via KL+L₁, the framework reveals orthogonal low-rank subfunctions superposed within a single component — on the IOI task, retaining only ~9% of directions suffices to reproduce model behavior with KLD=0.21.

## Background & Motivation
**State of the Field**: Mechanistic interpretability typically treats attention heads and MLP layers as atomic units, probing or ablating entire components via causal tracing, activation patching, and attribution analysis.

**Limitations of Prior Work**: This component-level perspective implicitly assumes a one-to-one correspondence between functions and component boundaries. In practice, a single head or MLP may multiplex multiple subfunctions through superposition, obscuring fine-grained internal computation.

**Root Cause**: Merullo et al. introduced a low-rank perspective for analyzing inter-component communication, showing that attention heads communicate via singular directions of the value matrix in the residual stream, but did not investigate intra-component functional decomposition.

**Paper Goals**: To use SVD singular vectors as orthogonal "computational directions," unifying the treatment of attention QK/OV transformations and MLP in/out projections, revealing independently superposed subfunctions within components, and enabling direction-level attribution via learnable masks.

**Starting Point**: Bias terms are folded into weight matrices to form augmented matrices, which are then decomposed via SVD; singular directions serve as orthogonal computational directions.

**Core Idea**: Transformer computation is distributed and compositional — overlapping subfunctions are embedded in shared subspaces and can be independently manipulated via SVD directions.

## Method

### Overall Architecture
For each component, an augmented matrix is constructed by folding biases into weights → SVD decomposes it into orthogonal directions → a learnable diagonal mask identifies task-critical directions → direction-level attribution and intervention are performed.

### Key Designs

1. **Unified Augmented Matrix**: Bias folding makes QK/OV/MLP comparable under the same SVD framework. For QK interaction: $[1,\mathbf{x}_i]\mathbf{W}_{aug}^{(QK)}[1,\mathbf{x}_j]^\top = \mathbf{q}_i\cdot\mathbf{k}_j^\top$
2. **SVD Directional Decomposition**: $\mathbf{W}_{aug} = \sum_k \sigma_k \mathbf{u}_k \mathbf{v}_k^\top$, where each singular direction encodes an independent subfunction.
3. **Learnable Diagonal Mask**: $\Lambda = \text{diag}(\lambda_1,...,\lambda_R)$, optimized via $\min KL(p_{orig}\|p_{masked}) + \alpha\|\Lambda\|_1$ to automatically identify the minimal set of necessary directions.
4. **Logit Receptors**: Naturally emerging directions in logit space that allow scalar-level control over model predictions.

## Key Experimental Results

### Direction-Level Sparsity (GPT-2 Small)

| Task | Directions Retained | KL Divergence | Note |
|------|---------------------|---------------|------|
| IOI | ~9% | 0.21 | 91% of directions can be discarded |
| GP | Sparse | Low | Gender directions are independently controllable |
| GT | Sparse | Low | Numerical comparison directions align |

### Key Findings

| Finding | Description |
|---------|-------------|
| Intra-component multifunctionality | Different directions within Head 9.6 separately encode entity separation, salience, and initialization |
| Circuit heads show stronger directional activation | Mask weights for IOI circuit heads are significantly higher than for non-circuit heads |
| Logit Receptors are controllable | Scalar intervention suffices to switch gender predictions |
| Applicable to MLPs | MLP layers also exhibit direction-level functional decomposition |

## Highlights & Insights
- **Paradigm Shift**: From "component = function" to "direction = function"; future interpretability work should attribute at the level of singular directions.
- **Unified Augmented Matrix Framework**: Bias folding places attention and MLP under the same comparable framework.
- **Logit Receptors**: Provide a new tool for model editing and behavioral control.

## Limitations & Future Work
- Validation is limited to GPT-2 Small; scalability to larger models remains unknown.
- The linear SVD assumption does not account for the effects of nonlinear activations.
- The semantic function of some directions is difficult to articulate in natural language.
- Evaluation covers only three tasks: IOI, GP, and GT.

## Related Work & Insights
- **vs. ACDC Circuit Discovery**: Standard methods perform component-level ablation; this paper operates at the direction level — achieving finer granularity.
- **vs. SAE**: Sparse autoencoders require training an auxiliary encoder; SVD decomposition is more lightweight by comparison.
- **Insight**: Direction-level controllability opens a new path toward precise model editing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Direction-level SVD interpretability is a genuinely new perspective.
- Experimental Thoroughness: ⭐⭐⭐ Three tasks are well-analyzed but limited to GPT-2 Small.
- Writing Quality: ⭐⭐⭐⭐⭐ The unified linear framework is elegantly designed.
- Value: ⭐⭐⭐⭐ Offers important inspiration for mechanistic interpretability research.

**Area**: LLM Interpretability / Transformer Circuit Analysis
**Keywords**: SVD interpretability, transformer circuits, singular vectors, mechanistic interpretability, directional masking, low-rank subfunctions

## TL;DR
This paper proposes a direction-level interpretability framework based on singular vectors. By applying SVD to augmented matrices of Transformer attention heads and MLPs, combined with learnable diagonal mask optimization (KL+L₁), it reveals orthogonal low-rank subfunctions within individual components. On the IOI task, retaining only ~9% of directions suffices to reproduce model behavior with KLD=0.21; moreover, different singular directions within Head 9.6 separately encode distinct computational primitives such as semantic entity separation, entity salience, and sequence initialization.

## Background & Motivation
1. **State of the Field**: Mechanistic interpretability typically treats attention heads and MLP layers as indivisible atomic units, probing or ablating entire components via causal tracing, activation patching, and attribution analysis.
2. **Core Problem**: This component-level perspective implicitly assumes a one-to-one correspondence between functions and component boundaries. In practice, a single head or MLP may multiplex multiple subfunctions through superposition, and this component-centrism obscures fine-grained internal computation.
3. **Limitations of Prior Work**: Merullo et al. proposed a low-rank perspective for analyzing inter-component communication, demonstrating that attention heads communicate via singular directions of the value matrix in the residual stream, but did not investigate intra-component functional decomposition.
4. **Starting Point**: SVD singular vectors are treated as orthogonal "computational directions" to unify the handling of attention QK/OV transformations and MLP in/out projections, revealing independently superposed subfunctions within components and enabling direction-level attribution via learnable masks.

## Method

### Key Designs
1. **SVD Decomposition**: SVD is applied to the augmented weight matrices of attention heads and MLPs.
2. **Learnable Diagonal Mask**: Optimized via KL+L₁ regularization to identify task-critical orthogonal directions.

## Key Experimental Results
- IOI task: sparsity **91.32%** with only KLD=0.21.
- Generality validated on GT/GP tasks.
- "Name mover" heads encode overlapping subfunctions across multiple singular vectors.

## Highlights & Insights
- Fine-grained direction-level interpretation surpasses the assumptions of component-centrism.

## Limitations & Future Work
- Validation is limited to GPT-2 Small; large-scale model evaluation is absent.

## Rating
- Novelty: ⭐⭐⭐⭐ Direction-level SVD interpretability is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐ Sufficient validation on a small model but limited in scale.
- Writing Quality: ⭐⭐⭐⭐⭐ The unified linear framework is elegantly designed.
- Value: ⭐⭐⭐⭐ Offers important inspiration for mechanistic interpretability research.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)
- [\[CVPR 2026\] From Weights to Concepts: Data-Free Interpretability of CLIP via Singular Vector Decomposition](../../CVPR2026/interpretability/from_weights_to_concepts_data-free_interpretability_of_clip_via_singular_vector_.md)
- [\[NeurIPS 2025\] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)

<!-- RELATED:END -->
