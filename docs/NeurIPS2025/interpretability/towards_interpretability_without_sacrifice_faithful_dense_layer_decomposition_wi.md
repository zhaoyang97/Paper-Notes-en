---
title: >-
  [Paper Note] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders
description: >-
  [NeurIPS 2025][Interpretability][mechanistic interpretability] This paper proposes Mixture of Decoders (MxD), which decomposes the MLP layers of LLMs into tens of thousands of sparsely activated expert sub-layers (layer-…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "mechanistic interpretability"
  - "sparse approximation"
  - "mixture of experts"
  - "tensor factorization"
  - "MLP decomposition"
date: 2026-05-08
content_hash: a3e125833a1e552d
---

# Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders

**Conference**: NeurIPS 2025
**arXiv**: [2505.21364](https://arxiv.org/abs/2505.21364)  
**Code**: [GitHub](https://github.com/james-oldfield/MxD/)  
**Area**: LLM Efficiency / Interpretability
**Keywords**: mechanistic interpretability, sparse approximation, mixture of experts, tensor factorization, MLP decomposition

## TL;DR
This paper proposes Mixture of Decoders (MxD), which decomposes the MLP layers of LLMs into tens of thousands of sparsely activated expert sub-layers (layer-level sparsity). Each expert implements a full-rank linear transformation via Hadamard product tensor factorization. MxD significantly outperforms Transcoders on the sparsity–accuracy trade-off while maintaining interpretability.

## Background & Motivation
**Background**: MLP layer representations in LLMs are dense — individual neurons encode multiple concepts, making it difficult to isolate specific features. SAEs and Transcoders learn sparse overcomplete bases to approximate MLP layer outputs, rendering features more interpretable.

**Limitations of Prior Work**: Existing methods adopt **neuron-level sparsity** (constraining the number of non-zero elements in the hidden layer), which leads to a severe **sparsity–accuracy trade-off** — higher sparsity results in greater reconstruction error of the original MLP mapping. Unfaithful reconstruction risks missing critical behaviors and precludes direct layer substitution during inference.

**Key Challenge**: Interpretability demands high sparsity, yet neuron-level methods with high sparsity can only exploit a low-dimensional subspace of the output space ($K$ non-zero hidden units → $K$-dimensional subspace), sacrificing the expressive capacity of the original layer.

**Goal**: Faithfully reconstruct the functionality of the original MLP layer while maintaining high sparsity.

**Key Insight**: Shift from neuron-level sparsity to **layer-level sparsity** — selecting a small number of full-rank linear transformations (expert sub-layers) at each step, where each expert is far more expressive than a single neuron.

**Core Idea**: Construct a large collection of parameter-efficient full-rank expert sub-layers via Hadamard product tensor factorization, and faithfully reconstruct the original MLP by sparsely activating $K$ of them.

## Method

### Overall Architecture
MxD approximates the original MLP output as a sparse weighted combination of $N$ linear transformations:
$$\text{MxD}(\mathbf{x}) = \sum_{n=1}^N a_n(\mathbf{W}_n^\top\mathbf{z}),$$
where $\mathbf{z} = \phi(\mathbf{E}^\top\mathbf{x})$ is the dense hidden representation, $\mathbf{a} = \mathcal{S}(\mathbf{G}^\top\mathbf{x})$ are the sparse expert coefficients (top-$K$), and $\mathbf{W}_n$ is the decoder weight of the $n$-th expert.

### Key Designs

1. **Hadamard Product Tensor Factorization**:

    - **Function**: Parameter-efficiently stores $N$ full-rank expert weights.
    - **Mechanism**: $\boldsymbol{\mathcal{W}}(n,h,:) = \mathbf{c}_n * \mathbf{d}_h$, where $\mathbf{C} \in \mathbb{R}^{N \times O}$ are expert-specific parameters and $\mathbf{D} \in \mathbb{R}^{H \times O}$ is a shared transformation. Parameter count reduces from $NHO$ to $O(N+H)$.
    - Equivalent forward pass: $\text{MxD}(\mathbf{x}) = (\mathbf{C}^\top\mathbf{a}) * (\mathbf{D}^\top\mathbf{z})$.

2. **Full-Rank Guarantee (Lemma 1)**:

    - **Function**: Proves that each expert weight matrix is full rank.
    - **Core Result**: $\mathbf{W}_n = \mathbf{D}\,\text{diag}(\mathbf{c}_n)$; as long as $\mathbf{c}_n$ contains no zero entries, $\text{rank}(\mathbf{W}_n) = \text{rank}(\mathbf{D})$.
    - **Design Motivation**: At sparsity $K$, Transcoder output is confined to a $K$-dimensional subspace, whereas MxD sums $K$ full-rank transformations, yielding substantially greater expressive power.

3. **GLU Extension**:

    - **Function**: Generalizes to the Gated Linear Unit architecture used in modern LLMs.
    - **Mechanism**: Directly substitutes the GLU hidden representation $\mathbf{z}_{\text{GLU}} = \psi(\mathbf{E}_{\text{GLU}}^\top\mathbf{x}) * (\mathbf{E}^\top\mathbf{x})$ into MxD.

### Loss & Training
- MSE distillation loss (MxD output vs. original MLP output).
- Top-$K$ routing; trained on 480M tokens from OpenWebText.
- $\mathbf{D}$ is initialized to zero for gradual learning.

## Key Experimental Results

### Main Results
Sparsity–accuracy frontier across 4 LLMs (parameter-matched):

| Method | Sparsity Level | CE Loss Increase | Interpretability |
|--------|---------------|-----------------|-----------------|
| Transcoder | Neuron-level | Large | Good |
| Skip Transcoder | Neuron-level + skip | Moderate | Good |
| **MxD** | **Layer-level** | **Significantly smaller** | **Comparable** |

MxD Pareto-dominates Transcoders across all sparsity levels.

### Ablation Study

| Configuration | Result | Note |
|---------------|--------|------|
| GELU vs. ReLU | GELU significantly better | Matching the original activation function matters |
| Full-rank vs. low-rank MoE | Full-rank better | Validates the practical value of Lemma 1 |
| Varying $N$ | More experts → better | Tensor factorization makes tens of thousands of experts feasible |

### Key Findings
- **MxD comprehensively outperforms Transcoder on the sparsity–accuracy frontier**: lower CE loss increase under identical parameter counts and sparsity levels.
- **No sacrifice in interpretability**: performance is on par with Transcoder across 34 sparse probing and steering tasks.
- **Full rank is critical**: even at $K$=32, high-fidelity reconstruction is achieved because each active expert contributes a full-rank transformation.

## Highlights & Insights
- **Paradigm shift to layer-level sparsity**: the atomic unit of interpretation is elevated from a neuron to a complete linear transformation, more closely resembling the concept of a "functional module."
- **Hadamard factorization achieves both efficiency and full rank**: the formulation $(\mathbf{C}^\top\mathbf{a}) * (\mathbf{D}^\top\mathbf{z})$ is elegantly simple while theoretically guaranteeing full rank.
- **Unified decomposition of MLP and GLU**: no reliance on specific sparsity assumptions.

## Limitations & Future Work
- **Experiments scale up to 3B only**: effectiveness on models with 10B+ parameters remains unverified.
- **Layer-wise independent training**: joint multi-layer or end-to-end training has not been explored.
- **Limited interpretability evaluation**: human evaluation and in-depth mechanistic analysis are absent.
- **Future directions**: joint multi-layer training, validation on larger models, and applications to model editing and safety control.

## Related Work & Insights
- **vs. SAE**: SAEs are applied post-hoc (with additional inference overhead), whereas MxD directly replaces layers.
- **vs. Transcoder**: Neuron-level sparsity vs. layer-level sparsity; the latter offers substantially greater expressive power.
- **vs. conventional MoE**: Traditional MoE uses few experts (on the order of tens), while MxD scales to tens of thousands via tensor factorization.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The layer-level sparsity perspective and the full-rank guarantee via Hadamard factorization are both entirely novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four models × multiple sparsity levels, with Pareto analysis and interpretability evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The theory–method–experiment chain is clear and coherent; Table 1 provides an effective summary.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a more faithful and practical layer decomposition tool for LLM interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions](../../ICML2026/interpretability/towards_steering_without_sacrifice_principled_training_of_steering_vectors_for_p.md)
- [\[NeurIPS 2025\] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)
- [\[NeurIPS 2025\] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions](partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](../../ICLR2026/interpretability/exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)

</div>

<!-- RELATED:END -->
