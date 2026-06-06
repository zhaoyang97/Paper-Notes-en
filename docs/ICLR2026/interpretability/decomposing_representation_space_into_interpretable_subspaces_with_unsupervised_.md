---
title: >-
  [Paper Note] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning
description: >-
  [ICLR 2026][Interpretability][subspace decomposition] This paper proposes NDM (Neighbor Distance Minimization), an unsupervised method that discovers interpretable…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "subspace decomposition"
  - "representation interpretation"
  - "neighbor distance minimization"
  - "unsupervised decomposition"
  - "knowledge localization"
  - "mechanistic interpretability"
date: 2026-05-08
content_hash: 85964beb76b994a4
---

# Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning

**Conference**: ICLR 2026
**arXiv**: [2508.01916](https://arxiv.org/abs/2508.01916)  
**Code**: [GitHub](https://github.com/huangxt39/SubspacePartition)  
**Area**: Interpretable AI / Mechanistic Interpretability / Representation Learning
**Keywords**: subspace decomposition, representation interpretation, neighbor distance minimization, unsupervised decomposition, knowledge localization, mechanistic interpretability

## TL;DR
This paper proposes NDM (Neighbor Distance Minimization), an unsupervised method that discovers interpretable, non-basis-aligned subspaces in neural network representation spaces by minimizing intra-subspace neighbor distances. On GPT-2, it achieves an average Gini coefficient of 0.71 (indicating highly concentrated information); on Qwen2.5-1.5B, it identifies separated subspaces for routing parametric knowledge versus in-context knowledge.

## Background & Motivation
**Background**: Mechanistic interpretability research aims to understand the internal mechanisms of neural networks. The basic units of analysis include: component-level (attention heads/MLPs), sparse feature-level (SAE), and subspace-level (DAS). Each has limitations: information transmitted between components is difficult to interpret; SAE produces input-dependent circuits; DAS requires manually specified causal models.

**Limitations of Prior Work**:
- **SAE's single-dimensional perspective**: assumes concepts align to individual basis vectors (1D features), yet concepts such as "knowledge type" or "syntactic role" may be distributed across multi-dimensional subspaces (Multi-Dimensional Superposition Hypothesis).
- **DAS requires supervision**: requires human-designed abstract causal models to search for subspaces, making it fundamentally hypothesis verification rather than structure discovery.
- No unsupervised method exists to automatically discover "natural" partitions of representation space.

**Key Challenge**: If mutually exclusive feature groups are compressed into orthogonal subspaces (superposition within groups, orthogonality between groups), how can these subspaces be identified without knowledge of the true features?

**Core Idea**: Mutually exclusive feature groups produce "sparse" projections within their subspace (data points concentrate along a few directions) → small neighbor distances. Incorrect partitions mix features from different groups → data points spread across the entire subspace → large neighbor distances. Therefore, minimizing intra-subspace neighbor distances = finding the correct partition.

## Method

### Overall Architecture
Collect model activations $\{\mathbf{h}_n\}_{n=1}^N$ → learn an orthogonal matrix $\mathbf{R}$ to rotate the space → partition by dimension configuration $c = [d_1, \ldots, d_S]$ → minimize average neighbor distance within all subspaces → MI-guided subspace merging → output interpretable subspace partition.

### Key Designs

1. **Neighbor Distance Minimization (NDM)**

    - Function: learns an orthogonal transformation $\mathbf{R}$ that minimizes intra-subspace neighbor distances.
    - Objective: $\min_{\mathbf{R}} \frac{1}{N} \sum_{s=1}^S \sum_{n=1}^N \text{dist}(\hat{\mathbf{h}}_n^{(s)}, \hat{\mathbf{h}}_{n^*}^{(s)})$, s.t. $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$
    - Where $\hat{\mathbf{h}}_n = \mathbf{R} \mathbf{h}_n$ and $n^* = \arg\min_{m \neq n} \text{dist}(\hat{\mathbf{h}}_n^{(s)}, \hat{\mathbf{h}}_m^{(s)})$
    - Intuition: a correct partition concentrates the projections of mutually exclusive intra-group features along a small number of directions (low neighbor distance); an incorrect partition mixes features from different groups, scattering projections across the entire subspace.
    - **Information-theoretic interpretation**: neighbor distance reflects entropy; minimizing intra-subspace entropy (orthogonal transformations preserve total entropy) = minimizing total correlation across subspaces = finding the most independent partition.

2. **MI-Guided Subspace Merging**

    - Function: starts from fine-grained partitions and merges subspaces with high mutual dependence using mutual information (MI).
    - Procedure: initialize small equal-sized subspaces → train $\mathbf{R}$ → periodically compute pairwise MI between subspaces → merge if MI/dim exceeds threshold → continue training after merging → repeat until no further merges are needed.
    - Design Motivation: determining the correct number and dimensionality of subspaces is itself critical — the MI merging strategy allows the method to adaptively determine the configuration.

3. **Gini Coefficient Evaluation**

    - Function: quantifies the concentration of intervention effects across subspaces using the Gini coefficient.
    - $G = \sum |{\Delta_s}_1 - {\Delta_s}_2| / (2S \sum \Delta_s)$; $G > 0.6$ indicates information is highly concentrated in a single subspace.
    - Baselines: Identity (no rotation), Random (random rotation), PCA.
    - Design Motivation: evaluation is grounded in known circuits (IOI, Greater-than) — if the method truly identifies "variable subspaces," intervention effects should concentrate in a single subspace.

### Loss & Training
- Orthogonality is enforced via PyTorch parameterization.
- Distance metric: Euclidean distance (outperforms cosine distance).
- Scalable to models with up to 2B parameters.

## Key Experimental Results

### Quantitative Evaluation on GPT-2 Small (5 Known Circuit Tests)

| Method | Test 1 | Test 2 | Test 3 | Test 4 | Test 5 | **Avg. Gini** |
|--------|--------|--------|--------|--------|--------|---------------|
| Identity | 0.33 | 0.32 | 0.40 | 0.31 | 0.32 | 0.21 |
| Random | 0.36 | 0.36 | 0.32 | 0.33 | 0.39 | 0.21 |
| PCA | 0.43 | 0.46 | 0.50 | 0.38 | 0.35 | — |
| **NDM** | **0.71** | **0.72** | **0.75** | **0.68** | **0.69** | **0.71** |

NDM's average Gini substantially exceeds all baselines (surpassing the >0.6 threshold for highly concentrated information), while Identity/Random/PCA all remain below 0.5.

### Qualitative Analysis on Qwen2.5-1.5B

| Finding | Description |
|---------|-------------|
| **Parametric knowledge subspace** | Encodes knowledge memorized from training data |
| **In-context knowledge subspace** | Encodes knowledge inferred from the current context |
| Separation of the two | Reside in distinct subspaces, supporting research on "knowledge conflicts" |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| Toy model (known feature groups) | Perfect subspace recovery | Validates NDM in principle |
| Varying number of feature groups | Successful decomposition in all cases | Method is robust |
| Without MI merging | Fragmented, uninterpretable | Merging strategy is necessary |

### Key Findings
- NDM's information concentration (Gini 0.71) far exceeds all baselines, indicating that representation space does possess a "natural" subspace structure.
- Known circuit variables in GPT-2 (e.g., subject position in the IOI circuit) are indeed concentrated in a single subspace, validating the method's effectiveness.
- The separation between parametric and in-context knowledge subspaces is an important interpretability finding that directly supports research on "knowledge conflicts" and "hallucination."
- The method scales to 2B-parameter models, demonstrating sufficient practical applicability.

## Highlights & Insights
- **Paradigm shift from single features to subspaces**: SAE assumes concept = single-dimensional feature; NDM allows concept = multi-dimensional subspace. This better aligns with the Multi-Dimensional Superposition Hypothesis and represents a natural elevation in the granularity of analysis.
- **Unsupervised discovery vs. supervised verification**: DAS first posits a causal model and then searches for subspaces (verification); NDM directly discovers subspaces from activation data (discovery), after which causal verification can follow. The discovery → verification pipeline has greater potential for scientific insight than verification → discovery.
- **Vision of "subspace circuits"**: If subspaces serve as reliable "variables," one could analyze weights to determine which subspaces each attention head reads from and writes to, constructing input-agnostic circuits — a more general framework than the input-dependent circuits produced by SAE.

## Limitations & Future Work
- The orthogonality constraint assumes strictly orthogonal subspaces, whereas in practice subspaces may exhibit small angular deviations.
- The MI threshold for subspace merging must be set manually.
- Scalability to larger models (>10B parameters) has not been verified.
- Evaluation is limited to the Transformer architecture; CNNs, MLPs, and other architectures are not addressed.
- NDM assumes a mutually exclusive feature group structure; if features exhibit more complex dependencies (e.g., hierarchical structure), the current method may be insufficient.

## Related Work & Insights
- **vs. SAE**: SAE identifies single-dimensional sparse features, where each feature is a "direction"; NDM identifies multi-dimensional subspaces, where each subspace represents a collection of mutually exclusive features. The advancement lies in capturing multi-dimensional concepts.
- **vs. DAS (Geiger 2024)**: Both learn orthogonal transformations, but DAS is supervised (requiring a specified causal model and counterfactual data), whereas NDM is unsupervised and discovers subspaces directly from activation data.
- **vs. Engels 2024 (Multi-Dim Superposition)**: NDM's notion of "feature groups" closely aligns with their "multi-dimensional irreducible features," and can be viewed as a computational realization of that hypothesis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unsupervised subspace decomposition approach is highly original; the theoretical connection between neighbor distance and mutually exclusive feature groups is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers toy model validation, GPT-2 quantitative evaluation, and Qwen2.5 qualitative analysis, spanning verification, discovery, and scalability.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from intuition to formalization to experiments is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Provides a new analytical granularity and unsupervised tool for mechanistic interpretability; the vision of "subspace circuits" carries transformative potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](information_shapes_koopman_representation.md)
- [\[ICLR 2026\] NIMO: a Nonlinear Interpretable MOdel](nimo_a_nonlinear_interpretable_model.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)

</div>

<!-- RELATED:END -->
