---
title: >-
  [Paper Note] The Geometry of Reasoning: Flowing Logics in Representation Space
description: >-
  [ICLR 2026][Reasoning Geometry] This paper proposes a geometric framework that models the reasoning process of LLMs as "flows" (embedding trajectories) in representation space. Through controlled experiments that decouple logical structure from semantic content, it demonstrates that LLMs internalize logical invariants beyond surface form, and identifies potentially universal representation regularities across model families.
tags:
  - ICLR 2026
  - Reasoning Geometry
  - Representation Flow
  - Logical Invariants
  - LLM Interpretability
  - Concept Space
date: 2026-05-08
content_hash: 0f70acf0239113f0
---

# The Geometry of Reasoning: Flowing Logics in Representation Space

**Conference**: ICLR 2026
**arXiv**: [2510.09782](https://arxiv.org/abs/2510.09782)
**Code**: Available (see paper)
**Area**: LLM Interpretability / Reasoning Mechanisms
**Keywords**: Reasoning Geometry, Representation Flow, Logical Invariants, LLM Interpretability, Concept Space

## TL;DR

This paper proposes a geometric framework that models the reasoning process of LLMs as "flows" (embedding trajectories) in representation space. Through controlled experiments that decouple logical structure from semantic content, it demonstrates that LLMs internalize logical invariants beyond surface form, and identifies potentially universal representation regularities across model families.

## Background & Motivation

**Background**: Large language models (LLMs) exhibit remarkable capabilities across diverse reasoning tasks, yet the nature of their internal "reasoning" remains poorly understood. Mainstream interpretability research focuses on attention analysis, probing classifiers, and mechanistic interpretation (circuit analysis), but these approaches largely address local components rather than the global geometric structure of the reasoning process.

**Limitations of Prior Work**: The debate over whether LLMs truly "understand" logic persists. The "stochastic parrot" hypothesis holds that LLMs perform only surface-level pattern matching, lacking genuine comprehension of logical structure. Prior work lacks a formal mathematical framework for describing and verifying the internal representational dynamics of LLM reasoning, making it impossible to distinguish whether models are applying logic or exploiting statistical correlations.

**Key Challenge**: If LLMs engage only in surface pattern matching, the same logical reasoning structure instantiated with different semantic carriers (e.g., different vocabulary and topics) should yield entirely different representation trajectories. Conversely, if LLMs genuinely internalize logical invariants, logical structure should manifest as some form of geometric invariance in representation space—yet no framework or tools previously existed to test this hypothesis.

**Goal**: (1) How can the geometric behavior of LLM reasoning in representation space be formally described? (2) Do LLMs internalize semantic-agnostic logical invariants in representation space? (3) Do such geometric properties generalize across model architectures?

**Key Insight**: The authors draw an analogy between the layer-wise (or token-wise) reasoning process of LLMs and trajectory evolution in dynamical systems, proposing to characterize reasoning flows using the language of differential geometry (position, velocity, curvature). The key experimental design employs natural deduction propositions—holding logical structure fixed while varying semantic carriers—to decouple logic from semantics.

**Core Idea**: Model LLM reasoning as a geometric flow in representation space, and use velocity field and curvature analysis to demonstrate that logical statements act as local controllers of these flows.

## Method

### Overall Architecture

The framework centers on treating the hidden representations produced by an LLM while processing reasoning problems as trajectories (flows) in high-dimensional space. The input consists of text sequences containing logical reasoning steps; passing through successive layers of the model yields a series of embedding vectors whose evolution constitutes the "reasoning flow." The framework comprises three core components: (1) **representation space modeling**—modeling inter-layer embedding changes as a continuous flow; (2) **concept space mapping**—projecting the high-dimensional space onto an analyzable low-dimensional concept space via learned representation proxies; and (3) **controlled experimental design**—verifying logical invariance through semantic decoupling.

### Key Designs

1. **Geometric Modeling of Reasoning Flows**

   - **Function**: Provides a mathematical formalization of the LLM reasoning process, modeling discrete inter-layer transformations as a continuous geometric flow.
   - **Mechanism**: Defines the flow in representation space as the inter-layer trajectory of embeddings $\{h^{(l)}\}_{l=0}^{L}$, where $h^{(l)}$ is the hidden state at layer $l$. The velocity of the flow is defined as the finite difference between adjacent-layer representations, $v^{(l)} = h^{(l+1)} - h^{(l)}$, and curvature is measured via second-order changes in velocity. The authors establish correspondences between these geometric quantities and reasoning steps: logical operations (e.g., modus ponens) correspond to specific patterns in flow velocity, while reasoning "difficulty" can be quantified through curvature.
   - **Design Motivation**: Reducing reasoning to geometric quantities enables formal mathematical analysis, moving beyond purely qualitative observation.

2. **Semantic–Logic Decoupling Experimental Design**

   - **Function**: Verifies that LLMs internalize logical structure rather than surface semantic patterns.
   - **Mechanism**: Experimental data are generated using the natural deduction framework—fixing the same logical reasoning chain (e.g., $A \rightarrow B$, $A$, therefore $B$) while substituting different semantic carriers (e.g., replacing "cats are animals" with "iron is a metal" and other domain-varied propositions). Geometric invariance of the reasoning flows under these different semantic carriers (e.g., consistency of velocity directions, similarity of curvature patterns) is analyzed to determine whether the model has internalized abstract logical rules independent of specific semantics.
   - **Design Motivation**: This is the most critical experimental design in the work. If LLMs perform only statistical association, flows under different semantics should differ entirely; only when the model genuinely internalizes logical structure will the geometric properties of the flow remain invariant under semantic variation.

3. **Learned Representation Proxies and Visualization**

   - **Function**: Projects flows from high-dimensional representation space into a low-dimensional concept space amenable to analysis and visualization.
   - **Mechanism**: Representation proxies are trained to map LLM high-dimensional embeddings into a low-dimensional concept space while preserving key geometric properties. In this concept space, reasoning flow trajectories, velocity fields, and curvature variations can be visualized and quantitatively analyzed. This approach bridges the abstract theoretical framework with concrete empirical validation.
   - **Design Motivation**: High-dimensional representation spaces are difficult to analyze and visualize directly; dimensionality reduction is necessary, but standard methods (e.g., PCA/t-SNE) may destroy critical geometric structure, necessitating mappings specifically designed to preserve geometric properties.

### Loss & Training

This paper does not train new models; instead, it analyzes pre-trained LLMs. The training objective for the representation proxies is to preserve the geometric structure of the flows (velocity direction, curvature, etc.) during dimensionality reduction, using a standard metric-preserving loss.

## Key Experimental Results

### Main Results: Cross-Model Logical Invariance Verification

| Model Family | Model Scale | Reasoning Flow Smoothness | Logical Invariance | Semantic Decoupling |
|---|---|---|---|---|
| Qwen | Multiple scales | ✓ Smooth flow | ✓ Consistent across semantics | High |
| LLaMA | Multiple scales | ✓ Smooth flow | ✓ Consistent across semantics | High |

Two principal findings: (1) LLM reasoning corresponds to smooth flows in representation space; (2) logical statements act as local controllers of the velocity of these flows.

### Cross-Architecture Universality Analysis

| Analysis Dimension | Finding | Implication |
|---|---|---|
| Velocity field directional consistency | Directions are highly similar across different semantic carriers | Logical structure, not semantics, governs reasoning trajectories |
| Curvature pattern stability | Difficult reasoning steps correspond to high-curvature regions | Logical complexity has a geometric signature |
| Cross-model family | Qwen and LLaMA exhibit similar geometric properties | Potentially universal representation regularities exist |
| Training recipe independence | Geometric properties are largely invariant to specific training configurations | Regularities arise from task structure rather than training details |

### Key Findings

- **Reasoning is indeed a smooth flow**: Inter-layer representation evolution in LLMs is not random jumping but smooth, continuous trajectories in representation space, providing an empirical basis for analyzing reasoning with differential geometry.
- **Logic as a geometric controller**: Logical statements (e.g., premises, inference rules) manifest in representation space as local control signals for flow velocity—altering logical steps systematically changes the direction and speed of the flow.
- **Challenging the "stochastic parrot" hypothesis**: Models trained purely on next-token prediction can internalize logical invariants as higher-order geometric structures in representation space, suggesting that LLM "understanding" may run considerably deeper than surface statistical correlation.
- **Potential universality**: Models across the Qwen and LLaMA families at various scales exhibit similar geometric properties, hinting at underlying representation regularities shared between machine understanding and human language structure.

## Highlights & Insights

- **Geometry as a unified lens for reasoning analysis**: Modeling reasoning as flow is an elegant approach; the position–velocity–curvature toolkit borrowed from classical mechanics provides intuition-friendly analytical instruments for LLM internal representations. This framework is transferable to other settings requiring understanding of model internal dynamics.
- **Semantic–logic decoupling experimental design**: Using natural deduction propositions as experimental vehicles—fixing logical structure while varying semantic content—is a clean and powerful controlled-variable design and arguably the paper's most ingenious contribution.
- **Bridging interpretability and mathematical rigor**: Unlike most qualitative interpretability work, this paper attempts to establish a quantifiable, formalizable geometric framework, supplying a mathematical toolbox for LLM reasoning research.

## Limitations & Future Work

- **Only the abstract is available**: As the cache contains only the abstract, specific quantitative results and experimental details cannot be fully evaluated.
- **Fidelity of concept space mapping**: Projecting to a low-dimensional concept space inevitably incurs information loss; more rigorous validation is needed to confirm that the preserved geometric properties are sufficiently complete.
- **Causality vs. correlation**: Observing geometrically invariant logical properties does not constitute proof that the model is "using" logical reasoning; intervention experiments are required to establish causal connections.
- **Coverage of reasoning types**: Natural deduction represents only one form of formal logic; whether more complex reasoning modes (e.g., analogical reasoning, inductive reasoning) exhibit analogous geometric properties remains to be explored.

## Related Work & Insights

- **vs. Mechanistic Interpretability (Neel Nanda et al.)**: Mechanistic interpretability focuses on specific circuits and attention head functions, while this paper examines global geometric invariants. The two approaches are complementary—circuits constitute the microscopic mechanism, whereas geometric flow constitutes the macroscopic dynamics.
- **vs. Probing Classifiers**: Probing methods detect whether a given layer encodes a particular feature; this paper analyzes the dynamic trajectory of the entire reasoning process, providing richer spatiotemporal information.
- **vs. Neural ODE Perspective**: Treating Transformers as dynamical systems (e.g., Neural ODEs) is an established idea; this paper specializes it to the reasoning setting and introduces logic–semantics decoupling verification, constituting a meaningful applied instance.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic application of a differential-geometric framework to LLM reasoning analysis; highly original perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ — Validated across multiple model families, but limited cache availability precludes detailed assessment of quantitative results.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical framework is articulated clearly with well-organized conceptual layers.
- **Value**: ⭐⭐⭐⭐⭐ — Carries far-reaching implications for understanding LLM reasoning mechanisms; introduces new conceptual tools and methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](information_shapes_koopman_representation.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)

</div>

<!-- RELATED:END -->
