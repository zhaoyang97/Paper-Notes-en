---
title: >-
  [Paper Note] The Cylindrical Representation Hypothesis for Language Model Steering
description: >-
  [ICML 2026][LLM/NLP][Activation Steering] This paper proposes the Cylindrical Representation Hypothesis (CRH), which abandons the orthogonality of the Linear Representation Hypothesis (LRH) while preserving "conceptual l…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "Activation Steering"
  - "Linear Representation Hypothesis"
  - "Representation Geometry"
  - "Controllability"
  - "Concept Vectors"
date: 2026-05-08
content_hash: aec7f7779b7cfcc2
---

# The Cylindrical Representation Hypothesis for Language Model Steering

**Conference**: ICML 2026  
**arXiv**: [2605.01844](https://arxiv.org/abs/2605.01844)  
**Code**: https://github.com/mbzuai-nlp/CRH  
**Area**: LLM/NLP / Representation Geometry / Interpretability  
**Keywords**: Activation Steering, Linear Representation Hypothesis, Representation Geometry, Controllability, Concept Vectors

## TL;DR
This paper proposes the Cylindrical Representation Hypothesis (CRH), which abandons the orthogonality of the Linear Representation Hypothesis (LRH) while preserving "conceptual linearity." It proves that the superposition of concept vectors naturally induces a cylindrical geometry consisting of a "central axis + normal plane + sensitive sector," providing the first geometric explanation for why activation steering is unpredictable at the sample level but observable at the population level.

## Background & Motivation

**Background**: "Activation steering" in LLMs has become a mainstream tool for interpretability and alignment research: by adding a conceptual direction vector $\mathbf{v}$ to a specific residual stream layer, one can promote or inhibit outputs based on that concept during inference. Existing theories are almost entirely based on the Linear Representation Hypothesis (LRH, Park et al. 2024)—the idea that concepts correspond to linear directions and can be orthogonalized and independently manipulated via "causal inner products."

**Limitations of Prior Work**: In practice, steering is extremely unstable, with the same direction showing massive variations in effectiveness across different samples. "Controllability predictions" based on LRH (such as representation separability) correlate weakly with measured steering success rates, making them almost unreliable for engineering purposes.

**Key Challenge**: The "lossless orthogonalization" assumed by LRH is fundamentally impossible when the number of concepts exceeds the finite dimensionality—any 2D space can hold at most 2 orthogonal directions, yet LLMs must represent thousands of concepts internally. Therefore, concepts must overlap, rendering the theoretical assumption of orthogonality untenable from the start.

**Goal**: (i) Relax LRH to "linear superposition while allowing non-orthogonality" and observe the resulting geometric structure; (ii) use this new structure to explain why steering is random at the sample level but estimable at the population level; (iii) provide empirically testable predictions.

**Key Insight**: The authors maintain the "soft core" that concept vectors are linear but allow arbitrary angles between them. Starting from the simplest premise that "difference vector = linear combination of multiple concepts," they derive that local geometry inevitably presents a trio of "central axis + normal plane + phase."

**Core Idea**: Sample-concept local geometry is modeled as a cylinder: the difference vector defines the central axis, while the projections of all concepts on the normal plane cancel each other out. The phase of the steering vector within the normal plane determines the steering outcomes—and since this phase cannot be derived from the axis and $\mathbf{v}$, steering is intrinsically unpredictable.

## Method

### Overall Architecture
The logic of CRH proceeds as follows: (1) start from the core hypothesis "difference vector $\mathbf{v}_d=\mathbf{r}_a-\mathbf{r}_b=\sum_i\alpha^{(i)}\mathbf{a}^{(i)}$"; (2) decompose each concept into axial and normal components; (3) derive the "sum of normal components equals zero" equilibrium from linear superposition constraints; (4) define sensitive sectors on the normal plane, attributing steering effects to whether the phase of $\mathbf{v}$ on the normal plane falls within a sensitive sector; (5) provide three observable experimental corollaries systematically verified on Gemma-2B / LLaMA2-7B.

### Key Designs

1. **Axial-Normal Decomposition + Normal Equilibrium Theorem**:
    - **Function**: Naturally induces a central axis $\mathbf{a}_d=\mathbf{v}_d/\|\mathbf{v}_d\|$ and a set of mutually cancelling normal components from any difference vector $\mathbf{v}_d$, geometricizing "concept superposition" into a cylindrical structure.
    - **Mechanism**: For each concept direction $\mathbf{a}^{(i)}$, perform standard projection $\mathbf{v}^{(i)}=d^{(i)}\mathbf{a}_d+\mathbf{v}_{\perp}^{(i)}$. Substituting into $\mathbf{v}_d=\sum_i\mathbf{v}^{(i)}$ yields $(\sum_i d^{(i)})=\|\mathbf{v}_d\|$ and $\sum_i\mathbf{v}_{\perp}^{(i)}=\mathbf{0}$. This means the difference vector simultaneously defines a 1D "principal axis" and a "normal equilibrium state" where all non-axial contributions cancel out. Using PCA to extract a 2D normal plane $\mathcal{P}_d=\text{span}(\mathbf{a}_{\perp}^{(c)},\text{PC}_1(\{\mathbf{a}_{\perp}^{(i)}\}_{i\neq c}))$, the equilibrium holds.
    - **Design Motivation**: This decomposition is the core step in replacing the "global single direction" of LRH with a "sample-specific axis + local normal plane." It suggests that steering is not sliding along a single direction but "passing through" or "bypassing" concepts along a cylindrical surface.

2. **Sensitive Sectors and Steering Decomposition**:
    - **Function**: Divides the normal plane into "high-sensitivity sectors" (where steering is accelerated toward the target concept) and "low-sensitivity sectors" (where it is suppressed or delayed), offering a sufficiency condition based on a simple $\beta$ coefficient comparison.
    - **Mechanism**: Decompose the steering vector as $\mathbf{v}=\mathbf{v}_{\text{axis}}+\mathbf{v}_{\perp,\mathcal{P}_d}+\boldsymbol{\epsilon}$, where the normal plane component is further written as $\mathbf{v}_{\perp,\mathcal{P}_d}=\beta_c\mathbf{v}_{\perp,\mathcal{P}_d}^{(c)}+\sum_{i\neq c}\beta_i\mathbf{v}_{\perp,\mathcal{P}_d}^{(i)}$. If the target concept contribution $\beta_c>\sum_{i\neq c}\beta_i$, it falls into the high-sensitivity sector (strengthening axial drive, rapidly activating target); otherwise, it falls into the low-sensitivity sector (dominated by competing concepts, delaying or even inhibiting activation).
    - **Design Motivation**: Uses a simple "dominance of contribution" criterion to avoid extra parameters. This naturally explains why "steering vectors with similar angles" produce opposite effects on different samples—they fall into different sectors after projecting onto similarly shaped but differently oriented normal planes.

3. **Binary Predictability Theorem + Three Observable Corollaries**:
    - **Function**: Formally states that "the magnitude of the normal plane is reliably predictable by $\mathbf{v}_d$, but the sensitive sector is not," translating this geometric property into three experimentally measurable trends.
    - **Mechanism**: Theorem 4.1 (Predictable Magnitude) states $\|\mathbf{v}_{\perp,\mathcal{P}_d}\|$ is a reliable proxy for steering intensity. Lemma 4.2 + Theorem 4.3 (Unpredictable Sectors) state that when placing more than $d$ concept directions in $d$-dimensional space, the mapping from the difference vector to concept intensity is non-injective; thus, sensitive sectors cannot be backward-derived from $\mathbf{v}_d$. Three corollaries provide experimental entry points: (i) suppressing normal components $\rho\mathbf{v}_{\perp}\to0$ simultaneously delays concept activation and delays output collapse (trade-off); (ii) fitting $\text{St}_c(\mathbf{r};\mathbf{v})/\|\mathbf{v}_d\|^k$ to $\sin^m\theta\cos^{k-m}\theta$ should yield a single peak, proving the normal plane is determined by the axis; (iii) if sectors were determined by the axis, samples with similar $\mathbf{v}_d$ should show similar steering effects—experimental negation proves sector unpredictability.
    - **Design Motivation**: The authors deliberately present a dual assertion of "predictable vs. unpredictable" to distinguish CRH from a mere geometric description, transforming it into a falsifiable theory. This section is the most theoretically significant part of the paper.

### Loss & Training
This paper does not train models. All "steering vectors" are constructed from contrastive pairs (positive/negative samples) using standard methods like DiffMean, PCA, Mean-Centering, and probe-based approaches. Probing experiments use one-shot optimization (Dunefsky & Cohan 2025): the model is frozen, and a trainable vector is optimized to maximize target sentence probability and suppress original sentence probability for 30 steps with a learning rate of 0.1.

## Key Experimental Results

### Main Results

| Model / Layer | Verification | Key Findings | Interpretation |
|-----------|------|----------|------|
| Gemma-2B-IT layer 9 | Corollary 1 (trade-off) | $\rho\downarrow$: Delayed concept activation + delayed output collapse | Normal component magnitude bidirectionally regulates steering |
| Gemma-2B-IT layer 9 | Corollary 2 (Axis determines normal plane) | $\rho_k$ curve single peak + lowest p-value | Normal plane can be determined by $\mathbf{v}_d$ |
| Gemma-2B-IT layer 9 | Corollary 3 (Sector unpredictable) | $\mathbf{v}_d$ cos similarity vs. steering diff Pearson = -0.034 (p > 0.05) | Similarity in $\mathbf{v}_d$ does not imply similarity in steering behavior |

Experiments were verified on LLaMA2-7B-Chat layers 16/24 with consistent conclusions.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Full CRH | All Corollaries 1/2/3 satisfied | Cylindrical structure is valid |
| Different steering methods (DiffMean / PCA / MC / probe) | Consistently fits CRH predictions | Cylindrical structure is independent of steering method choice |
| Zeroing normal components $\rho=1$ | Most stable output but slow concept activation | Validates that the axis alone cannot rapidly activate; normal assistance is required |

### Key Findings
- Concept activation and output collapse are two sides of the same normal component: increasing $\|\mathbf{v}_{\perp}\|$ activates the target concept faster but also pushes the representation off the reasonable semantic manifold sooner. This explains why steering often behaves as an "all-or-nothing" threshold in engineering.
- The single peak in Corollary 2 confirms that "Axis $\to$ Normal Plane" is determinable, implying the "outer shell shape" of the cylinder is uniquely determined by the difference vector, while the sensitive sector phase varies entirely by sample.
- The zero-correlation result in Corollary 3—where similarity in $\mathbf{v}_d$ does not imply similar steering behavior—is the strongest counter-evidence provided by CRH. It proves that predicting steering success solely via difference vectors is destined to fail, explaining why strategies of selecting concept vectors based on similarity have consistently failed.

## Highlights & Insights
- Abandoning "orthogonality" while retaining "linearity" is a highly astute move: it preserves the engineering-friendly nature of LRH (vector addition and subtraction still work) while simultaneously resolving the mystery of why similar directions produce vastly different effects.
- Cylindrical geometry redefines steering failure from "engineering noise" to "intrinsic geometric uncertainty." This implies that any effort to eliminate steering fluctuations via "better construction methods" and finding "purer directions" has a theoretical ceiling; the community should shift toward "searching within sensitive sectors."
- Using one-shot optimization to probe the cylindrical structure is a brilliant experimental design: borrowing a tool that maps output space back to the representation space as a "local geometric probe" bypasses the difficulty of unobservable true concept directions.

## Limitations & Future Work
- CRH models concepts as a finite set of fixed directions, but LLM "concepts" may drift across layers or contexts. This paper does not discuss context-dependent concept directions.
- The sensitive sector criterion ($\beta_c$ vs $\sum_{i\neq c}\beta_i$) is a sufficient rather than necessary condition; actual boundaries are likely more complex. The paper also provides no method to actively estimate sectors.
- Verification covers only mid-sized LLMs (2B / 7B); whether the hypothesis holds for 70B+ models or if sector structures become more complex remains an open question.
- Probing experiments were limited to specific layers (9/13 and 16/24). Whether CRH holds across all layers or evolves with depth warrants further research.

## Related Work & Insights
- **vs Linear Representation Hypothesis (Park et al. 2024)**: CRH is a strict extension of LRH that removes the "orthogonal separability" assumption; LRH is the degenerate case of CRH when $d\geq n$ (high enough dimensionality).
- **vs Toy Models of Superposition (Elhage et al. 2022)**: Superposition emphasizes unavoidable interference when "features > dimensions"; CRH provides a concrete, geometrically characterizable local model (the cylinder) for this interference.
- **vs AxBench / steering benchmarks**: CRH explains the massive volatility in steering success rates in evaluations like AxBench and suggests a methodological shift toward evaluating at the population level rather than the sample level.
- **vs Multi-dimensional Concept Geometry (Engels et al. 2025)**: While they argue "not all features are 1D linear," CRH agrees but takes a different path—not questioning linearity but allowing non-orthogonality, which is more empirically verifiable.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The perspective of "replacing the orthogonality assumption with cylindrical geometry" is highly original and geometricizes sample-specific randomness as sensitive sectors—a framework the community urgently needs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Probing experiments are ingeniously designed, with quantitative verification for all three corollaries and cross-comparisons across multiple models and construction methods.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, but the notation density is high and geometric illustrations are abstract, making the initial read challenging.
- **Value**: ⭐⭐⭐⭐⭐ Directly provides a new theoretical coordinate for the entire activation steering subfield, offering directional significance for future "interpretable + controllable" engineering improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](../../ICLR2026/llm_nlp/the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](../../ACL2026/llm_nlp/costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)
- [\[ICML 2026\] Optimizing Diversity and Quality through Base-Aligned Model Collaboration](optimizing_diversity_and_quality_through_base-aligned_model_collaboration.md)

</div>

<!-- RELATED:END -->
