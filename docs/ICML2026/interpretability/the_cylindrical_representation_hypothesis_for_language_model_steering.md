---
title: >-
  [Paper Note] The Cylindrical Representation Hypothesis for Language Model Steering
description: >-
  [ICML 2026][Interpretability][Activation Steering] This paper proposes the Cylindrical Representation Hypothesis (CRH), which relaxes the orthogonality assumption of the LRH while retaining "concept linearity." It demons…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Activation Steering"
  - "Linear Representation Hypothesis"
  - "Representation Geometry"
  - "Controllability"
  - "Concept Vector"
date: 2026-05-08
content_hash: 066480f2be5f5c71
---

# The Cylindrical Representation Hypothesis for Language Model Steering

**Conference**: ICML 2026  
**arXiv**: [2605.01844](https://arxiv.org/abs/2605.01844)  
**Code**: https://github.com/mbzuai-nlp/CRH  
**Area**: LLM/NLP / Representation Geometry / Interpretability  
**Keywords**: Activation Steering, Linear Representation Hypothesis, Representation Geometry, Controllability, Concept Vector

## TL;DR
This paper proposes the Cylindrical Representation Hypothesis (CRH), which relaxes the orthogonality assumption of the LRH while retaining "concept linearity." It demonstrates that the superposition of concept vectors naturally induces a cylindrical geometry of "axis + normal plane + sensitive sector," thereby providing the first geometric explanation for why activation steering is unpredictable at the sample level but observable at the population level.

## Background & Motivation

**Background**: "Activation steering" in LLMs has become a mainstream tool for interpretability and alignment research: by adding a concept direction vector $\mathbf{v}$ to the residual stream at a certain layer, one can promote or suppress outputs according to the concept during inference. Existing theories are almost all based on the Linear Representation Hypothesis (LRH, Park et al. 2024)—concepts correspond to linear directions, and can be orthogonalized and independently manipulated via "causal inner product."

**Limitations of Prior Work**: In practice, steering is highly unstable; the same direction can have vastly different effects across samples. The "controllability prediction" based on LRH (e.g., representation separability) correlates weakly with actual steering success rates and is nearly unreliable in engineering.

**Key Challenge**: The "lossless orthogonalization" assumed by LRH is fundamentally impossible when the number of concepts exceeds the dimension in finite-dimensional space—any 2D space can have at most 2 orthogonal directions, but LLMs must represent thousands of concepts. Thus, concepts inevitably overlap, and the orthogonalization assumption is untenable at its core.

**Goal**: (i) Relax LRH to "linear superposition but allow non-orthogonality" and examine the resulting geometric structure; (ii) Use this new structure to explain why steering is random at the sample level but estimable at the population level; (iii) Provide empirically testable predictions.

**Key Insight**: The authors retain the "concept vectors are linear" soft core, but allow arbitrary angles between them; starting from the basic "difference vector = linear combination of multiple concepts," they derive that the local geometry inevitably presents a "central axis + normal plane + phase" triad.

**Core Idea**: Model the local geometry of sample-concept as a cylinder: the difference vector defines the central axis, all concept projections in the normal plane cancel each other, and the phase of the steering vector in the normal plane is the key to steering success or failure—yet the phase cannot be inferred from the axis and $\mathbf{v}$, so steering is intrinsically unpredictable.

## Method

### Overall Architecture
The overall logic of CRH: (1) Start from the core assumption "difference vector $\mathbf{v}_d=\mathbf{r}_a-\mathbf{r}_b=\sum_i\alpha^{(i)}\mathbf{a}^{(i)}$"; (2) Decompose each concept into axial and normal components; (3) The linear superposition constraint leads to the key balance that "the sum of all concepts' normal components is zero"; (4) Define sensitive sectors in the normal plane, attributing steering effects to "whether $\mathbf{v}$'s phase in the normal plane falls into the sensitive sector"; (5) Provide three observable experimental predictions and systematically validate them on Gemma-2B / LLaMA2-7B.

### Key Designs

1. **Axis-Normal Decomposition & Normal Balance Theorem**:

    - **Function**: Any difference vector $\mathbf{v}_d$ naturally induces a central axis $\mathbf{a}_d=\mathbf{v}_d/\|\mathbf{v}_d\|$ and a set of mutually cancelling normal components, geometrizing "concept superposition" into a cylindrical structure.
    - **Mechanism**: For each concept direction $\mathbf{a}^{(i)}$, perform standard projection $\mathbf{v}^{(i)}=d^{(i)}\mathbf{a}_d+\mathbf{v}_{\perp}^{(i)}$, substitute into $\mathbf{v}_d=\sum_i\mathbf{v}^{(i)}$, yielding $(\sum_i d^{(i)})=\|\mathbf{v}_d\|$ and $\sum_i\mathbf{v}_{\perp}^{(i)}=\mathbf{0}$. That is, the difference vector defines both a one-dimensional "main axis" and a "normal balance state" where all non-axial contributions cancel. Then, use PCA to extract the two-dimensional normal plane $\mathcal{P}_d=\text{span}(\mathbf{a}_{\perp}^{(c)},\text{PC}_1(\{\mathbf{a}_{\perp}^{(i)}\}_{i\neq c}))$, where the balance still holds.
    - **Design Motivation**: This decomposition replaces the "global single direction" of LRH with a "sample-specific axis + local normal plane"; it shows that steering does not slide along a single direction, but "passes through" or "circumvents" concepts along the cylinder surface.

2. **Sensitive Sector and Steering Decomposition**:

    - **Function**: Divide the normal plane into "high-sensitivity sectors" (steering is accelerated toward the target concept) and "low-sensitivity sectors" (suppressed or delayed), using a simple β coefficient comparison as a sufficient condition.
    - **Mechanism**: Decompose the steering vector as $\mathbf{v}=\mathbf{v}_{\text{axis}}+\mathbf{v}_{\perp,\mathcal{P}_d}+\boldsymbol{\epsilon}$, where the normal plane component is further written as $\mathbf{v}_{\perp,\mathcal{P}_d}=\beta_c\mathbf{v}_{\perp,\mathcal{P}_d}^{(c)}+\sum_{i\neq c}\beta_i\mathbf{v}_{\perp,\mathcal{P}_d}^{(i)}$. When the target concept's contribution $\beta_c>\sum_{i\neq c}\beta_i$, it falls into the high-sensitivity sector (enhanced axial drive, rapid activation of the target concept); otherwise, it falls into the low-sensitivity sector (dominated by competing concepts, activation is delayed or suppressed).
    - **Design Motivation**: Using the simplest "whose contribution is larger" as the criterion avoids introducing extra parameters; this naturally explains why "steering vectors with similar angles" can have completely opposite effects on different samples—after projection onto the same-shaped but differently positioned normal planes, they fall into different sectors.

3. **Predictability Dichotomy Theorem & Three Observable Predictions**:

    - **Function**: Formally states that "the magnitude of the normal plane can be reliably predicted from $\mathbf{v}_d$, but the sensitive sector cannot," and translates this geometric property into three experimentally testable trends.
    - **Mechanism**: Theorem 4.1 (magnitude predictability) states that $\|\mathbf{v}_{\perp,\mathcal{P}_d}\|$ is a reliable proxy for steering strength; Lemma 4.2 + Theorem 4.3 (sector unpredictability) state that in $d$-dimensional space with more than $d$ concept directions, the mapping from difference vector to concept strength is non-injective, so the sensitive sector cannot be inferred from $\mathbf{v}_d$. The three predictions provide experimental entry points: (i) Suppressing the normal component $\rho\mathbf{v}_{\perp}\to0$ simultaneously delays concept activation and output collapse (trade-off); (ii) Fitting $\text{St}_c(\mathbf{r};\mathbf{v})/\|\mathbf{v}_d\|^k$ to $\sin^m\theta\cos^{k-m}\theta$ should yield a unimodal curve, indicating the normal plane is determined by the axis; (iii) If the sector is determined by the axis, then samples with similar $\mathbf{v}_d$ should have similar steering effects—if experiments show otherwise, the sector is not determined by the axis.
    - **Design Motivation**: The authors deliberately make a "predictable vs. unpredictable" dual assertion, distinguishing CRH from a mere "concept geometry description" and turning it into a falsifiable geometric theory; this section is the most theoretically substantial part of the paper.

### Loss & Training
No model training is performed; all "steering vectors" are constructed from contrastive pairs (positive/negative samples) using standard methods such as DiffMean, PCA, Mean-Centering, and probe-based approaches. Probing experiments use one-shot optimization (Dunefsky & Cohan 2025): the model is frozen, a trainable vector is optimized to maximize the target sentence probability and suppress the original sentence probability, running for 30 steps with lr=0.1.

## Key Experimental Results

### Main Results

| Model / Layer | Validation | Key Result | Interpretation |
|--------------|------------|------------|---------------|
| Gemma-2B-IT layer 9 | Prediction 1 (trade-off) | $\rho$↓ Concept activation advances + output collapse advances | Normal component magnitude indeed bidirectionally regulates steering |
| Gemma-2B-IT layer 9 | Prediction 2 (axis determines normal plane) | $\rho_k$ curve is unimodal + lowest p-value | Normal plane can be determined by $\mathbf{v}_d$ |
| Gemma-2B-IT layer 9 | Prediction 3 (sector undetermined) | $\mathbf{v}_d$ cosine similarity vs. steering difference Pearson = -0.034 (p > 0.05) | Similar $\mathbf{v}_d$ does not imply similar steering behavior |

Experiments are also validated on LLaMA2-7B-Chat layers 16/24, with consistent conclusions.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|---------------|------------|-------------|
| Full CRH | All three predictions satisfied | Cylindrical structure holds |
| Different steering construction methods (DiffMean / PCA / MC / probe) | Also matches CRH predictions | Cylindrical structure is independent of steering method choice |
| Normal component completely zeroed $\rho=1$ | Most stable output but slow concept activation | Verifies that the axis alone cannot rapidly activate, normal component is needed for assistance |

### Key Findings
- Concept activation and output collapse are two sides of the same normal component: increasing $\|\mathbf{v}_{\perp}\|$ accelerates target concept activation but also pushes representations off the semantic manifold earlier, explaining why "steering always fails abruptly near a threshold" in practice.
- The unimodal result in Prediction 2 confirms that "axis → normal plane" is determined, i.e., the "shell shape" of the cylinder is uniquely determined by the difference vector, but the sensitive sector's position on the shell varies completely by sample.
- The zero correlation in Prediction 3—"similar $\mathbf{v}_d$ does not imply similar steering behavior"—is CRH's strongest counterexample weapon: it directly proves that predicting steering success solely from the difference vector is doomed to fail, explaining why "selecting concept vectors by similarity" strategies consistently fail in practice.

## Highlights & Insights
- Abandoning "orthogonality" while retaining "linearity" is a very clever move: it preserves the engineering friendliness of LRH (vector addition/subtraction is still possible) and simultaneously resolves the engineering puzzle of "why similar directions yield very different effects."
- The cylindrical geometry redefines steering failure from "engineering noise" to "intrinsic geometric uncertainty," implying that any attempt to eliminate steering fluctuations by "better concept vector construction" is fundamentally limited; the community should shift to "searching within the sensitive sector" rather than "finding purer directions."
- Using one-shot optimization to probe the cylindrical structure is a smart experimental design: it borrows tools for "mapping output space back to representation space" as "local geometric probes," circumventing the unobservability of true concept directions.

## Limitations & Future Work
- CRH models concepts as a finite set of fixed directions, but LLM "concepts" may themselves drift across layers and contexts; this work does not discuss context-dependent concept directions.
- The sensitive sector criterion $\beta_c$ vs $\sum_{i\neq c}\beta_i$ is sufficient but not necessary; actual boundaries are more complex, and the paper does not provide a feasible method for actively estimating the sector.
- Validation only covers two medium-scale LLMs (2B / 7B); whether it holds for 70B+ models and whether the sector structure is more complex remain open questions.
- Probing experiments are only conducted on two layers (layers 9/13 and 16/24); whether CRH holds across all layers and how it varies with depth merit further study.

## Related Work & Insights
- **vs Linear Representation Hypothesis (Park et al. 2024)**: CRH is a strict extension of LRH, removing the "orthogonal separability" assumption; LRH is a degenerate case of CRH when $d\geq n$ (dimension is large enough).
- **vs Toy Models of Superposition (Elhage et al. 2022)**: Superposition emphasizes the inevitable interference when the number of features exceeds the dimension; CRH provides a concrete, geometrically characterizable local model (cylinder) for such interference.
- **vs AxBench / steering benchmark evaluations**: CRH explains why steering success rates fluctuate greatly in benchmarks like AxBench and recommends evaluating at the population rather than sample level.
- **vs Multidimensional Concept Geometry (Engels et al. 2025)**: They point out that "not all features are one-dimensional linear"; CRH agrees but takes a different route—not questioning linearity, but allowing non-orthogonality, which is more empirically accessible geometrically.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective of "replacing the orthogonality assumption with cylindrical geometry" is highly original, and attributing sample-specific randomness to sensitive sectors geometrically is a much-needed explanatory framework for the community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Probing experiments are ingeniously designed, each of the three predictions is quantitatively validated, and cross-model and cross-construction method comparisons are conducted.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, but the notation is dense and geometric illustrations are abstract, making initial reading challenging.
- Value: ⭐⭐⭐⭐⭐ Directly provides a new theoretical coordinate for the entire activation steering subfield, with directional significance for future "interpretable + controllable" engineering improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[AAAI 2026\] Hypothesis Generation via LLM-Automated Language Bias for ILP](../../AAAI2026/interpretability/hypothesis_generation_via_llm-automated_language_bias_for_ilp.md)
- [\[NeurIPS 2025\] Steering Information Utility in Key-Value Memory for Language Model Post-Training](../../NeurIPS2025/interpretability/steering_information_utility_in_key-value_memory_for_language_model_post-trainin.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)

</div>

<!-- RELATED:END -->
