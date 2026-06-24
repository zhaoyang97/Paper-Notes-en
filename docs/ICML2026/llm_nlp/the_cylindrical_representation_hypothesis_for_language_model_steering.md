---
title: >-
  [Paper Note] The Cylindrical Representation Hypothesis for Language Model Steering
description: >-
  [ICML 2026][LLM (Other)][Activation Steering] This paper proposes the Cylindrical Representation Hypothesis (CRH). By maintaining "concept linearity" while abandoning the orthogonality assumption of LRH, it demonstrates that the superposition of concept vectors naturally induces a cylindrical geometry consisting of an "axis + normal plane + sensitive sector." This provides the first geometric explanation for why activation steering is unpredictable at the sample level but obs…
tags:
  - "ICML 2026"
  - "LLM (Other)"
  - "Activation Steering"
  - "Linear Representation Hypothesis"
  - "Representation Geometry"
  - "Controllability"
  - "Concept Vectors"
date: 2026-05-08
content_hash: c0ae08e566b9815f
---

# The Cylindrical Representation Hypothesis for Language Model Steering

**Conference**: ICML 2026  
**arXiv**: [2605.01844](https://arxiv.org/abs/2605.01844)  
**Code**: https://github.com/mbzuai-nlp/CRH  
**Area**: LLM/NLP / Representation Geometry / Interpretability  
**Keywords**: Activation Steering, Linear Representation Hypothesis, Representation Geometry, Controllability, Concept Vectors

## TL;DR
This paper proposes the Cylindrical Representation Hypothesis (CRH). By maintaining "concept linearity" while abandoning the orthogonality assumption of LRH, it demonstrates that the superposition of concept vectors naturally induces a cylindrical geometry consisting of an "axis + normal plane + sensitive sector." This provides the first geometric explanation for why activation steering is unpredictable at the sample level but observable at the group level.

## Background & Motivation

**Background**: LLM "activation steering" has become a mainstream tool in interpretability and alignment research: by adding a concept direction vector $\mathbf{v}$ to the residual stream of a certain layer, one can promote or suppress outputs according to that concept during inference. Existing theories are almost entirely based on the Linear Representation Hypothesis (LRH, Park et al. 2024)—assuming concepts correspond to linear directions and can be orthogonalized and independently manipulated through "causal inner products."

**Limitations of Prior Work**: Real-world steering is extremely unstable; the same direction vector produces vastly different effects across different samples. "Controllability predictions" based on LRH (such as representation separation) correlate weakly with measured steering success rates, making them practically unreliable for engineering.

**Key Challenge**: The "lossless orthogonalization" assumed by LRH is fundamentally impossible in finite dimensions when the number of concepts exceeds the number of dimensions—any 2D space can hold at most 2 orthogonal directions, yet LLMs must represent thousands of concepts. Therefore, overlap between concepts is inevitable, and the theoretical assumption of orthogonality is inherently flawed.

**Goal**: (i) Relax LRH to "linearly additive but non-orthogonal" and examine the resulting geometric structure; (ii) use this new structure to explain why steering is random at the sample level yet estimable at the group level; (iii) provide empirically testable predictions.

**Key Insight**: The authors retain the "soft core" that concept vectors are linear but allow arbitrary angles between them. Starting from the simplest premise that "difference vectors = linear combinations of multiple concepts," they derive that local geometry inevitably presents a triplet of "central axis + normal plane + phase."

**Core Idea**: The sample-concept local geometry is modeled as a cylinder: the difference vector defines the central axis, while the projections of all concepts onto the normal plane cancel each other out. The phase of the steering vector within the normal plane is the key determinant of steering success—and since this phase cannot be derived from the axis and $\mathbf{v}$ alone, steering is inherently unpredictable.

## Method

### Overall Architecture
CRH does not train any models but performs geometric derivation from a simple assumption: the difference vector $\mathbf{v}_d=\mathbf{r}_a-\mathbf{r}_b=\sum_i\alpha^{(i)}\mathbf{a}^{(i)}$ corresponding to "concept promotion/suppression" is a linear superposition of multiple concept directions. The paper proves that as long as this linear superposition is maintained and the orthogonality of LRH is relaxed, the local geometry is forced to present a cylindrical structure of "central axis + normal plane + sensitive sector." The difference vector defines the axis, the projections of all concepts on the normal plane cancel each other out, and the phase of the steering vector in the normal plane determines whether it activates or suppresses the target concept. Following this logic, the method provides an axis-normal decomposition, sensitive sector criteria, and a predictability dichotomy theorem, ultimately resulting in three corollaries testable on Gemma-2B / LLaMA2-7B.

### Key Designs

**1. Axis-Normal Decomposition + Normal Balance Theorem: Geometrically modeling "concept superposition" as a cylinder**

LRH assumes concepts are independently arranged along a set of globally orthogonal directions, which fails when thousands of concepts are packed into finite dimensions. CRH takes a different perspective: for any difference vector $\mathbf{v}_d$, first define the central axis $\mathbf{a}_d=\mathbf{v}_d/\|\mathbf{v}_d\|$, then perform a standard projection of each concept direction $\mathbf{a}^{(i)}$ onto this axis: $\mathbf{v}^{(i)}=d^{(i)}\mathbf{a}_d+\mathbf{v}_{\perp}^{(i)}$. Substituting this back into $\mathbf{v}_d=\sum_i\mathbf{v}^{(i)}$ yields two constraints: the sum of axial components exactly completes the axis $\sum_i d^{(i)}=\|\mathbf{v}_d\|$, and all normal components must cancel each other out $\sum_i\mathbf{v}_{\perp}^{(i)}=\mathbf{0}$. That is, the difference vector simultaneously defines a 1D principal axis and a normal state where non-axial contributions balance each other. Using PCA, a 2D normal plane $\mathcal{P}_d=\text{span}(\mathbf{a}_{\perp}^{(c)},\text{PC}_1(\{\mathbf{a}_{\perp}^{(i)}\}_{i\neq c}))$ is extracted from these normal components, and the balance relationship still holds on this plane. This step is the foundation—it replaces the "global single direction of LRH" with a "sample-specific axis + local normal plane," so steering is no longer sliding along a line but "passing through" or "bypassing" concepts along a cylindrical surface.

**2. Sensitive Sectors & Steering Decomposition: Why vectors with similar angles have vastly different effects**

Given the cylinder, the next question is what happens when a steering vector $\mathbf{v}$ is added. It is decomposed along the cylinder: $\mathbf{v}=\mathbf{v}_{\text{axis}}+\mathbf{v}_{\perp,\mathcal{P}_d}+\boldsymbol{\epsilon}$, where the part falling into the normal plane is expanded by concepts as $\mathbf{v}_{\perp,\mathcal{P}_d}=\beta_c\mathbf{v}_{\perp,\mathcal{P}_d}^{(c)}+\sum_{i\neq c}\beta_i\mathbf{v}_{\perp,\mathcal{P}_d}^{(i)}$. The paper uses a simple "major contributor" criterion: when the coefficient of the target concept $\beta_c>\sum_{i\neq c}\beta_i$, the vector falls into the **high-sensitivity sector**, strengthening the axial drive and rapidly activating the target concept; otherwise, it falls into the **low-sensitivity sector**, dominated by competing concepts, causing delayed or even suppressed activation. This sufficient condition introduces no extra parameters but perfectly explains a counter-intuitive engineering phenomenon: two steering vectors with nearly identical angles may fall into different sectors after being projected onto normal planes with the same shape but different orientations on the cylinder, resulting in completely opposite effects across different samples.

**3. Predictability Dichotomy Theorem + Three Observable Corollaries: Turning geometric properties into falsifiable experiments**

The most significant assertion of CRH is a set of dual theorems: the "magnitude" of the normal plane is predictable, while the "sector" is not. Theorem 4.1 states that $\|\mathbf{v}_{\perp,\mathcal{P}_d}\|$ is a reliable proxy for steering intensity, meaning the shape of the cylindrical shell is uniquely determined by the difference vector. Lemma 4.2 and Theorem 4.3 point out that once more than $d$ concept directions are packed into a $d$-dimensional space, the mapping from difference vector to concept intensity is no longer injective; thus, the location of the sensitive sector cannot be reverse-engineered from $\mathbf{v}_d$. This "predictable vs. unpredictable" duality distinguishes CRH from a mere geometric description, making it a testable theory and directly deriving three experimental entry points: first, suppressing the normal component $\rho\mathbf{v}_{\perp}\to0$ will simultaneously delay concept activation and output collapse, showing a trade-off; second, fitting $\text{St}_c(\mathbf{r};\mathbf{v})/\|\mathbf{v}_d\|^k$ to $\sin^m\theta\cos^{k-m}\theta$ should produce an unimodal peak, proving the normal plane is determined by the axis; third, if sectors were also determined by the axis, samples with similar $\mathbf{v}_d$ should have similar steering effects—if they do not, it proves sectors cannot be predicted by the axis.

### Loss & Training
No models are trained in this paper. All steering vectors are constructed from contrastive pairs (positive/negative samples) using standard methods such as DiffMean, PCA, Mean-Centering, and probe-based methods. Probing experiments use one-shot optimization (Dunefsky & Cohan 2025): the model is frozen, and a trainable vector is optimized to maximize the probability of the target sentence while suppressing the original sentence probability, running for 30 steps with lr=0.1, using it as a local geometric probe to "map the output space back to the representation space."

## Key Experimental Results

### Main Results

| Model / Layer | Validation | Key Results | Interpretation |
|-----------|------|----------|------|
| Gemma-2B-IT layer 9 | Corollary 1 (trade-off) | $\rho$↓ Concept activation delayed + output collapse delayed | Normal component magnitude indeed bidirectionally regulates steering |
| Gemma-2B-IT layer 9 | Corollary 2 (Axis determines plane) | Unimodal $\rho_k$ curve + lowest p-value | Normal plane can be determined by $\mathbf{v}_d$ |
| Gemma-2B-IT layer 9 | Corollary 3 (Sector unpredictable) | $\mathbf{v}_d$ cos similarity vs steering diff Pearson = -0.034 (p > 0.05) | Similarity in $\mathbf{v}_d$ does not imply similarity in steering behavior |

Experiments were synchronized and validated on LLaMA2-7B-Chat layers 16/24 with consistent conclusions.

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Full CRH | All Corollaries 1/2/3 satisfied | Cylindrical structure holds |
| Various steering methods (DiffMean / PCA / MC / probe) | Also conform to CRH predictions | Cylindrical structure is independent of steering method choice |
| Normal component zeroed $\rho=1$ | Most stable output but slow concept activation | Validates that the axis alone cannot rapidly activate; normal assistance is required |

### Key Findings
- Concept activation and output collapse are two sides of the same normal component: increasing $\|\mathbf{v}_{\perp}\|$ enables faster activation of the target concept but also pushes the representation off the reasonable semantic manifold earlier, explaining the "all-or-nothing" threshold in engineering steering.
- The unimodal peak in Corollary 2 confirms that "axis → normal plane" is deterministic, equivalent to saying the "shell shape" of the cylinder is uniquely determined by the difference vector, but the location of the sensitive sector on that shell varies entirely by sample.
- The zero-correlation result in Corollary 3, where "similarity in $\mathbf{v}_d$ does not imply similarity in steering behavior," is the strongest counter-proof of CRH—it directly demonstrates that predicting steering success based solely on the difference vector is destined to fail, explaining why strategies of choosing concept vectors based on similarity consistently fail in engineering.

## Highlights & Insights
- Moving away from "orthogonality" without abandoning "linearity" is highly effective: it retains the engineering friendliness of LRH (vector arithmetic still works) while simultaneously resolving the engineering mystery of why similar directions produce vastly different results.
- Cylindrical geometry redefines steering failure from "engineering noise" to "intrinsic geometric uncertainty." This implies that any effort to eliminate steering fluctuations by "better concept vector construction" has a deterministic upper bound; community strategy should shift toward "searching within the sensitive sector" rather than "finding a purer direction."
- Using one-shot optimization to probe the cylindrical structure is a clever experimental design: borrowing a tool that "maps the output space back to the representation space" as a "local geometric probe" bypasses the difficulty of unobservable true concept directions.

## Limitations & Future Work
- CRH models concepts as a finite set of fixed directions, but LLM "concepts" may themselves drift across layers or contexts; this paper does not discuss context-dependent concept directions.
- The sensitive sector criterion $\beta_c$ vs $\sum_{i\neq c}\beta_i$ is a sufficient rather than a necessary condition; real boundaries are more complex, and the paper does not provide a practical method for actively estimating sectors.
- Validation only covered two medium-scale LLMs (2B / 7B); whether this holds on 70B+ models or if sector structures become more complex remains an open question.
- Probing experiments were only conducted on a few layers (e.g., 9/13 and 16/24); whether CRH holds across all layers and how it changes with depth warrants follow-up research.

## Related Work & Insights
- **vs Linear Representation Hypothesis (Park et al. 2024)**: CRH is a strict extension of LRH, removing the "orthogonal separability" assumption; LRH is a degenerate case of CRH when $d\geq n$ (sufficiently high dimensions).
- **vs Toy Models of Superposition (Elhage et al. 2022)**: Superposition emphasizes unavoidable interference when "features > dimensions"; CRH provides a specific, geometrically characterizable local model (the cylinder) for this interference.
- **vs AxBench / steering benchmarks**: CRH explains the massive success rate fluctuations in benchmarks like AxBench and provides a methodological suggestion that "evaluation should be conducted at the group level rather than the sample level."
- **vs Multi-dimensional concept geometry (Engels et al. 2025)**: They point out that "not all features are 1D linear"; CRH agrees but takes a different path—not by questioning linearity, but by allowing non-orthogonality, which is geometrically easier to validate.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective of "replacing orthogonality with cylindrical geometry" is highly original and geometrically attributes sample-specific randomness to sensitive sectors, providing a much-needed explanatory framework for the community.
- Experimental Thoroughness: ⭐⭐⭐⭐ Probing experiments are ingeniously designed, the three corollaries are quantitatively validated, and cross-comparisons across multiple models and construction methods are provided.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, but the notation density is high and geometric diagrams are abstract, making initial reading challenging.
- Value: ⭐⭐⭐⭐⭐ Directly provides a new theoretical coordinate for the activation steering subfield and offers directional significance for future "interpretable + controllable" engineering improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](../../ICLR2026/llm_nlp/the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ACL 2025\] Representation Bending for Large Language Model Safety](../../ACL2025/llm_nlp/repbend_representation_bending_safety.md)
- [\[ICML 2026\] Creative Collision: Directorial Persona Steering and Competition in Large Language Models](creative_collision_directorial_persona_steering_and_competition_in_large_languag.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] PerFit: Exploring Personalization Shifts in Representation Space of LLMs](../../ICLR2026/llm_nlp/perfit_exploring_personalization_shifts_in_representation_space_of_llms.md)

</div>

<!-- RELATED:END -->
