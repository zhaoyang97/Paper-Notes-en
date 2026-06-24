---
title: >-
  [Paper Note] Exogenous Isomorphism for Counterfactual Identifiability
description: >-
  [ICML2025 Spotlight][Causal Inference][Counterfactual Identifiability] This paper proposes the concept of **Exogenous Isomorphism (EI)**, proving that $\sim_{\mathrm{EI}}$-identifiability implies $\sim_{\mathcal{L}_3}$-identifiability (complete counterfactual layer identifiability). It provides sufficient conditions for achieving EI in two special classes of models: Bijective SCMs (BSCMs) and Triangular Monotonic SCMs (TM-SCMs), thereby unifying and generalizing existing coun…
tags:
  - "ICML2025 Spotlight"
  - "Causal Inference"
  - "Counterfactual Identifiability"
  - "Pearl's Causal Hierarchy"
  - "Exogenous Isomorphism"
  - "Structural Causal Model"
  - "Triangular Monotonic SCM"
date: 2026-05-08
content_hash: f20ef0e839797346
---

# Exogenous Isomorphism for Counterfactual Identifiability

**Conference**: ICML2025 Spotlight  
**arXiv**: [2505.02212](https://arxiv.org/abs/2505.02212)  
**Code**: [cyisk/tmscm](https://github.com/cyisk/tmscm)  
**Area**: Causal Inference  
**Keywords**: Counterfactual Identifiability, Pearl's Causal Hierarchy, Exogenous Isomorphism, Structural Causal Model, Triangular Monotonic SCM

## TL;DR

This paper proposes the concept of **Exogenous Isomorphism (EI)**, proving that $\sim_{\mathrm{EI}}$-identifiability implies $\sim_{\mathcal{L}_3}$-identifiability (complete counterfactual layer identifiability). It provides sufficient conditions for achieving EI in two special classes of models: Bijective SCMs (BSCMs) and Triangular Monotonic SCMs (TM-SCMs), thereby unifying and generalizing existing counterfactual identifiability theories.

## Background & Motivation

- **Counterfactual reasoning** is the highest layer ($\mathcal{L}_3$) of Pearl's Causal Hierarchy (PCH) and encodes all causal information of an SCM.
- Prior work either identifies only specific counterfactual effects (structural constraints) or only counterfactual outcomes (functional constraints), lacking a unified identification framework for the **entire counterfactual layer**.
- $\mathcal{L}_3$-identifiability requires that all SCMs satisfying the assumptions yield consistent answers to **any** counterfactual query, making it the ultimate goal of causal identifiability within the PCH framework.
- Directly dealing with $\sim_{\mathcal{L}_3}$ from the definition of PCH is extremely difficult, while the assumption of completely recovering exogenous variables ($=$-identifiability) is overly restrictive.
- **Key Motivation**: To find an intermediate concept that is weaker than "perfect model recovery" but still guarantees full counterfactual consistency.

## Method

### Overall Architecture

1. Unifies causal identifiability problems from the perspective of model identifiability.
2. Proposes Exogenous Isomorphism (EI) as a bridge connecting model equivalence to counterfactual consistency.
3. Provides sufficient conditions for EI in two classes of models: Bijective SCMs (BSCMs) and Triangular Monotonic SCMs (TM-SCMs).
4. Implements practical counterfactual inference using neural TM-SCMs.

### Core Concept: Exogenous Isomorphism

**Definition**: Two recursive SCMs $\mathcal{M}^{(1)}$ and $\mathcal{M}^{(2)}$ are exogenously isomorphic ($\sim_{\mathrm{EI}}$) if there exists a shared causal ordering and component-wise bijections $\mathbf{h} = (h_i)_{i \in \mathcal{I}}$ such that:

- **Component-wise Bijection**: Each $h_i: \Omega_{U_i}^{(1)} \to \Omega_{U_i}^{(2)}$ is a bijection.
- **Exogenous Distribution Isomorphism**: $P_{\mathbf{U}}^{(2)} = \mathbf{h}_{\sharp} P_{\mathbf{U}}^{(1)}$
- **Causal Mechanism Isomorphism**: $f_i^{(2)}(\mathbf{v}, h_i(u_i^{(1)})) = f_i^{(1)}(\mathbf{v}, u_i^{(1)})$

**Core Theorem (Theorem 3.2)**: $\sim_{\mathrm{EI}}$ implies $\sim_{\mathcal{L}_3}$, meaning two exogenously isomorphic SCMs yield identical results for all counterfactual statements.

### EI on Bijective SCMs (BSCMs)

- BSCM Definition: The solver mapping $\Gamma$ is bijective, which is equivalent to every $f_i(\mathbf{v}, \cdot)$ being bijective for a fixed $\mathbf{v}$.
- **Counterfactual Transport**: Defined as $K_{\mathcal{M},i}(\cdot, \mathbf{v}, \mathbf{v}') = (f_i(\mathbf{v}', \cdot)) \circ (f_i(\mathbf{v}, \cdot))^{-1}$.
- In Markov BSCMs, counterfactual transport is the transport map between conditional distributions.
- **Theorem 4.6**: Given BSCM + causal ordering + observational distribution + counterfactual transport $\to$ $\sim_{\mathrm{EI}}$-identifiable.
- **Theorem 4.8**: If the counterfactual transport happens to be the KR transport, then only BSCM + causal ordering + Markov + observational distribution is required.

### EI on Triangular Monotonic SCMs (TM-SCMs)

- TM Mapping: A triangular mapping where each component is strictly monotonic with respect to its last variable.
- TM-SCM Definition: The solver mapping, after vectorized rearrangement, is a TM mapping.
- **Key Property**: Composition and inversion of TM mappings preserve the TM property; the composition of two TM mappings with the same monotonic signature yields a TMI mapping.
- **Corollary 5.4 (Core Corollary)**: TM-SCM + causal ordering + Markov + observational distribution $\to$ $\sim_{\mathrm{EI}}$-identifiable.

This corollary unifies the findings of Lu et al. 2020, Nasr-Esfahany et al. 2023, and Scetbon et al. 2024.

### Loss & Training

Neural TM-SCMs are trained using maximum likelihood estimation, with the loss function being the negative log-likelihood (NLL):

$$\arg\min_\theta -\sum_{i=1}^N \log p_{\mathbf{V}_\theta}(\mathbf{v}^{(i)})$$

The exogenous distribution is modeled using an unconstrained normalizing flow (MAF) to satisfy Markov independence.

### Four Neural TM-SCM Prototypes

| Prototype | Functional Form | Representative Work |
|------|---------|---------|
| DNME | Diagonal Noise: $f_{i,\theta} = \mathbf{b} + \mathbf{a} \odot \mathbf{u}_i$ | LSNM |
| TNME | Triangular Noise: $f_{i,\theta} = \mathbf{b} + \mathbf{A} \mathbf{u}_i^\intercal$ | FiP |
| CMSM | Solver map = Composition of multiple TM maps | CausalNF |
| TVSM | Solver map defined by a triangular velocity field ODE | CFM |

## Key Experimental Results

### Synthetic Datasets

| Dataset | Description |
|--------|------|
| TM-SCM-Sym | 4 synthetic datasets (Barbell, Stair, Fork, Backdoor), with $\le 4$ causal variables |

Experiments utilize synthetic datasets to evaluate the effectiveness of neural TM-SCMs in addressing counterfactual consistency:
- Trained solely on samples from the observational distribution, and consistency is evaluated on a counterfactual test set.
- All four prototype models can learn effectively and yield counterfactual outcomes consistent with the ground-truth SCM.

### Key Findings

- Verified the theoretical validity of Corollary 5.4: TM-SCM models do achieve $\mathcal{L}_3$-consistency when their assumptions are fulfilled.
- Different prototypes (DNME, TNME, CMSM, TVSM) display unique trade-offs between expressivity and computational efficiency.
- The specific implementation of the exogenous distribution does not affect identifiability (consistent with theoretical predictions).

## Highlights & Insights

1. **Precise Characterization of the EI Concept**: Identifies a suitable equivalence relation between "exact model recovery" and "counterfactual consistency," precisely capturing the strength of model identification required to achieve complete counterfactual identifiability.
2. **Unification of Existing Theories**: Corollary 5.4 unifies three lines of work—Lu, Nasr-Esfahany, and Scetbon—as special cases of a single corollary.
3. **Generalization from Scalar to Vector**: Extends the endogenous variable space from $\mathbb{R}$ to $\mathbb{R}^{d_i}$, supporting a broader class of SCMs.
4. **New Perspective on Counterfactual Transport**: Establishes a connection between counterfactual identifiability and optimal transport theory, providing a novel interpretation.
5. **End-to-End Path from Theory to Practice**: Transitions from abstract equivalence classes to concrete neural network implementations, facilitating theoretically guaranteed deployments.

## Limitations & Future Work

1. **Strong TM-SCM Assumptions**: The requirement of strict monotonicity on causal mechanisms excludes many non-monotonic relationships commonly found in the real world.
2. **Assumed Causal Ordering**: All theoretical results rely on a known causal ordering, which itself remains a challenging problem in causal discovery.
3. **Limited to Recursive SCMs**: Non-recursive SCMs (i.e., those containing causal loops) are not discussed.
4. **Evaluation on Synthetic Data Only**: Experiments are conducted solely on synthetic data; effectiveness on real-world data (such as in healthcare or fairness scenarios) has yet to be verified.
5. **Gap Between $\sim_{\mathrm{EI}}$ and $\sim_{\mathcal{L}_3}$**: While EI is a sufficient condition for $\mathcal{L}_3$-consistency, it is not a necessary one, and it remains unclear whether a weaker yet still sufficient condition exists.

## Related Work & Insights

- **Counterfactual Equivalence**: The counterfactual equivalence defined by Peters et al. 2017 requires identical exogenous distributions, which is stronger than EI.
- **BGM Equivalence**: The equivalence relation defined by Nasr-Esfahany et al. 2023 within the BSCM framework is a special case of EI.
- **CausalNF**: Javaloy et al. 2023 established representation identifiability of TMI mappings; this work extends it to the complete counterfactual layer.
- **Optimal Transport**: The KR transport provides a canonical construction of counterfactual transport and shares a profound connection with TMI mappings (Lemma 5.1).
- **Future Directions**: Extending EI to semi-Markov SCMs and causal representation learning with latent variables.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The concept of exogenous isomorphism is highly novel, precisely capturing the "optimal intermediate layer" of identifiability.
- Experimental Thoroughness: ⭐⭐⭐ — Evaluated only on small-scale synthetic datasets.
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous, though highly dense in notation, presenting a steep learning curve.
- Value: ⭐⭐⭐⭐⭐ — Unifies and generalizes the core theories of counterfactual identifiability in causal inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Identifiability of Causal Graphs with the Invariance Principle](../../ICLR2026/causal_inference/on_the_identifiability_of_causal_graphs_with_the_invariance_principle.md)
- [\[ICML 2025\] Transformer-Based Spatial-Temporal Counterfactual Outcomes Estimation](transformer-based_spatial-temporal_counterfactual_outcomes_estimation.md)
- [\[ICML 2025\] Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes](classifier_reconstruction_through_counterfactual-aware_wasserstein_prototypes.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)

</div>

<!-- RELATED:END -->
