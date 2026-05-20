---
title: >-
  [Paper Note] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets
description: >-
  [ICLR 2026][Lipschitz continuity] This paper systematically investigates the Lipschitz continuity of three commonly used set aggregation functions (sum, mean…
tags:
  - "ICLR 2026"
  - "Lipschitz continuity"
  - "set aggregation functions"
  - "multisets"
  - "robustness"
  - "generalization"
date: 2026-05-08
content_hash: dbdfa18a2f4d4114
---

# On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets

**Conference**: ICLR 2026
**arXiv**: [2505.24403](https://arxiv.org/abs/2505.24403)  
**Code**: None  
**Area**: Other
**Keywords**: Lipschitz continuity, set aggregation functions, multisets, robustness, generalization

## TL;DR

This paper systematically investigates the Lipschitz continuity of three commonly used set aggregation functions (sum, mean, max) and attention mechanisms under three multiset distance functions, derives upper bounds on the Lipschitz constants of set neural networks, and connects these results to perturbation stability and generalization under distribution shift.

## Background & Motivation

The Lipschitz constant of a neural network is closely related to its robustness and generalization ability. Prior work has primarily focused on estimating Lipschitz constants for MLPs and CNNs, yet in many practical settings (e.g., point cloud processing, natural language processing), inputs are naturally collections or multisets of vectors. Models such as DeepSets and PointNet typically employ permutation-invariant aggregation functions (sum, mean, max) to handle multiset inputs. Nevertheless, the Lipschitz continuity and stability of these aggregation functions remain insufficiently studied.

**Core Problem**: Under which multiset distance functions are standard aggregation functions Lipschitz continuous, and what are their Lipschitz constants? Do neural network models built upon these aggregation functions also preserve Lipschitz continuity?

## Method

### Overall Architecture

The paper considers three multiset distance functions — Earth Mover's Distance (EMD), Hausdorff distance, and Matching distance — and analyzes the Lipschitz continuity of three standard aggregation functions (sum, mean, max) as well as attention-based aggregation. The results are then extended to complete set neural networks that incorporate MLPs.

### Key Designs

1. **Lipschitz Continuity Analysis of Aggregation Functions**:
    - Function: Proves that each aggregation function is Lipschitz continuous with respect to exactly one distance function.
    - Mechanism: mean is continuous under EMD ($L=1$), sum under Matching distance ($L=1$), and max under Hausdorff distance ($L=\sqrt{d}$).
    - Design Motivation: A natural correspondence exists between aggregation functions and distance functions; understanding this correspondence guides appropriate model and metric selection.

2. **Extended Analysis for Equicardinality Multisets**:
    - Function: When all multisets share the same cardinality, aggregation functions exhibit Lipschitz continuity under additional distance functions.
    - Mechanism: Exploits the relationship between EMD and Matching distance for equal-sized multisets ($d_M = M \cdot d_{\mathrm{EMD}}$) to derive additional Lipschitz constants.
    - Design Motivation: In practice (e.g., point clouds), multiset sizes are often fixed, enabling stronger stability guarantees.

3. **Analysis of Attention Aggregation**:
    - Function: Proves that attention mechanisms are not Lipschitz continuous under any of the three considered distance functions.
    - Mechanism: Constructs counterexamples showing that even $\ell_2$ attention variants fail to achieve Lipschitz continuity.
    - Design Motivation: Exposes a fundamental limitation of attention mechanisms with respect to stability.

4. **Lipschitz Constant Upper Bounds for Set Neural Networks**:
    - Function: Derives Lipschitz upper bounds for $\mathrm{NN}_{\mathrm{mean}}$ and $\mathrm{NN}_{\mathrm{max}}$, and shows that $\mathrm{NN}_{\mathrm{sum}}$ may not be Lipschitz continuous.
    - Mechanism: Decomposes the network into $\mathrm{MLP}_1$ + aggregation + $\mathrm{MLP}_2$ and exploits the composability of Lipschitz constants.
    - Design Motivation: Provides theoretical tools for robustness analysis of practical models.

### Generalization Analysis

Leveraging the framework of Shen et al. (2018), the paper combines Lipschitz constants with Wasserstein distance to bound the target error:

$$\varepsilon_T(h) \leq \varepsilon_S(h) + 2L \cdot W_1(\mu_S, \mu_T) + \lambda$$

where EMD and Hausdorff distance serve as the underlying metrics for $\mathrm{NN}_{\mathrm{mean}}$ and $\mathrm{NN}_{\mathrm{max}}$, respectively.

## Key Experimental Results

### Main Results

| Dataset | Model | Perturbation Type | Accuracy Drop | Notes |
|---------|-------|-------------------|---------------|-------|
| ModelNet40 | $\mathrm{NN}_{\mathrm{mean}}$ | Element addition | 2.0% (±1.3) | Robust to single-element addition |
| ModelNet40 | $\mathrm{NN}_{\mathrm{max}}$ | Element addition | 20.1% (±1.8) | Sensitive to single-element addition |
| Polarity | $\mathrm{NN}_{\mathrm{mean}}$ | Random noise | 13.6% (±7.1) | Sensitive to distributed noise |
| Polarity | $\mathrm{NN}_{\mathrm{max}}$ | Random noise | 4.8% (±3.7) | Robust to distributed noise |

### Ablation Study

| Configuration | Correlation (small→large / large→small) | Notes |
|---------------|----------------------------------------|-------|
| $\mathrm{NN}_{\mathrm{mean}}$ + EMD | $r=0.92$ / $r=0.94$ | Wasserstein distance strongly correlated with generalization error |
| $\mathrm{NN}_{\mathrm{max}}$ + Hausdorff | $r=0.90$ / $r=0.90$ | Strong correlation similarly observed |

### Key Findings

- Each aggregation function exhibits a natural correspondence with exactly one distance function.
- Mean aggregation is more robust to "local" perturbations (single-element addition); max aggregation is more robust to "global" noise (small perturbations to all elements).
- Generalization error under distribution shift is strongly correlated with Wasserstein distance ($r > 0.90$).
- Attention mechanisms are not Lipschitz continuous under any of the three distance functions.

## Highlights & Insights

- The paper is the first to systematically establish Lipschitz continuity correspondences between set aggregation functions and multiset distance functions.
- Theoretical results show that sum aggregation, despite its greatest expressive power, may render models non-Lipschitz continuous (in the presence of non-zero bias).
- Each aggregation function exhibits complementary robustness characteristics under different perturbation types, providing a principled basis for model selection.

## Limitations & Future Work

- Only permutation-invariant aggregation functions are considered; permutation-equivariant message-passing architectures are not addressed.
- The derived Lipschitz upper bounds may be loose, particularly for the max function; tighter bounds remain to be explored.
- Experiments are conducted on simple three-layer networks, and applicability to more complex architectures has not been verified.
- The analysis of attention mechanisms yields only negative conclusions without proposing alternative designs.

## Related Work & Insights

- Complements the work of Chuang & Jegelka (2022), which investigates Lipschitz continuity in graph neural networks.
- Suggests that the design choice of "which aggregation function to use" in set learning can be made based on desired stability guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of Lipschitz properties of set aggregation functions, though the technical machinery is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theory and experiments mutually validate each other, though the scale of datasets is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-organized and rigorously stated theorems.
- Value: ⭐⭐⭐⭐ Provides an important theoretical foundation for robustness analysis of models in the set learning domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)
- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)

</div>

<!-- RELATED:END -->
