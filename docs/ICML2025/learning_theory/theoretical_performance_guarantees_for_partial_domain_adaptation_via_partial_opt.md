---
title: >-
  [Paper Note] Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport
description: >-
  [ICML 2025][Transfer Learning][partial domain adaptation] This paper derives generalization bounds for Partial Domain Adaptation (PDA) based on the theory of partial optimal transport, proves the rationality of using partial Wasserstein distance as the domain alignment term alongside the proposed theoretically-driven weighting scheme, and subsequently develops a practical algorithm named WARMPOT.
tags:
  - "ICML 2025"
  - "Transfer Learning"
  - "Optimal Transport"
  - "partial domain adaptation"
  - "partial optimal transport"
  - "Wasserstein distance"
  - "generalization bounds"
  - "WARMPOT"
date: 2026-05-08
content_hash: 77dcd2443b3936fa
---

# Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport

**Conference**: ICML 2025  
**arXiv**: [2506.02712](https://arxiv.org/abs/2506.02712)  
**Code**: None  
**Area**: Transfer Learning / Optimal Transport  
**Keywords**: partial domain adaptation, partial optimal transport, Wasserstein distance, generalization bounds, WARMPOT

## TL;DR
This paper derives generalization bounds for Partial Domain Adaptation (PDA) based on the theory of partial optimal transport, proves the rationality of using partial Wasserstein distance as the domain alignment term alongside the proposed theoretically-driven weighting scheme, and subsequently develops a practical algorithm named WARMPOT.

## Background & Motivation
**Background**: Domain adaptation (DA) aims to transfer knowledge from a labeled source domain to a target domain where labels are scarce. Partial domain adaptation (PDA) is a more realistic setting: the target domain's label space is a subset of the source domain's label space (e.g., the source domain has 100 classes, while the target domain only has 50 classes).

**Limitations of Prior Work**: PDA methods typically minimize a domain alignment term plus a weighted source domain empirical loss, but: (1) the choice of the alignment term lacks theoretical justification; (2) the weighting schemes are mostly heuristic (e.g., weights based on class prediction probabilities); (3) they lack generalization guarantees.

**Key Challenge**: Although PDA methods are effective in practice, their theoretical foundation remains weak, and there lacks theoretical explanation for the strengths and weaknesses of different weighting schemes.

**Goal**: Provide theoretical generalization bounds for PDA, from which a principled weighting formula and domain alignment metric can be derived.

**Key Insight**: Partial Optimal Transport (Partial OT)—a variant of optimal transport that only transports a fraction of the mass—is naturally suited for handling asymmetric label spaces.

**Core Idea**: Partial Wasserstein distance is the correct domain alignment metric in PDA, and the generalization bounds naturally yield the optimal weights for the source domain samples.

## Method

### Overall Architecture
Input: Labeled source domain data $D_S$ ($C_S$ classes), unlabeled target domain data $D_T$ ($C_T \subseteq C_S$ classes)  
Output: Classifier $h$ that is accurate on the target domain

### Key Designs

1. **Generalization Bound based on Partial Optimal Transport**:

    - Function: Derives the upper bound of the target domain risk.
    - Mechanism: 
    $\mathcal{R}_T(h) \leq \sum_i w_i \ell(h(x_i^S), y_i^S) + \lambda \cdot W_s^{(p)}(\hat{\mu}_S^w, \hat{\mu}_T) + \text{complexity terms}$
      where $W_s^{(p)}$ is the partial Wasserstein distance (transporting only a fraction $s = |C_T|/|C_S|$ of the mass), and $w_i$ is the weight of the source sample.
    - Design Motivation: Standard Wasserstein distance is unsuitable in PDA (as it forces alignment of samples from source-only classes), while partial Wasserstein distance automatically ignores irrelevant source domain samples.

2. **Theoretically-driven Weighting Scheme**:

    - Function: Derives optimal weights for source domain samples from the generalization bound.
    - Mechanism: In the dual problem of partial OT, the transport plan $\pi^*$ naturally indicates which source samples are relevant to the target domain. The optimal weight is proportional to the marginal of this transport plan:
    $w_i^* \propto \sum_j \pi^*(x_i^S, x_j^T)$
      That is, the weight of a source sample depends on how much it is "transported" to target samples.
    - Design Motivation: Heuristic weights (e.g., those based on classifier predictions) ignore the geometric structure of the data distribution, whereas OT weights are directly based on the distance relationships between samples.

3. **WARMPOT Algorithm**:

    - Function: A practical PDA algorithm.
    - Mechanism: 
        - Step 1: Map source and target data into a feature space using a feature extractor $g$.
        - Step 2: Compute the partial optimal transport plan $\pi^*$ in the feature space.
        - Step 3: Extract source sample weights $w_i$ from $\pi^*$.
        - Step 4: Minimize $\sum_i w_i \ell(h(g(x_i^S)), y_i^S) + \lambda W_s^{(p)}(g_\#\hat{\mu}_S^w, g_\#\hat{\mu}_T)$.
        - Iteratively update $g, h, w$.
    - Design Motivation: To directly translate the theory into an implementable algorithm.

### Loss & Training
$$\mathcal{L} = \sum_i w_i \cdot \text{CE}(h(g(x_i^S)), y_i^S) + \lambda \cdot \hat{W}_s^{(p)}(g(X_S), g(X_T))$$
The partial Wasserstein distance is efficiently approximated via a partial transport variant of the Sinkhorn algorithm.

## Key Experimental Results

### Main Results

| Dataset | Metric (ACC) | WARMPOT | PADA | ETN | BA3US |
|---|---|---|---|---|---|
| Office-Home (A→C) | Target ACC | **72.3** | 68.7 | 69.1 | 71.5 |
| Office-Home (R→P) | Target ACC | **81.5** | 78.3 | 79.0 | 80.8 |
| Office-31 (A→W) | Target ACC | **93.2** | 89.5 | 90.1 | 92.4 |
| VisDA (12→6 classes) | Target ACC | **85.7** | 81.2 | 82.5 | 84.3 |

### Ablation Study

| Weighting Scheme | Office-Home Avg ACC | Description |
|---|---|---|
| WARMPOT (OT Weights) | **72.3** | Theoretically-derived weights in this paper |
| Uniform Weights | 64.8 | No weighting |
| Weights based on classifier predictions | 69.5 | PADA-style |
| Weights based on domain discriminator | 70.1 | DANN-style |
| Replaced with standard Wasserstein | 68.2 | Non-partial OT |
| Replaced with MMD alignment | 67.5 | Non-OT method |

### Key Findings
- WARMPOT achieves competitive performance or outperforms state-of-the-art methods across multiple PDA benchmarks.
- OT weights significantly outperform heuristic weighting schemes (uniform, prediction-based, discriminator-based).
- The partial Wasserstein distance is more suitable than the standard Wasserstein distance and MMD as an alignment term for PDA.
- Theoretical bounds align consistently with empirical performance trends.

## Highlights & Insights
- A solid closed loop of Theory $\leftrightarrow$ Practice: The method is derived from generalization bounds, and the method in turn validates the theory.
- Natural correspondence between Partial OT and PDA: Transporting partial mass $\leftrightarrow$ aligning only the shared classes.
- Clear geometric intuition of OT weights: Source samples far from the target domain automatically receive lower weights.

## Limitations & Future Work
- The partial ratio $s$ needs to be pre-set or estimated (it is usually assumed that the ratio of the target domain's class count is known).
- The computational efficiency of partial OT in high-dimensional feature spaces needs optimization.
- Exploring the connection with Open-Set Domain Adaptation (OSDA) is a promising direction.

## Related Work & Insights
- Complementary to the DA generalization bounds of Ben-David et al. (2010): This work focuses specifically on the partial setting.
- The theory of partial OT was founded by Caffarelli & McCann (2010) and Figalli (2010).
- Provides theoretical guidance for PDA practices: which alignment metric to select, and how to weight.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining partial OT and PDA generalization bounds holds significant theoretical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets plus comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless integration of theory and experiments.
- Value: ⭐⭐⭐⭐ Establishes a solid theoretical foundation for PDA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](../../ICML2026/learning_theory/semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[ICLR 2026\] Slicing Wasserstein over Wasserstein via Functional Optimal Transport](../../ICLR2026/learning_theory/slicing_wasserstein_over_wasserstein_via_functional_optimal_transport.md)
- [\[ICLR 2026\] Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality](../../ICLR2026/learning_theory/test-time_verification_via_optimal_transport_coverage_roc_sub-optimality.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](../../ICLR2026/learning_theory/a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)
- [\[NeurIPS 2025\] How Many Domains Suffice for Domain Generalization? A Tight Characterization via the Domain Shattering Dimension](../../NeurIPS2025/learning_theory/how_many_domains_suffice_for_domain_generalization_a_tight_characterization_via_.md)

</div>

<!-- RELATED:END -->
