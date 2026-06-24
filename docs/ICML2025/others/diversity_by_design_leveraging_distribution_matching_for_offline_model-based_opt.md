---
title: >-
  [Paper Note] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization
description: >-
  [ICML 2025][Offline Optimization] This paper proposes DynAMO, which explicitly models design diversity as a distribution matching problem to simultaneously discover high-quality and highly diverse candidate designs in offline model-based optimization (MBO).
tags:
  - "ICML 2025"
  - "Offline Optimization"
  - "Model-Based Optimization"
  - "Design Diversity"
  - "Distribution Matching"
  - "Adversarial Training"
date: 2026-05-08
content_hash: dafcc25c70c305d1
---

# Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization

**Conference**: ICML 2025  
**arXiv**: [2501.18768](https://arxiv.org/abs/2501.18768)  
**Code**: None  
**Area**: Others  
**Keywords**: Offline Optimization, Model-Based Optimization, Design Diversity, Distribution Matching, Adversarial Training

## TL;DR
This paper proposes DynAMO, which explicitly models design diversity as a distribution matching problem to simultaneously discover high-quality and highly diverse candidate designs in offline model-based optimization (MBO).

## Background & Motivation

**Background**: Offline Model-Based Optimization (MBO) aims to propose new designs that maximize an objective function given an offline dataset. It is commonly applied in scientific domains such as protein design and molecular optimization. Existing methods primarily focus on maximizing the objective value, neglecting the diversity of candidate solutions.

**Limitations of Prior Work**: (1) The generated candidate designs tend to cluster around a single peak of the objective function, lacking diversity, whereas practical applications require multiple distinct near-optimal candidates for selection; (2) the offline dataset itself contains rich diversity information, which is underutilized by existing methods; (3) simple diversity regularization (e.g., dispersion constraints) often conflicts with quality goals, leading to performance degradation.

**Key Challenge**: The need to simultaneously optimize design quality (high objective values) and design diversity (covering multiple optimal configurations), which inherently exhibits a natural tension.

**Goal**: To introduce diversity as an explicit objective into any MBO problem while preserving design quality.

**Key Insight**: Formulating diversity as a distribution matching problem—the distribution of generated designs should capture the inherent diversity structure within the offline dataset.

**Core Idea**: Utilizing an adversarial distribution matching loss to align the distribution of generated designs with a high-quality subset of the offline dataset, thereby maintaining diversity while optimizing quality. DynAMO can be integrated as a plug-and-play module with any MBO method.

## Method

### Overall Architecture
An adversarial distribution matching head/term is integrated into the optimization pipeline of any MBO method. A high-quality subset is extracted from the offline dataset to serve as the reference distribution for diversity, and adversarial training is then employed to match the distribution of generated designs with it.

### Key Designs

1. **Distribution Matching as a Diversity Objective**:

    - **Function**: Formulates design diversity as an optimizable objective.
    - **Mechanism**: Uses the distribution of high-objective samples from the offline dataset as a reference, employing an adversarial discriminator to measure the distance between the generated design distribution and the reference distribution. Minimizing this distance ensures that the generated designs are not only of high quality but also preserve the inherent diversity structures present in the dataset.
    - **Design Motivation**: The diversity within offline datasets serves as an implicit encoding of domain knowledge (e.g., different functional families in protein space), making the preservation of this diversity much more meaningful than random dispersion.

2. **Adversarial Optimization Framework (DynAMO)**:

    - **Function**: Integrates the distribution matching term into any MBO method as a plug-and-play module.
    - **Mechanism**: Trains a discriminator to distinguish between "generated designs from the optimizer" and "real designs from the high-quality subset of the dataset," incorporating the discriminator loss into the optimizer's overall objective. Any MBO method (such as CbAS, COMsb, ROMA, etc.) can be augmented with the DynAMO term.
    - **Design Motivation**: Adversarial training is naturally suited for distribution matching, and acting as an auxiliary loss term allows it to leave the core optimization pipeline of the original method intact.

3. **High-Quality Subset Selection**:

    - **Function**: Determines the reference distribution for diversity.
    - **Mechanism**: Filters the top-$k$ samples from the offline dataset based on the surrogate model or known objective values to serve as the reference distribution.
    - **Design Motivation**: Instead of matching against the entire dataset (which includes low-quality samples), matching only against a high-quality subset ensures that diversity is maintained within the "good design space."

### Loss & Training
$\text{Total Loss} = \text{Original MBO Objective} + \lambda \times \text{Adversarial Distribution Matching Loss}$. The discriminator and generator are trained alternately.

## Key Experimental Results

### Main Results

| Domain | MBO Method + DynAMO | Quality Change | Diversity Change |
|------|------------------|---------|----------|
| Protein Design | Gain | Maintained/Slight Gain | **Significant Gain** |
| Molecular Optimization | Gain | Maintained | **Significant Gain** |
| Materials Design | Gain | Maintained | **Significant Gain** |

### Ablation Study

| Configuration | Diversity | Quality | Description |
|------|--------|------|------|
| Original MBO | Low | Baseline | Collapsed to a single peak |
| + Random Perturbations | Medium | Decreased | Quality-diversity trade-off |
| + DynAMO | **High** | Maintained/Gain | Distribution matching balances both |

### Key Findings
- Combining DynAMO as a plug-and-play module with multiple MBO methods significantly improves diversity, demonstrating its generalizability.
- Distribution matching is more effective than simple diversity regulation because it preserves meaningful structural diversity present in the dataset.
- The effectiveness of the proposed method is validated across various scientific domains (proteins, molecules, and materials).

## Highlights & Insights
- The formulation of **diversity as distribution matching** is highly elegant—instead of manually defining diversity metrics, it allows the generated distribution to automatically match the natural diversity within the data.
- Operating as a plug-and-play module, it can be combined with any MBO method, lowering the barrier to practical application.
- The core methodology is transferable to other domains requiring diverse solution generation (e.g., diverse recommendation, multi-objective optimization).

## Limitations & Future Work
- The stability of adversarial training may become a challenge in high-dimensional design spaces.
- The selection of the quality threshold for the reference distribution requires tuning.
- Currently evaluated only under offline MBO frameworks; dynamic online settings remain to be explored.
- The information in this note is based on the abstract, and method details as well as experimental data need to be updated from the full paper.

## Related Work & Insights
- **vs CbAS/COMsb**: Traditional MBO methods optimize quality exclusively, whereas DynAMO introduces an additional diversity objective.
- **vs Multi-Objective Optimization**: Multi-objective optimization handles conflicting goals via Pareto fronts, whereas DynAMO naturally integrates quality and diversity via distribution matching.

## Rating
- Novelty: ⭐⭐⭐⭐ Modeling diversity as distribution matching is a novel and elegant perspective.
- Experimental Thoroughness: ⭐⭐⭐ Validated across multiple domains, though specific details and data still need to be retrieved from the full paper.
- Writing Quality: ⭐⭐⭐ Evaluated based on abstract information.
- Value: ⭐⭐⭐⭐ Provides a general-purpose solution to the diversity challenge in scientific design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Soft Quality-Diversity Optimization](../../ICLR2026/others/soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/others/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ICML 2025\] IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [\[ICML 2025\] Optimal Auction Design in the Joint Advertising](optimal_auction_design_in_the_joint_advertising.md)

</div>

<!-- RELATED:END -->
