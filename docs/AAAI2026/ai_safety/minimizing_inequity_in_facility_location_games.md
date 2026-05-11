---
title: >-
  [Paper Note] Minimizing Inequity in Facility Location Games
description: >-
  [AAAI 2026][AI Safety][Facility Location] This paper studies the problem of minimizing the Maximum Group Effect in facility location games on the real line…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Facility Location"
  - "Inter-group Fairness"
  - "Strategyproofness"
  - "Approximation Ratio"
  - "Social Choice"
date: 2026-05-08
content_hash: 7342e6485ba31134
---

# Minimizing Inequity in Facility Location Games

**Conference**: AAAI 2026
**arXiv**: [2602.01048](https://arxiv.org/abs/2602.01048)
**Code**: None
**Area**: AI Safety / Algorithmic Game Theory
**Keywords**: Facility Location, Inter-group Fairness, Strategyproofness, Approximation Ratio, Social Choice

## TL;DR
This paper studies the problem of minimizing the Maximum Group Effect in facility location games on the real line, proposing two strategyproof mechanisms—BALANCED and MAJOR-PHANTOM—that achieve tight approximation ratios in the single-facility setting. The framework unifies classical objectives (utilitarian social cost, egalitarian maximum cost) with group fairness objectives, and extends the endpoint mechanism to the two-facility setting.

## Background & Motivation

**Background**: Facility location games are a classical problem in social choice theory, concerning the selection of facility positions on a line to serve agents distributed at various locations. Agents may strategically misreport their positions for personal gain. Classical strategyproof mechanisms (e.g., the median mechanism) optimize utilitarian objectives (minimizing total distance) or egalitarian objectives (minimizing maximum distance).

**Limitations of Prior Work**: Traditional objective functions overlook inter-group fairness. When agents belong to different groups (e.g., different communities or income classes), a solution optimal in total cost may impose disproportionately high costs on certain groups. Although the maximum group effect framework proposed by Marsh and Schilling (1994) provides a fairness-oriented objective, a known gap exists in the approximation guarantees for strategyproof mechanisms under group fairness objectives, as identified by Zhou, Li, and Chan (2022).

**Key Challenge**: There is a fundamental tension between strategyproofness (truthfulness) and minimizing the maximum group effect—strategyproofness constrains the feasible solution space, while the fairness objective requires delicate balancing of inter-group interests.

**Goal**: Design strategyproof mechanisms that minimize the maximum group effect and close the existing approximation ratio gap.

**Key Insight**: Unify multiple classical facility location objectives (utilitarian, egalitarian, group fairness) under the maximum group effect framework and analyze them within a single analytical framework.

**Core Idea**: Propose two novel mechanisms—BALANCED (based on weighted medians) and MAJOR-PHANTOM (based on phantom voters)—that achieve tight approximation ratios under the two formulations of total group effect and maximum group effect, respectively.

## Method

### Overall Architecture
On the real line $\mathbb{R}$, $n$ agents are partitioned into $m$ groups. Each group $g$ has a weight factor $w_g$. The group effect is defined as the sum (or maximum) of distances from group members to the nearest facility, multiplied by the group weight. The objective is to select a facility location $y$ that minimizes the maximum group effect $\max_g w_g \cdot \text{effect}_g(y)$, while the mechanism must be strategyproof (no agent benefits from misreporting their position).

### Key Designs

1. **BALANCED Mechanism (single facility, total effect)**:

    - Function: Minimizes the maximum weighted total distance across groups.
    - Mechanism: Computes the weighted median position for each group, then selects a balance point among these medians such that the weighted total effect is equalized across groups. Specifically, BALANCED outputs the position that balances the group effects on the left and right sides.
    - Design Motivation: A naive global median ignores inter-group differences, whereas BALANCED explicitly accounts for inter-group fairness by searching for a balance point among group medians.

2. **MAJOR-PHANTOM Mechanism (single facility, max effect)**:

    - Function: Minimizes the maximum weighted individual distance across groups.
    - Mechanism: Inspired by the phantom voter framework, a phantom agent is constructed for each group—its position is determined by the group weight and the distribution of group members. The median mechanism is then applied to the augmented set of real and phantom agents. The phantom positions are designed to guarantee strategyproofness.
    - Design Motivation: The standard phantom mechanism ignores group structure; MAJOR-PHANTOM achieves group-awareness by assigning distinct phantom voters to each group.

3. **Extended Endpoint Mechanism (two facilities)**:

    - Function: Minimizes the maximum group effect when two facilities must be placed.
    - Mechanism: Extends the classical endpoint mechanism (placing facilities at the two extremes of the agent distribution) to the group fairness setting. A carefully designed group-aware endpoint selection strategy achieves tight approximation ratios under both maximum group effect objectives.
    - Design Motivation: Guaranteeing strategyproofness in the multi-facility setting is significantly harder; the endpoint mechanism is one of the few known strategyproof mechanisms for two facilities.

### Theoretical Analysis
Strategyproofness of all mechanisms is established via standard game-theoretic arguments (single-peaked preferences combined with fixed rules imply strategyproofness). Approximation ratios are derived through worst-case analysis, and matching lower bounds are constructed to confirm tightness.

## Key Experimental Results

### Main Results: Approximation Ratios

| Setting | Objective | Mechanism | Approx. Ratio | Lower Bound | Status |
|---|---|---|---|---|---|
| Single facility | max total group effect | BALANCED | $1 + \frac{w_{\max}}{w_{\min}}$ | $1 + \frac{w_{\max}}{w_{\min}}$ | Tight |
| Single facility | max-max group effect | MAJOR-PHANTOM | $1 + 2\frac{w_{\max}}{w_{\min}}$ | $1 + 2\frac{w_{\max}}{w_{\min}}$ | Tight |
| Two facilities | max total group effect | Extended Endpoint | $n-1$ | $n-1$ | Tight |
| Two facilities | max-max group effect | Extended Endpoint | $\frac{n}{2}$ | $\frac{n}{2}$ | Tight |

### Unification with Classical Objectives

| Classical Objective | Special Case | Corresponding Weight Setting |
|---|---|---|
| Utilitarian (social cost) | $m=1$ group | $w_1 = 1$ |
| Egalitarian (maximum cost) | $m=n$ (one agent per group) | $w_g = 1, \forall g$ |
| Max total group cost | Equal weights | $w_g = 1/|g|, \forall g$ |
| Max average group cost | Normalized equal weights | $w_g = 1, \forall g$ |

### Closed Approximation Ratio Gaps

| Objective | Zhou et al. (2022) Upper Bound | Zhou et al. (2022) Lower Bound | Ours |
|---|---|---|---|
| Max total group cost | 3 | 2 | **2 (tight)** |
| Max avg group cost | $1+n/2$ | Unknown | **$1+n/2$ (tight)** |

### Key Findings
- **Fully closes the open gap of Zhou et al. (2022)**: The approximation ratio for max total group cost is precisely determined to be 2, resolving the prior range of $[2, 3]$.
- **Unified framework**: Utilitarian, egalitarian, and group fairness objectives—previously treated as distinct—are unified under the maximum group effect formulation, with different weight settings recovering different classical objectives.
- **Tight approximation ratios**: All proposed mechanisms admit matching lower bound constructions, establishing that no further improvement is possible under the strategyproofness constraint.
- **BALANCED and MAJOR-PHANTOM generalize classical mechanisms**: Under specific weight configurations, they reduce to the standard median and phantom mechanisms, respectively.

## Highlights & Insights
- **Elegant unifying framework**: A single parameterized objective—the maximum group effect—subsumes multiple classical objectives in social choice theory, with different weight configurations recovering distinct targets. This unified perspective provides a clear analytical foundation for future research.
- **Tight approximation bounds**: All results feature matching upper and lower bounds, which is particularly valuable in mechanism design—establishing not only what is achievable but also precisely what cannot be improved upon.
- **Practical relevance**: The results are directly applicable to urban planning and public facility allocation. When different communities have different service-demand weights, the BALANCED mechanism provides a principled fairness guarantee.

## Limitations & Future Work
- The analysis is restricted to the real line (one-dimensional setting); higher-dimensional facility location is more realistic but significantly harder to make strategyproof.
- The approximation ratio of $O(n)$ in the two-facility setting may be unsatisfactory when the number of agents is large.
- Group weights $w_g$ are assumed to be common knowledge and immune to manipulation; if weights can themselves be strategically reported, the problem changes fundamentally.
- Only deterministic mechanisms are considered; randomized mechanisms may achieve better approximation ratios.

## Related Work & Insights
- **vs. Zhou, Li, Chan (2022)**: Introduced the group fairness facility location problem but left approximation ratio gaps open; the present paper fully closes these gaps.
- **vs. Procaccia & Tennenholtz (2013)**: Established the classical single-facility mechanism design framework; the present paper extends it to the group fairness setting.
- **vs. Moulin (1980)**: The standard phantom mechanism does not account for group structure; MAJOR-PHANTOM extends this classical framework through group-aware phantom voters.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified framework perspective is novel, and the two new mechanisms represent solid contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ All theoretical results are accompanied by tight lower bounds, yielding theoretical completeness.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous, though accessibility for readers outside algorithmic game theory is limited.
- Value: ⭐⭐⭐⭐ Closes an open problem, unifies classical results, and makes a significant theoretical contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Facility Location Mechanisms for Envy Ratio](../../NeurIPS2025/ai_safety/learning-augmented_facility_location_mechanisms_for_envy_ratio.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)
- [\[AAAI 2026\] Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)
- [\[AAAI 2026\] Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation](plug-and-play_parameter-efficient_tuning_of_embeddings_for_federated_recommendat.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)

</div>

<!-- RELATED:END -->
