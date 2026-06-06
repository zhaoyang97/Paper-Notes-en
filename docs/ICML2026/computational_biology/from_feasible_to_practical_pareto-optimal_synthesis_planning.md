---
title: >-
  [Paper Note] From Feasible to Practical: Pareto-Optimal Synthesis Planning
description: >-
  [ICML 2026][Computational Biology][Multi-objective Search] PareSP employs **multi-objective MCTS search** to jointly optimize synthesis paths for **cost / time / feasibility / environmental impact**—identifying the compl…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Multi-objective Search"
  - "Pareto Optimization"
  - "Synthesis Planning"
  - "MCTS"
date: 2026-05-08
content_hash: e388fe123a9d712c
---

# From Feasible to Practical: Pareto-Optimal Synthesis Planning

**Conference**: ICML 2026  
**arXiv**: [2605.29113](https://arxiv.org/abs/2605.29113)  
**Code**: To be confirmed  
**Area**: Optimization / Chemical Synthesis Planning / Multi-objective Search  
**Keywords**: Multi-objective Search, Pareto Optimization, Synthesis Planning, MCTS

## TL;DR
PareSP employs **multi-objective MCTS search** to jointly optimize synthesis paths for **cost / time / feasibility / environmental impact**—identifying the complete Pareto front rather than a single "optimal" path. On USPTO and ASKCOS benchmarks, it reduces costs by 23% and time by 35% compared to single-objective methods while maintaining $\geq 95\%$ chemical feasibility.

## Background & Motivation

**Background**: Computer-Aided Synthesis Planning (CASP) aims to identify economically viable multi-step reaction paths for target molecules. Traditional methods (e.g., EFMC, Retro*) optimize for a single objective (chemical feasibility or shortest path), but real-world scenarios require balancing conflicting objectives such as cost, time, and environmental impact.

**Limitations of Prior Work**: (1) Single-objective MCTS tends to converge on "optimal" paths while ignoring balanced solutions; (2) Post-processing re-ranking cannot guarantee Pareto optimality; (3) Multi-objective methods (e.g., NSGA-II) require evaluation of the entire space, which is infeasible for combinatorial search spaces.

**Key Challenge**: Synthesis planning is inherently a **combinatorial search + multi-objective trade-off** problem, yet existing methods sacrifice either diversity (single-objective) or scalability (brute-force multi-objective).

**Goal**: To find the Pareto front—all "non-dominated" trade-off solutions—within the synthesis path search.

**Key Insight**: The capability of MCTS to balance exploration and exploitation combined with dominance relations in multi-objective optimization leads to Multi-Objective MCTS (MO-MCTS).

**Core Idea**: The MCTS UCT formula is extended to a multi-objective setting where each node maintains a **Pareto front** instead of a single value, and search is guided by dominance relations and hypervolume.

## Method

### Overall Architecture
(1) **Objective Definition**: Four objectives $(c, t, f, e)$ = (Cost, Time, Feasibility, Environmental impact); (2) **MO-MCTS Search Tree**: Each node stores a Pareto front $\mathcal{P}_n$; (3) **Multi-objective UCT Selection**: Priority is given to expanding non-dominated candidates based on dominance relations; (4) **Backpropagation**: After expanding a node, all ancestor Pareto fronts are updated with the new solution; (5) **Output**: The Pareto front of the root node is returned as the set of all "optimal" trade-off paths upon search completion.

### Key Designs

1. **Multi-objective Value Representation + Pareto Front Maintenance**:
    - Function: Maintains the complete Pareto front within each node of the search tree.
    - Mechanism: The value of node $n$ is not a scalar but a set $\mathcal{P}_n = \{(c_i, t_i, f_i, e_i)\}_i$ (non-dominated solutions). A new solution $\mathbf{v}^*$ is added based on dominance: if $\exists \mathbf{v} \in \mathcal{P}_n: \mathbf{v} \succeq \mathbf{v}^*$, it is discarded; otherwise, solutions dominated by $\mathbf{v}^*$ are removed and $\mathbf{v}^*$ is added.
    - Design Motivation: Traditional scalar values compress multi-objective information; maintaining the Pareto front preserves the trade-off space and diversifies the final output.

2. **Dominance-Aware UCT Selection Strategy**:
    - Function: Balances exploration and multi-objective exploitation during the MCTS selection phase.
    - Mechanism: The UCT formula is extended as $\text{UCT}(n) = HV(\mathcal{P}_n, \mathbf{r}_{\text{ref}}) + c \sqrt{\ln N(p) / N(n)}$, where $HV(\cdot, \mathbf{r}_{\text{ref}})$ is the hypervolume based on a reference point $\mathbf{r}_{\text{ref}}$, and $c = \sqrt{2}$ aligns with single-objective MCTS. Child nodes with high hypervolume and low visit counts are prioritized.
    - Design Motivation: Hypervolume measures both the quality and diversity of the Pareto front; the UCT exploration reward ensures comprehensive search coverage.

3. **Chemical Prior Fusion + Enhanced Local Search**:
    - Function: Incorporates chemical knowledge (reaction template feasibility, starting material costs, etc.) as priors into value estimation.
    - Mechanism: Feasibility $f$ is given as a probability by a neural reaction prediction model; cost $c$ is queried from starting material databases; time $t$ is estimated based on reaction steps and temperature; environmental impact $e$ is based on green chemistry metrics (PMI, E-factor). Search at leaf nodes is supplemented by a Localized Reaction Suggestion Network (LRSN).
    - Design Motivation: Pure MCTS is inefficient due to random expansion in chemical space; chemical priors significantly reduce invalid branches.

## Key Experimental Results

### Main Results: Single-objective vs Multi-objective

| Dataset | Method | Avg Cost | Avg Time | Avg Feasibility | PMI | Pareto Size |
|--------|------|---------|---------|----------|-----|-----------|
| USPTO-50K | Retro* | $52.3 | 8.7h | 92.1% | 18.4 | 1 |
| USPTO-50K | EFMC | $48.7 | 9.2h | 94.5% | 16.8 | 1 |
| **USPTO-50K** | **Ours** | **$40.1** | **5.6h** | **95.3%** | **12.7** | **8.4** |
| ASKCOS-100 | Retro* | $124.6 | 22.4h | 88.7% | 24.1 | 1 |
| ASKCOS-100 | EFMC | $115.3 | 19.8h | 91.2% | 22.6 | 1 |
| **ASKCOS-100** | **Ours** | **$95.7** | **14.5h** | **96.4%** | **15.8** | **12.7** |

### Pareto Front Diversity

| Target Molecule | Pareto Solutions | Lowest Cost | Fastest Time | Highest Feasibility | Greenest |
|---------|-----------|---------|---------|----------|--------|
| Aspirin | 6 | $3.2 | 1.2h | 99.5% | PMI=4.8 |
| Sildenafil | 11 | $89.4 | 12.3h | 96.7% | PMI=18.2 |
| Imatinib | 14 | $124.7 | 16.8h | 94.2% | PMI=24.1 |

### Ablation Study

| Configuration | Avg Cost | Pareto Size | Search Time |
|------|---------|-----------|---------|
| Single-objective MCTS (Cost) | $42.1 | 1 | 5.2 min |
| Single-objective MCTS (Feasibility) | $58.9 | 1 | 4.8 min |
| Multi-objective MCTS (HV-UCT) | $40.3 | 7.2 | 7.5 min |
| **PareSP Full** | **$40.1** | **8.4** | **8.1 min** |
| - w/o LRSN | $43.7 | 6.5 | 7.8 min |
| - w/o Chemical Prior | $48.2 | 4.3 | 9.4 min |

### User Study

| Chemist Preference (n=30) | Prefer PareSP | Prefer Retro* | Prefer EFMC | No Preference |
|--------------------|----------|----------|--------|--------|
| Overall Preference | **63.3%** | 16.7% | 13.3% | 6.7% |
| Industrial Synthesis | **76.7%** | 10.0% | 6.7% | 6.7% |
| Academic Research | **53.3%** | 23.3% | 16.7% | 6.7% |

### Key Findings
- **Multi-objective solutions consistently outperform single-objective**: Average costs decreased by 23%, time decreased by 35%, and feasibility actually improved.
- **Pareto front provides decision flexibility**: Chemists can select paths based on specific context.
- **Critical contribution of chemical priors**: Search efficiency improved by 16%.
- **Hypervolume UCT effectiveness**: Achieves the best balance between search time and diversity.

## Highlights & Insights
- **Elegant Application of Multi-objective Search**: MO-MCTS is well-suited for discrete combinatorial search and multi-objective trade-offs in synthesis planning.
- **Fusion of Chemical Priors and Search Algorithms**: Avoids the "hallucinations" of pure learning methods and the "blindness" of pure search.
- **Practical Design**: The four objectives cover the core trade-offs in industrial synthesis; the user study confirms preferences of practical chemists.
- **Interpretable and Diverse Output**: The complete Pareto front empowers user decision-making rather than providing black-box recommendations.

## Limitations & Future Work
- Objective Scalability: Currently handles four objectives; the Pareto front may explode in higher dimensions.
- Multi-step Uncertainty: Costs and times for each step are currently estimates.
- Capturing Chemist Preferences: The user study sample size (n=30) is relatively small.
- Future Work: Exploration of higher-dimensional multi-objective search; introduction of active learning to update chemist preferences; extension to biosynthesis.

## Related Work & Insights
- **vs Retro* / EFMC**: These use single-objective methods with post-processing; this work utilizes direct multi-objective search.
- **vs NSGA-II**: Population evolution is suitable for continuous spaces, whereas MCTS is better for discrete combinatorial spaces.
- **vs RL-based CASP**: RL requires extensive training data, while MCTS provides flexible, plug-and-play search.
- **Insight**: MO-MCTS can be extended to other combinatorial optimization scenarios like drug design and materials discovery.

## Rating
- Novelty: ⭐⭐⭐⭐ While MO-MCTS exists, the innovation lies in the domain application, chemical prior fusion, and practical implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive cross-dataset testing, multiple baselines, Pareto analysis, user studies, and detailed ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed algorithm description, and strong conclusions.
- Value: ⭐⭐⭐⭐⭐ High industrial value for chemical synthesis; provides the decision flexibility essential for chemists.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Amortized Active Generation of Pareto Sets](../../NeurIPS2025/computational_biology/amortized_active_generation_of_pareto_sets.md)
- [\[ACL 2026\] ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](../../ACL2026/computational_biology/protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design.md)
- [\[ICML 2026\] Influence-Guided Symbolic Regression: Scientific Discovery via LLM-Driven Equation Search with Granular Feedback](influence-guided_symbolic_regression_scientific_discovery_via_llm-driven_equatio.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](protein_autoregressive_modeling_via_multiscale_structure_generation.md)
- [\[ICML 2026\] SwitchCraft: A Programmatic Framework for Designing State-Switching Proteins](switchcraft_a_programmatic_framework_for_designing_state-switching_proteins.md)

</div>

<!-- RELATED:END -->
