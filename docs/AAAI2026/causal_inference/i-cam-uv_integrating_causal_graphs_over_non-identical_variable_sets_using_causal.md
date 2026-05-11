---
title: >-
  [Paper Note] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables
description: >-
  [AAAI 2026][Causal Inference][causal discovery] This paper proposes I-CAM-UV, a method that enumerates consistent DAGs satisfying structural constraints derived from multiple CAM-UV outputs over non-identical variable se…
tags:
  - "AAAI 2026"
  - "Causal Inference"
  - "causal discovery"
  - "causal additive model"
  - "unobserved variables"
  - "non-identical variable sets"
  - "graph integration"
  - "combinatorial search"
date: 2026-05-08
content_hash: 8723b93859323027
---

# I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables

**Conference**: AAAI 2026
**arXiv**: [2603.03207](https://arxiv.org/abs/2603.03207)
**Code**: Not released
**Area**: Causal Inference
**Keywords**: causal discovery, causal additive model, unobserved variables, non-identical variable sets, graph integration, combinatorial search

## TL;DR

This paper proposes I-CAM-UV, a method that enumerates consistent DAGs satisfying structural constraints derived from multiple CAM-UV outputs over non-identical variable sets, recovering causal relations lost due to unobserved variables, and introduces an optimal-first search algorithm exploiting cost monotonicity for efficient combinatorial search.

## Background & Motivation

- **Practical demand for causal discovery**: Identifying causal relations from observational data is a fundamental scientific task; randomized controlled trials are often infeasible due to cost or ethical constraints, making observational causal discovery increasingly important.
- **Prevalence of multi-dataset scenarios**: In practice, multiple datasets often share the same research targets but observe different variable subsets (e.g., different laboratories measuring different indicators), making their integration a key challenge.
- **Unobserved confounders**: Even with a moderate number of variables (around 10), unobserved confounders remain common, yet most existing methods assume causal sufficiency.
- **Limitations of naive overlap approaches**: Estimating causal graphs independently per dataset and overlapping them leads to missed true causal relations due to confounding by unobserved variables; moreover, some variable pairs are never jointly observed across any dataset.
- **Shortcomings of existing methods**: ION/IOD/COmbINE output PAGs (containing uncertainty), and CD-MiNi supports only linear non-Gaussian models, neither handling nonlinear causal relations.
- **New opportunity via CAM-UV**: CAM-UV identifies unobserved causal paths (UCPs) and unobserved backdoor paths (UBPs) arising from unobserved variables, providing structural consistency constraints that enable multi-dataset causal graph integration.

## Method

### Overall Architecture

Given $m$ datasets $V_1, \dots, V_m \subset V$ over non-identical variable sets, CAM-UV is first applied to each dataset to obtain a mixed graph $G_k = (V_k, A_k, N_k)$ containing directed and undirected edges. I-CAM-UV then enumerates all DAGs satisfying consistency constraints. The core pipeline is: (1) overlay all directed edges to obtain $\hat{G}$; (2) collect the set of undetermined edges $E = E_{\text{imp}} \cup E_{\text{uno}}$ (edges unidentified due to UCP/UBP and variable pairs never jointly observed); (3) assign directions or exclude each edge in $E$ to enumerate all consistent DAGs.

### Key Design 1: Consistency Definition Based on UCP/UBP

- **Function**: Defines structural consistency between a candidate DAG and CAM-UV outputs.
- **Mechanism**: For each dataset $V_k$, variable pairs in the identified set $I_k$ should have neither UCP nor UBP between them in the candidate DAG, while variable pairs connected by undirected edges $N_k$ must have either a UCP or UBP. The candidate DAG must satisfy consistency with all $m$ datasets simultaneously.
- **Design Motivation**: Undirected edges in CAM-UV precisely reflect the existence of unobserved paths, and exploiting this structural information constrains the candidate solution space. Theorem 1 proves that under ideal conditions (full variable coverage, error-free CAM-UV), the true causal graph is always consistent.

### Key Design 2: Inconsistency Cost Relaxation

- **Function**: Introduces an inconsistency cost $C(\tilde{G})$ to relax the problem from exact consistency enumeration to approximate consistency enumeration.
- **Mechanism**: The total number of variable pairs violating consistency across all datasets is used as the cost; all DAGs with inconsistency cost $\leq C^* + b$ are enumerated, where $C^*$ is the minimum cost and $b$ is a user-specified parameter.
- **Design Motivation**: In practice, CAM-UV may produce estimation errors (e.g., spurious undirected edges or missing directed edges), making strict consistency infeasible. The relaxation allows recovery of the true DAG even under estimation errors.

### Key Design 3: Optimal-First Search Based on Monotonicity

- **Function**: Designs an efficient combinatorial search algorithm to enumerate all DAGs satisfying the cost constraint.
- **Mechanism**: A cost lower bound function $\tilde{C}(t, \tilde{G})$ is defined and shown to be monotone during search (Theorem 2), enabling a priority queue (heap) to search in ascending cost order and terminate early once the dequeued state's cost exceeds $C^* + b$.
- **Design Motivation**: Exhaustively evaluating all $3^{|E|}$ connection patterns is infeasible. Monotonicity enables substantial pruning of the search space. UCP/UBP existence can be determined via BFS in $O(|A|)$ time, ensuring efficient evaluation of each search state.

### Key Design 4: Polynomial-Time UCP/UBP Detection

- **Function**: Designs fast sub-algorithms for detecting UCPs and UBPs.
- **Mechanism**: For UCP, BFS is launched from $v_i$ to check reachability to unobserved parents of $v_j$; for UBP, ancestor sets $S_i, S_j$ of the unobserved parents of $v_i$ and $v_j$ are computed, and $S_i \cap S_j \neq \emptyset$ is checked.
- **Design Motivation**: UCP/UBP existence must be evaluated repeatedly for each search state; $O(|A|)$ detection ensures the overall time efficiency of the search.

## Loss & Training

This paper presents a non-parametric method without neural network training. The overall pipeline is: (1) independently run the CAM-UV estimation algorithm (including regression analysis and independence testing) on each dataset; (2) collect the overlaid graph and undetermined edge set; (3) run combinatorial search to enumerate consistent DAGs. The core optimization objective is to minimize the inconsistency cost $C(\tilde{G})$.

## Key Experimental Results

### Experimental Setup

100 synthetic datasets were generated using the Erdős–Rényi random graph model (10 variables, edge probability 0.3, nonlinear functions), constructing $m \in \{2, 3\}$ sub-datasets with non-identical variable sets, each containing $|U| \in \{3, 4\}$ unobserved variables and 1000 samples.

### Table 1: F1 Score Comparison Across Methods and Settings

| Method | $(m{=}2, \|U\|{=}3)$ | $(m{=}2, \|U\|{=}4)$ | $(m{=}3, \|U\|{=}3)$ | $(m{=}3, \|U\|{=}4)$ |
|------|---------|---------|---------|---------|
| I-CAM-UV | **Highest recall** | **Highest recall** | **Highest recall** | **Highest recall** |
| CAM-UV-OVL | F1 close to I-CAM-UV | F1 close to I-CAM-UV | Below I-CAM-UV | Below I-CAM-UV |
| PC-OVL | Low | Low | Low | Low |
| Imputation | Low | Low | Low | Low |
| CD-MiNi | Low | Low | Low | Low |

I-CAM-UV significantly outperforms all competing methods in recall across all settings (including observed and unobserved variable pairs), though its precision is slightly lower than CAM-UV-OVL, with overall F1 scores being comparable between the two.

### Table 2: Statistics on Number of Enumerated DAGs and Computation Time for I-CAM-UV

| Metric | $(m{=}2, \|U\|{=}3)$ | $(m{=}2, \|U\|{=}4)$ | $(m{=}3, \|U\|{=}3)$ | $(m{=}3, \|U\|{=}4)$ |
|------|---------|---------|---------|---------|
| Median DAG count | Small | Moderate | Moderate | Large |
| Median enumeration time | <1s | Several seconds | Several seconds | Tens of seconds |
| Maximum enumeration time | Outliers up to hundreds of seconds | Outliers larger | — | — |

The number of enumerated DAGs varies widely (from 1 to thousands), but the precision distribution shows a unimodal clustering pattern, indicating that most enumerated DAGs are of comparable quality. The optimal-first search yields acceptable computation time on most instances.

## Highlights & Insights

- **Novel problem formulation**: This is the first work to leverage UCP/UBP structural information from CAM-UV for causal graph integration over non-identical variable sets, enabling identification of causal relations—especially causal directions between variable pairs never jointly observed—that are entirely undetectable by naive overlay approaches.
- **Solid theoretical guarantees**: The paper proves consistency of the true DAG under ideal conditions (Theorem 1), monotonicity of the cost function (Theorem 2), and correctness of the algorithm.
- **Elegant algorithm design**: Optimal-first search combined with monotonicity-based pruning and polynomial-time UCP/UBP detection effectively reduces an exponential search space.
- **Strong practical relevance**: Non-identical variable sets across multiple datasets is a common scenario in real-world scientific research.

## Limitations & Future Work

- The worst-case time complexity remains exponential $O(3^{|E|})$, rendering the algorithm potentially infeasible for large variable counts or large $|E|$.
- Accuracy is highly dependent on the quality of the underlying CAM-UV estimates, whose theoretical completeness has not been fully established.
- The method may output a large number of DAGs, making manual inspection impractical; effective compact representations or ranking mechanisms are lacking.
- Validation is limited to synthetic data; experiments on real-world datasets are absent.
- Comparison with a broader range of nonlinear causal discovery methods is not conducted.

## Related Work & Insights

- **Multi-dataset causal discovery**: ION (Tillman et al. 2009), IOD (Tillman & Spirtes 2011), and COmbINE (Triantafillou et al. 2010) are constraint-based methods outputting PAGs rather than complete DAGs; CD-MiNi (Huang et al. 2020) is based on LiNGAM and handles only linear settings.
- **Causal additive models**: CAM (Bühlmann et al. 2014) assumes nonlinear additive functional relationships; CAM-UV (Maeda & Shimizu 2021) extends CAM to allow unobserved variables; Pham et al. (2025) improve CAM-UV estimation accuracy.
- **Functional model methods**: LiNGAM (Shimizu et al. 2006) addresses linear non-Gaussian settings; ANM (Hoyer et al. 2009) handles general additive noise models.

## Rating

- Novelty: ⭐⭐⭐⭐ — Leveraging UCP/UBP information from CAM-UV for multi-variable-set integration is an entirely new direction.
- Experimental Thoroughness: ⭐⭐⭐ — Synthetic experiments are well-designed but lack real-world data validation.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, examples are abundant, and algorithm descriptions are rigorous.
- Value: ⭐⭐⭐⭐ — Addresses a practically meaningful open problem, though scalability remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)

</div>

<!-- RELATED:END -->
